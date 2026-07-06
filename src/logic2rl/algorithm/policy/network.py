"""Shared policy-value network architecture with deduplicated forward paths.

Contains SharedBody (residual MLP backbone) and SharedPolicyValueNetwork
(unified architecture for policy and value estimation).
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from .layers import FusedLinearRelu, FusedLinearReluLayerNorm


class SharedBody(nn.Module):
    """Residual MLP backbone [E] -> [H] -> [H] shared by policy and value heads.

    Uses fused Linear+ReLU+LayerNorm modules for better performance.
    """
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 256, num_layers: int = 8,
                 init: str = "relu", **kwargs):
        """Initialize shared feature extraction body: input projection [E]->[H] followed
        by residual MLP blocks [H]->[H].
        """
        super().__init__()

        self.input_transform = FusedLinearReluLayerNorm(embed_dim, hidden_dim, init=init)
        self.res_blocks = nn.ModuleList([
            FusedLinearReluLayerNorm(hidden_dim, hidden_dim, init=init)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [..., E] -> Returns: features [..., H]"""
        x = self.input_transform(x)
        for block in self.res_blocks:
            x = block(x) + x
        return x


class SharedPolicyValueNetwork(nn.Module):
    """Unified network architecture for policy and value estimation.

    Supports asymmetric obs/action processing via per-side layer counts:
      - obs_body_layers / action_body_layers: residual blocks (None = Identity)
      - obs_head_layers / action_head_layers: projection layers (None = Identity)
      - shared_policy_body: share body weights between obs and actions
      - shared_policy_head: share head weights (requires shared_policy_body)
    """
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 256, num_layers: int = 8,
                 temperature: Optional[float] = None, use_l2_norm: bool = False, sqrt_scale: bool = True,
                 separate_value_network: bool = False,
                 value_head_scale: float = 1.0,
                 learnable_temperature: bool = False,
                 obs_body_layers: Optional[int] = 8,
                 obs_head_layers: Optional[int] = 2,
                 action_body_layers: Optional[int] = 8,
                 action_head_layers: Optional[int] = 2,
                 shared_policy_body: bool = False,
                 shared_policy_head: bool = False, init: str = "relu", **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.use_l2_norm = use_l2_norm
        self.sqrt_scale = sqrt_scale
        self.separate_value_network = separate_value_network
        self.value_head_scale = value_head_scale
        self.learnable_temperature = learnable_temperature
        self.init = init
        # Learnable temperature (CLIP-style): logits * exp(log_temp)
        if learnable_temperature:
            self.log_temp = nn.Parameter(torch.tensor(math.log(math.sqrt(float(embed_dim)))))

        # Compute scaled value head dimension
        value_hidden_dim = int(hidden_dim * value_head_scale)

        # Output dims: body with layers outputs H, Identity outputs E
        obs_body_out = hidden_dim if obs_body_layers is not None else embed_dim
        act_body_out = hidden_dim if action_body_layers is not None else embed_dim

        # --- Obs body ---
        self.obs_body = self._make_body(obs_body_layers, embed_dim, hidden_dim, init)
        self.shared_body = self.obs_body  # alias for external access

        # --- Action body ---
        if shared_policy_body:
            self.action_body = self.obs_body
        else:
            self.action_body = self._make_body(action_body_layers, embed_dim, hidden_dim, init)

        # --- Value backbone ---
        if separate_value_network:
            self.value_body = self._make_body(obs_body_layers, embed_dim, hidden_dim, init)
        else:
            self.value_body = self.obs_body

        self._value_hidden_dim = value_hidden_dim

        # --- Obs policy head ---
        self.obs_head = self._make_head(obs_head_layers, obs_body_out, embed_dim, init)

        # --- Action policy head ---
        if shared_policy_head:
            self.action_head = self.obs_head
        else:
            self.action_head = self._make_head(action_head_layers, act_body_out, embed_dim, init)

        # --- Value head ---
        self.value_head_fused = FusedLinearRelu(obs_body_out, value_hidden_dim, init=init)
        self.value_head_final = nn.Linear(value_hidden_dim, 1)

        # SB3 scaffold compatibility
        self.latent_dim_pi = 1
        self.latent_dim_vf = 1

    # -----------------------------------------------------------------
    # FACTORY HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def _make_body(layers: Optional[int], embed_dim: int, hidden_dim: int,
                   init: str = "relu") -> nn.Module:
        """Create a body module. None → Identity, int → SharedBody with that many residual blocks."""
        if layers is None:
            return nn.Identity()
        return SharedBody(embed_dim, hidden_dim, layers, init=init)

    @staticmethod
    def _make_head(layers: Optional[int], input_dim: int, output_dim: int,
                   init: str = "relu") -> nn.Module:
        """Create a policy head. None → Identity, int → N-layer projection to output_dim.

        For N=1: Linear(input_dim, output_dim).
        For N>=2: (N-1) × FusedLinearRelu(input_dim, input_dim) + Linear(input_dim, output_dim).
        """
        if layers is None:
            return nn.Identity()
        mods: list[nn.Module] = []
        for _ in range(layers - 1):
            mods.append(FusedLinearRelu(input_dim, input_dim, init=init))
        mods.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*mods) if len(mods) > 1 else mods[0]

    # -----------------------------------------------------------------
    # DEDUPLICATED HELPERS
    # -----------------------------------------------------------------

    def _compute_logits(self, fused: torch.Tensor, act_emb: torch.Tensor,
                        act_mask: torch.Tensor) -> torch.Tensor:
        """Compute masked action logits from fused embedding -> [B, G].

        Obs and actions may pass through different body+head paths (asymmetric).
        """
        obs_body_out = self.obs_body(fused)    # [B, 1, H]
        act_body_out = self.action_body(act_emb)  # [B, G, H]

        p_obs = self.obs_head(obs_body_out)    # [B, 1, E]
        p_act = self.action_head(act_body_out)  # [B, G, E]

        if self.use_l2_norm:
            p_obs = F.normalize(p_obs, dim=-1)
            p_act = F.normalize(p_act, dim=-1)
        logits = torch.matmul(p_obs, p_act.transpose(-2, -1)).squeeze(-2)

        # Scaling: learnable temperature takes precedence over fixed options
        if self.learnable_temperature:
            logits = logits * self.log_temp.exp().clamp(max=100.0)
        elif self.sqrt_scale:
            logits = logits / (self.embed_dim ** 0.5)
        if self.temperature is not None:
            logits = logits / self.temperature

        return logits.masked_fill(~act_mask.bool(), float("-inf"))

    def _compute_value(self, obs_emb: torch.Tensor) -> torch.Tensor:
        """Compute state value from the obs embedding -> [B]."""
        value_obs = self.value_body(obs_emb)  # [B, 1, H]
        return self.value_head_final(self.value_head_fused(value_obs)).squeeze(-1).squeeze(-1)

    # -----------------------------------------------------------------
    # PUBLIC FORWARD METHODS (thin wrappers around helpers)
    # -----------------------------------------------------------------

    def forward(self, features: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Joint forward pass for training (Actor + Critic).

        Args: TensorDict with keys: obs_emb, act_emb, action_mask.
        Returns: (logits [B, G], values [B])
        """
        obs_emb = features["obs_emb"]
        logits = self._compute_logits(obs_emb, features["act_emb"], features["action_mask"])
        values = self._compute_value(obs_emb)
        return logits, values

    def forward_actor(self, features: TensorDict) -> torch.Tensor:
        """Actor-only pass [B, G]. Skips value computation.

        Args: TensorDict with keys: obs_emb, act_emb, action_mask.
        """
        return self._compute_logits(features["obs_emb"], features["act_emb"], features["action_mask"])

    def forward_critic(self, features: TensorDict) -> torch.Tensor:
        """Critic-only pass -> [B]. Args: TensorDict with key: obs_emb."""
        return self._compute_value(features["obs_emb"])

