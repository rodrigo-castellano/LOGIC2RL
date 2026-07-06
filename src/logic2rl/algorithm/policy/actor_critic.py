"""ActorCriticPolicy: main policy class."""
import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
from stable_baselines3.common.distributions import CategoricalDistribution
from tensordict import TensorDict

from .base import BasePolicy
from .network import SharedPolicyValueNetwork


class ActorCriticPolicy(BasePolicy):
    """Main Policy class orchestrating extraction, architecture, and action distribution.

    The neural ``BasePolicy``: a learned actor + value head sharing one trunk. Overrides the
    compute hooks (``_logits_and_values`` / ``get_logits`` / ``predict_values``); the
    distribution plumbing and ``Policy`` contract come from ``BasePolicy``.
    """

    def __init__(self, features_extractor, hidden_dim: int, num_layers: int,
                 device: torch.device, action_dim: int = None,
                 parity: bool = False, separate_value_network: bool = False,
                 value_head_scale: float = 1.0,
                 temperature: Optional[float] = None, use_l2_norm: bool = False,
                 sqrt_scale: bool = False, learnable_temperature: bool = False,
                 obs_body_layers: Optional[int] = 8,
                 obs_head_layers: Optional[int] = 2,
                 action_body_layers: Optional[int] = 8,
                 action_head_layers: Optional[int] = 2,
                 shared_policy_body: bool = False,
                 shared_policy_head: bool = False, **kwargs):
        super().__init__()
        self.device = device
        self.features_extractor = features_extractor  # built outside (embedder → extractor)

        # Parity mode reproduces SB3 nn.Linear-compatible init; production uses He init.
        init = "linear" if parity else "relu"
        self.mlp_extractor = SharedPolicyValueNetwork(
            embed_dim=features_extractor.embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            temperature=temperature,
            use_l2_norm=use_l2_norm,
            sqrt_scale=sqrt_scale,
            separate_value_network=separate_value_network,
            value_head_scale=value_head_scale,
            learnable_temperature=learnable_temperature,
            obs_body_layers=obs_body_layers,
            obs_head_layers=obs_head_layers,
            action_body_layers=action_body_layers,
            action_head_layers=action_head_layers,
            shared_policy_body=shared_policy_body,
            shared_policy_head=shared_policy_head,
            init=init,
        )

        # Action distribution
        self.action_dist = CategoricalDistribution(action_dim)

        # SB3 Parity Scaffold: Dummy layers to match RNG consumption during initialization. Removed in final version
        if parity:
            self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_dim if action_dim else 1)
            self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        self.to(device)

    def _logits_and_values(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Actor + critic in one shared-trunk pass → ``(logits [B, G], values [B])``.
        Drives the inherited ``forward`` / ``evaluate_actions`` (rollout sampling + the
        PPO clip ratio)."""
        features = self.features_extractor(obs)
        logits, values = self.mlp_extractor(features)
        return logits, values

    def predict_values(self, obs: TensorDict) -> torch.Tensor:
        """Critic-only value prediction [B]. Skips action embedding computation."""
        return self.mlp_extractor.forward_critic(self.features_extractor.forward_critic(obs))

    def get_logits(self, obs: TensorDict) -> torch.Tensor:
        """Actor-only logits ``[B, G]`` — skips value computation. The CUDA-graph-friendly
        compile primitive."""
        return self.mlp_extractor.forward_actor(self.features_extractor.forward_actor(obs))
