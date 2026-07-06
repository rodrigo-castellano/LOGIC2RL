"""Fused kernel layers for policy networks.

Provides optimized Linear+ReLU+LayerNorm modules for PPO (CUDA-graph safe).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedLinearReluLayerNorm(nn.Module):
    """Fused Linear + ReLU + LayerNorm for kernel optimization.

    ``init`` selects the weight init: ``"relu"`` (production He init, zero bias)
    or ``"linear"`` (nn.Linear-compatible kaiming_uniform a=sqrt(5) + uniform
    bias, for SB3 weight parity).
    """
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-5,
                 init: str = "relu"):
        super().__init__()
        self.out_features = out_features
        self.eps = eps
        self.init = init
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.ln_weight = nn.Parameter(torch.empty(out_features))
        self.ln_bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights. ``"linear"`` matches nn.Linear defaults; ``"relu"`` is He init."""
        if self.init == "linear":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.bias)
        nn.init.ones_(self.ln_weight)
        nn.init.zeros_(self.ln_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Linear → ReLU → LayerNorm. Input/output: [*, in_features] → [*, out_features]."""
        x = F.linear(x, self.weight, self.bias)
        x = F.relu(x, inplace=True)
        return F.layer_norm(x, (self.out_features,), self.ln_weight, self.ln_bias, self.eps)


class FusedLinearRelu(nn.Module):
    """Fused Linear + ReLU for cases without LayerNorm.

    ``init`` selects the weight init: ``"relu"`` (production He init, zero bias)
    or ``"linear"`` (nn.Linear-compatible kaiming_uniform a=sqrt(5) + uniform
    bias, for SB3 weight parity).
    """
    def __init__(self, in_features: int, out_features: int, init: str = "relu"):
        super().__init__()
        self.init = init
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights. ``"linear"`` matches nn.Linear defaults; ``"relu"`` is He init."""
        if self.init == "linear":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Linear → ReLU. Input/output: [*, in_features] → [*, out_features]."""
        return F.relu(F.linear(x, self.weight, self.bias), inplace=True)
