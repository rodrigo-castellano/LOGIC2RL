"""algorithm — reference RL algorithms with a Stable-Baselines3-faithful API.

Holds the generic ``BaseAlgorithm`` contract plus the ``policy`` and ``ppo``
packages. Domain-agnostic: imports ``logic2rl`` only, never application/KGE
specifics (those are dependency-injected by consumers).
"""
from .base import BaseAlgorithm

__all__ = ["BaseAlgorithm"]
