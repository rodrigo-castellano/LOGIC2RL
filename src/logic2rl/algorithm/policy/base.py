"""BasePolicy: the shared PPO-policy plumbing behind ActorCriticPolicy and QKGEPolicy.

The distribution machinery (``forward`` / ``evaluate_actions``) is identical across both
policies; only the *compute* differs. BasePolicy implements that machinery once in terms of
three hooks each subclass overrides:

- ``_logits_and_values(obs) -> (logits, values)`` — the producer used by ``forward`` /
  ``evaluate_actions``. ActorCriticPolicy fuses both in one shared-trunk pass; QKGEPolicy
  computes logits from its H + frozen-V scoring and the value from the frozen V.
- ``get_logits(obs) -> logits`` — logits-only fast path (the CUDA-graph compile primitive).
- ``predict_values(obs) -> values`` — values-only fast path.

Subclasses may override ``prepare_step``. Realises the ``Policy`` contract
(``algorithms.policy.protocol``).
"""
from __future__ import annotations

from typing import Tuple

import torch
from stable_baselines3.common.distributions import CategoricalDistribution
from torch import Tensor, nn


class BasePolicy(nn.Module):
    """Shared distribution plumbing; subclasses supply the compute hooks."""

    #: Action distribution over the padded action slots, built by the subclass ``__init__``.
    action_dist: CategoricalDistribution

    # --- compute hooks (subclass overrides) ---------------------------------

    def _logits_and_values(self, obs) -> Tuple[Tensor, Tensor]:
        """``(action_logits [B, G], state_values [B])`` for one obs batch — the producer
        ``forward`` / ``evaluate_actions`` share."""
        raise NotImplementedError

    def get_logits(self, obs) -> Tensor:
        """``action_logits [B, G]`` — logits only. The CUDA-graph-friendly compile primitive."""
        raise NotImplementedError

    def predict_values(self, obs) -> Tensor:
        """Per-env state-value baseline ``V(s_t) [B]`` (feeds GAE) — values only."""
        raise NotImplementedError

    def prepare_step(self, obs) -> None:
        """No-op eager prelude before the rollout core; subclasses fill per-step caches."""

    # --- shared distribution plumbing ---------------------------------------

    def forward(self, obs, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample (or argmax) one step → ``(actions [B], values [B], log_probs [B])``."""
        logits, values = self._logits_and_values(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        actions = dist.mode() if deterministic else dist.sample()
        return actions, values, dist.log_prob(actions)

    def evaluate_actions(self, obs, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Re-score ``actions`` under the current parameters → ``(values, log_probs,
        entropy)``. Produces the PPO importance ratio."""
        logits, values = self._logits_and_values(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        return values, dist.log_prob(actions), dist.entropy()


__all__ = ["BasePolicy"]
