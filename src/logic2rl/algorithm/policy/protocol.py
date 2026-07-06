"""The policy contract the PPO stack depends on.

PPO never type-checks its policy — the rollout / loss / compiled-eval stack just calls a
fixed method surface (duck-typed). ``Policy`` names that surface, so the two implementations
— the neural ``ActorCriticPolicy`` and the scored ``QKGEPolicy`` — read as two siblings of
one contract instead of an accidental match.

It is a structural ``typing.Protocol``: a class conforms by implementing the members, with no
inheritance required (both already do). ``@runtime_checkable`` so a freshly built policy can be
asserted against it once at construction.

Out of scope here (part of the de-facto contract, but not re-declared):

- ``torch.nn.Module`` membership — PPO builds the optimizer from ``policy.parameters()`` and
  checkpoints via ``state_dict()`` / ``load_state_dict()`` / ``to()`` / ``train()`` / ``eval()``.
  Implementers are expected to be ``nn.Module`` subclasses.
- the SB3-style predict API (``get_distribution`` / ``_predict`` / ``predict``) — both policies
  implement it, but it is self-contained (only called from inside each policy).
"""
from __future__ import annotations

from typing import Any, Protocol, Tuple, runtime_checkable

from torch import Tensor


@runtime_checkable
class Policy(Protocol):
    """Method surface the PPO rollout, loss, and compiled-eval stack call on a policy.

    ``obs`` is the env's batch-first observation ``TensorDict``; ``B`` envs, ``G`` action slots.
    """

    def forward(self, obs: Any, deterministic: bool = ...) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample (or argmax) one step. Invoked as ``policy(obs)``. Returns
        ``(actions [B], values [B], log_probs [B])`` — SB3 tuple order."""
        ...

    def get_logits(self, obs: Any) -> Tensor:
        """``action_logits [B, G]`` — logits only (the compile primitive)."""
        ...

    def predict_values(self, obs: Any) -> Tensor:
        """Per-env state-value baseline ``V(s_t) [B]`` (feeds GAE)."""
        ...

    def evaluate_actions(self, obs: Any, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Re-score ``actions`` under the current parameters → ``(values, log_probs, entropy)``."""
        ...

    def prepare_step(self, obs: Any) -> None:
        """Eager per-step hook the rollout runs before the compiled forward (may be a no-op)."""
        ...


__all__ = ["Policy"]
