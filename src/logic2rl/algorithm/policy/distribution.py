"""The categorical action distribution over the padded action slots: SB3's ``CategoricalDistribution``
(``torch.distributions.Categorical`` on the logits, the mode its argmax), without importing SB3,
whose package import pulls TensorFlow in through tensorboard (3.4 s at every start)."""
from __future__ import annotations

from typing import Optional

from torch import Tensor
from torch.distributions import Categorical


class CategoricalDistribution:
    """``proba_distribution(logits)``, then ``log_prob``, ``entropy``, ``sample`` and ``mode``."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        self.distribution: Optional[Categorical] = None

    def proba_distribution(self, action_logits: Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: Tensor) -> Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> Tensor:
        return self.distribution.entropy()

    def sample(self) -> Tensor:
        return self.distribution.sample()

    def mode(self) -> Tensor:
        return self.distribution.probs.argmax(dim=1)


__all__ = ["CategoricalDistribution"]
