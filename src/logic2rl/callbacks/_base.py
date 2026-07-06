"""BaseCallback — the lifecycle interface every training callback shares.

The manager injects ``self.model`` (the algorithm) via ``init_callback`` before
training; hooks read training state off it — ``model.num_timesteps`` (global
step), ``model.iteration``, and ``model.last_metrics`` (the single shared metrics
dict). All hooks are no-ops by default; a callback overrides only the ones it
needs (so the manager never has to probe for them).

``on_iteration_end`` returns ``True`` to keep training, ``False`` to stop
(early-stop / pruning). Metric *producers* (the metrics/eval/ranking callbacks)
write into ``model.last_metrics``; *consumers* (checkpoint, early-stop) read it —
build_callbacks orders producers before consumers.

The callback manager is itself a ``BaseCallback`` (the Composite) that fans each
hook out to its children.
"""
from typing import Any, Dict, List


class BaseCallback:
    """Lifecycle hooks for a training callback (all no-ops by default)."""

    model: Any = None

    def init_callback(self, model: Any) -> None:
        """Bind the algorithm (the ``model``). Called once before training."""
        self.model = model

    def on_training_start(self, total_timesteps: int = 0) -> None:
        ...

    def on_iteration_start(self) -> None:
        ...

    def on_step(self, infos: List[Dict[str, Any]]) -> None:
        ...

    def on_iteration_end(self) -> bool:
        """Return ``False`` to stop training (early-stop / pruning), else ``True``."""
        return True

    def on_training_end(self) -> None:
        ...
