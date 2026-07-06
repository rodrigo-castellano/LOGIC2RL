"""TorchRLCallbackManager — the Composite callback (fans hooks out to children)."""

import logging
from typing import Any, Dict, List, Optional

from ._base import BaseCallback

logger = logging.getLogger(__name__)


class TorchRLCallbackManager(BaseCallback):
    """Composite ``BaseCallback`` holding a list of child callbacks.

    Every child is a ``BaseCallback`` with the full no-op lifecycle, so the
    manager forwards each hook unconditionally — no ``hasattr`` / signature
    introspection. ``init_callback`` binds the algorithm into every child.
    """

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        self.callbacks = list(callbacks or [])

    def add_callback(self, callback: BaseCallback) -> None:
        self.callbacks.append(callback)
        if self.model is not None:
            callback.init_callback(self.model)

    def init_callback(self, model: Any) -> None:
        self.model = model
        for cb in self.callbacks:
            cb.init_callback(model)

    def on_training_start(self, total_timesteps: int = 0) -> None:
        for cb in self.callbacks:
            cb.on_training_start(total_timesteps)

    def on_iteration_start(self) -> None:
        for cb in self.callbacks:
            cb.on_iteration_start()

    def on_training_end(self) -> None:
        for cb in self.callbacks:
            cb.on_training_end()

    def on_step(self, infos: List[Dict[str, Any]]) -> None:
        for cb in self.callbacks:
            cb.on_step(infos)

    def on_iteration_end(self) -> bool:
        """Forward to every child; return False if any child asked to stop.

        All children run (so e.g. checkpoint still saves on the stopping
        iteration); the stop signal is aggregated, not short-circuited.
        """
        keep_training = True
        for cb in self.callbacks:
            if cb.on_iteration_end() is False:
                keep_training = False
        return keep_training

    def prepare_batch_infos(
        self,
        rewards: Any,
        lengths: Any,
        done_idx_cpu: Any,
        episode_query_indices: Any = None,  # per-episode query-pool index (parallel to rewards)
        successes: Optional[Any] = None,
        step_labels: Optional[Any] = None,  # per-episode label (from the env step output)
        terminal_categories: Optional[Any] = None,  # per-episode terminal taxonomy code
        predicate_indices: Optional[Any] = None,  # per-episode query predicate
    ) -> None:
        """Build GENERIC per-episode infos from raw rollout stats and fan out via on_step.

        Emits only env-derived, task-agnostic fields: episode reward/length, success,
        the episode's query-pool index (``episode_idx``, recorded per DONE EVENT — a
        rollout flushes many episodes per env), the per-episode label, and the query
        predicate. Task-specific interpretation that needs a side table — e.g. the KGE
        per-depth breakdown, which maps ``episode_idx`` through the query-depth table —
        lives in the task's metrics collector (``kge.callbacks.DetailedMetricsCollector``),
        not here.
        """
        num_dones = len(rewards)
        if num_dones == 0:
            return

        batch_succ = list(successes) if successes is not None else [False] * num_dones
        batch_q_idxs = episode_query_indices

        iterators = [
            rewards, lengths, batch_succ,
            batch_q_idxs if batch_q_idxs is not None else [None] * num_dones,
            step_labels if step_labels is not None else [None] * num_dones,
            terminal_categories if terminal_categories is not None else [None] * num_dones,
            predicate_indices if predicate_indices is not None else [None] * num_dones,
        ]

        batch_infos = [
            {
                "episode": {"r": float(r), "l": int(l)},
                "is_success": bool(s),
                **({"episode_idx": int(q)} if q is not None else {}),
                **({"label": int(lbl)} if lbl is not None else {}),
                **({"terminal_category": int(tc)} if tc is not None else {}),
                **({"predicate_idx": int(p)} if p is not None else {}),
            }
            for r, l, s, q, lbl, tc, p in zip(*iterators)
        ]

        self.on_step(batch_infos)
