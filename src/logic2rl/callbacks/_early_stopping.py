"""EarlyStoppingCallback and OptunaPruningCallback."""

import logging
from typing import Any, Dict, Optional

from ._base import BaseCallback

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(BaseCallback):
    """Stops training when the eval metric stalls over a patience window.

    Checks whether the best observed value of ``metric`` has improved by at
    least ``min_delta`` since the last checkpoint. Checkpoints are placed every
    ``patience_steps`` timesteps. ``metric`` is the metrics-dict key to monitor
    (generic; the app builder passes its metric, e.g. KGE's ``"mrr_mean"``). ``best_metric_key``
    is an optional pre-aggregated "best so far" key the producer may supply.
    """

    def __init__(self, patience_steps: int = 3_000_000,
                 min_delta: float = 0.01, verbose: bool = True,
                 metric: str = "ep_rew_mean", best_metric_key: Optional[str] = None):
        self.patience_steps = patience_steps
        self.min_delta = min_delta
        self.verbose = verbose
        self.metric = metric
        self.best_metric_key = best_metric_key if best_metric_key is not None else f"best_{metric}"
        self.should_stop = False
        self._checkpoint_value: float = 0.0
        self._checkpoint_step: int = 0
        self._best_value: float = 0.0
        self._initialized = False

    def on_iteration_end(self) -> bool:
        return not self.check_early_stop(self.model.last_metrics, self.model.num_timesteps)

    def check_early_stop(self, metrics: Dict[str, Any], global_step: int) -> bool:
        """Whether training should stop. Called from ``on_iteration_end``.

        Checkpoints are placed every ``patience_steps``.  At each checkpoint,
        if the best metric value has improved by at least ``min_delta`` since
        the last checkpoint, the checkpoint is updated and training continues.
        Otherwise training stops.
        """
        value = metrics.get(self.metric)
        if value is None:
            return False

        best_value = metrics.get(self.best_metric_key, value)
        self._best_value = max(self._best_value, best_value)

        if not self._initialized:
            self._checkpoint_value = self._best_value
            self._checkpoint_step = global_step
            self._initialized = True
            if self.verbose:
                logger.info("[EarlyStop] Initialized checkpoint: %s=%.4f at step %d",
                            self.metric, self._checkpoint_value, global_step)
            return False

        if global_step >= self._checkpoint_step + self.patience_steps:
            improvement = self._best_value - self._checkpoint_value
            if improvement >= self.min_delta:
                if self.verbose:
                    logger.info("[EarlyStop] Checkpoint updated: %s %.4f -> %.4f (+%.4f) at step %d",
                               self.metric, self._checkpoint_value, self._best_value, improvement, global_step)
                self._checkpoint_value = self._best_value
                self._checkpoint_step = global_step
            else:
                self.should_stop = True
                if self.verbose:
                    logger.info("[EarlyStop] STOPPING: best %s=%.4f, checkpoint %s=%.4f, "
                               "improvement=%.4f < min_delta=%.4f",
                               self.metric, self._best_value, self.metric, self._checkpoint_value,
                               improvement, self.min_delta)
                return True
        return False


class OptunaPruningCallback(BaseCallback):
    """Reports the intermediate eval metric to an Optuna trial and prunes if needed.

    ``metric`` is the metrics-dict key to report (generic).
    """

    def __init__(self, trial, verbose: bool = True, metric: str = "ep_rew_mean"):
        self.trial = trial
        self.verbose = verbose
        self.metric = metric
        self._step_count = 0

    def on_iteration_end(self) -> bool:
        return not self.check_early_stop(self.model.last_metrics, self.model.num_timesteps)

    def check_early_stop(self, metrics: Dict[str, Any], global_step: int) -> bool:
        """Report the eval metric to Optuna and check pruning."""
        import optuna

        value = metrics.get(self.metric)
        if value is None:
            return False
        self._step_count += 1
        self.trial.report(float(value), step=self._step_count)
        if self.trial.should_prune():
            if self.verbose:
                logger.info("[OptunaPrune] Trial %d pruned at step %d, %s=%.4f",
                            self.trial.number, global_step, self.metric, value)
            raise optuna.TrialPruned()
        return False
