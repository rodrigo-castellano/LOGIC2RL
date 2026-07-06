"""Metric trackers and collectors for training callbacks (pillar: base).

The collectors here track overall episode metrics (reward, length, success),
mirroring SB3's rollout block. Per-label / per-depth / per-predicate breakdowns
are a KGE debugging concern and live in ``kge/callbacks/metrics.py`` as
subclasses that extend ``MetricsCollector`` / ``MetricsCallback``.
"""

import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ._base import BaseCallback
from ._display import Display

logger = logging.getLogger(__name__)


class RewardTracker:
    """Tracks training rewards."""

    def __init__(self, patience: int = 10):
        self.history: list = []
        self.best_reward: float = float('-inf')
        self.best_iteration: int = 0
        self.patience = patience
        self.no_improvement_count: int = 0

    def update(self, reward: float, iteration: int) -> Dict[str, Any]:
        self.history.append({'iteration': iteration, 'reward': reward})
        is_best = reward > self.best_reward
        if is_best:
            self.best_reward = reward
            self.best_iteration = iteration
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        return {
            'current_reward': reward,
            'best_reward': self.best_reward,
            'best_iteration': self.best_iteration,
            'is_best': is_best,
        }

    def get_summary(self) -> str:
        if not self.history:
            return "No reward data recorded"
        current = self.history[-1]['reward']
        return f"Reward (train): current={current:.3f}, best={self.best_reward:.3f} (iter {self.best_iteration})"


class MetricsCollector:
    """Collects overall episode metrics (reward, length, success rate).

    SB3-like: only the aggregate rollout numbers, with no per-label / per-depth /
    per-predicate breakdown. Subclasses extend ``_record`` / ``compute_metrics``
    to add richer buckets (see ``kge.callbacks.DetailedMetricsCollector``).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._episodes: List[Dict[str, Optional[float]]] = []
        self._last_episode_id: Dict[int, int] = {}

    def _iter_new_episodes(
        self, infos: List[Dict[str, Any]]
    ) -> Iterator[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """Yield ``(env_idx, info, episode_data)`` for fresh (non-duplicate) episodes."""
        for env_idx, info in enumerate(infos):
            if not info or "episode" not in info:
                continue
            episode_data = info.get("episode")
            if not isinstance(episode_data, dict):
                continue
            # Dedup: prefer the explicit episode_idx, else fall back to object id.
            episode_idx = info.get("episode_idx")
            marker = episode_idx if episode_idx is not None else id(episode_data)
            if self._last_episode_id.get(env_idx) == marker:
                continue
            self._last_episode_id[env_idx] = marker
            yield env_idx, info, episode_data

    @staticmethod
    def _extract_episode_stats(info: Dict[str, Any], episode_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
        reward = episode_data.get("r")
        length = episode_data.get("l")
        return {
            "reward": float(reward) if reward is not None else None,
            "length": float(length) if length is not None else None,
            "success": 1.0 if bool(info.get("is_success", False)) else 0.0,
        }

    def accumulate(self, infos: List[Dict[str, Any]]) -> None:
        for _env_idx, info, episode_data in self._iter_new_episodes(infos):
            self._record(self._extract_episode_stats(info, episode_data), info)

    def _record(self, stats: Dict[str, Optional[float]], info: Dict[str, Any]) -> None:
        """Store one episode's stats. Override to add breakdown buckets (call super first)."""
        self._episodes.append(stats)

    def compute_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if not self._episodes:
            return metrics

        rewards = [ep["reward"] for ep in self._episodes if ep.get("reward") is not None]
        lengths = [ep["length"] for ep in self._episodes if ep.get("length") is not None]
        successes = [ep.get("success", 0.0) for ep in self._episodes]

        if rewards:
            metrics["ep_rew_mean"] = float(np.mean(rewards))
            metrics["reward"] = Display._format_stat_string(
                np.mean(rewards), np.std(rewards), len(rewards)
            )
        if lengths:
            metrics["ep_len_mean"] = float(np.mean(lengths))
            metrics["len"] = Display._format_stat_string(
                np.mean(lengths), np.std(lengths), len(lengths)
            )
        if successes:
            metrics["success_rate"] = f"{np.mean(successes):.3f}"

        return metrics


class MetricsCallback(BaseCallback):
    """Collects training metrics and logs a COMPACT per-iteration summary line.

    Generic: timing, the PPO train-metric passthrough, the reward tracker, and the
    one-line console summary (``_log_compact``, overridden by richer subclasses).
    Episode metrics come from a ``MetricsCollector`` built by ``_build_collector`` —
    subclasses override that hook to plug in a richer collector (see
    ``kge.callbacks.DetailedMetricsCallback``). A metric *producer*: it writes its
    rollout aggregates into ``model.last_metrics`` (which the PPO loop has already
    populated with the ``train/*`` losses) — the FULL metric set is always collected
    and stored; ``log_diagnostics`` only gates printing it as a table.
    """

    def __init__(self, log_interval: int = 1, verbose: bool = True,
                 log_diagnostics: bool = False):
        self.log_interval = log_interval
        self.verbose = verbose
        self.log_diagnostics = log_diagnostics
        self.collector = self._build_collector()
        self.reward_tracker = RewardTracker(patience=20)
        self.train_start_time = None
        self.last_time = None
        self.last_step = 0

    def _build_collector(self) -> MetricsCollector:
        """Collector factory; override to plug in a richer collector."""
        return MetricsCollector()

    def on_training_start(self, total_timesteps: int = 0) -> None:
        self.train_start_time = time.time()
        self.last_time = time.time()

    def on_step(self, infos: List[Dict[str, Any]]) -> None:
        self.collector.accumulate(infos)

    def on_iteration_end(self) -> bool:
        iteration = self.model.iteration
        global_step = self.model.num_timesteps
        if iteration % self.log_interval != 0:
            self.collector.reset()
            return True

        # Compute metrics
        rollout_metrics = self.collector.compute_metrics()

        # Compute timing
        current_time = time.time()
        elapsed = current_time - self.last_time
        steps_done = global_step - self.last_step
        fps = int(steps_done / elapsed) if elapsed > 0 else 0

        self.last_time = current_time
        self.last_step = global_step

        timing = {
            "time/iteration_fps": fps,
            "time/elapsed": int(current_time - self.train_start_time),
            "total_timesteps": global_step,
        }

        # The PPO loop already wrote the train/* losses into model.last_metrics; copy
        # the display-relevant ones (+ formatted diagnostics) into the console block.
        train_metrics = self.model.last_metrics
        if train_metrics:
            for key in ("approx_kl", "clip_fraction", "clip_range", "entropy_loss",
                        "explained_var", "learning_rate", "loss",
                        "policy_gradient_loss", "value_loss"):
                if key in train_metrics:
                    timing[key] = train_metrics[key]
            # Also add explained_variance alias for display compatibility
            if "explained_var" in train_metrics:
                timing["explained_variance"] = train_metrics["explained_var"]
            # Diagnostic: value and advantage as "mean +/- std"
            if "value_mean" in train_metrics and "value_std" in train_metrics:
                timing["value (mean +/- std)"] = f"{train_metrics['value_mean']:.3f} +/- {train_metrics['value_std']:.3f}"
            if "advantage_mean" in train_metrics and "advantage_std" in train_metrics:
                timing["advantage (mean +/- std)"] = f"{train_metrics['advantage_mean']:.3f} +/- {train_metrics['advantage_std']:.3f}"
            # Pass through logit diagnostics
            for key in train_metrics:
                if key.startswith("logit_") or key in ("obs_norm", "act_norm"):
                    timing[key] = train_metrics[key]

        if "ep_rew_mean" in rollout_metrics:
            self.reward_tracker.update(rollout_metrics["ep_rew_mean"], iteration)

        if self.verbose:
            self._log_compact(rollout_metrics, timing, iteration, global_step)
            if self.log_diagnostics:
                Display.print_formatted_metrics(
                    metrics=rollout_metrics, prefix="rollout",
                    extra_metrics=timing, global_step=global_step)

        # Write rollout aggregates + timing into the shared metrics dict (consumers
        # — checkpoint, early-stop — read it later this iteration).
        self.model.last_metrics.update({**rollout_metrics, **timing})
        return True

    def _log_compact(self, m: Dict[str, Any], timing: Dict[str, Any],
                     iteration: int, global_step: int) -> None:
        """The default one-line rollout summary. Subclasses override to add their
        breakdowns (the KGE callback adds pos/neg splits + proven-by-depth)."""
        logger.info(
            "[roll it %d | %s steps | %s fps] rwd %s (best %s) | len %s | proven %s",
            iteration, f"{global_step:,}", timing.get("time/iteration_fps", "?"),
            Display.fmt(m.get("ep_rew_mean")), Display.fmt(self.reward_tracker.best_reward),
            Display.fmt(m.get("ep_len_mean"), 1), m.get("success_rate", "—"))
