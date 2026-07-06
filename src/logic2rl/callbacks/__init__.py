"""Generic training callbacks (pillar: base).

Domain-agnostic callback hooks invoked during PPO training: rollout/metric
tracking, checkpointing, early stopping, and console
display. The KGE-specific ranking/MRR callback lives in
``kge/callbacks/eval_mrr.py`` (it depends on ``algorithm.evaluator``).
"""

from ._base import BaseCallback
from ._checkpoint import CheckpointCallback
from ._display import Display
from ._early_stopping import EarlyStoppingCallback, OptunaPruningCallback
from ._eval import SimpleEvalCallback
from ._manager import TorchRLCallbackManager
from ._metrics import (
    MetricsCallback,
    MetricsCollector,
    RewardTracker,
)

__all__ = [
    "BaseCallback",
    "Display",
    "RewardTracker",
    "MetricsCollector",
    "TorchRLCallbackManager",
    "MetricsCallback",
    "CheckpointCallback",
    "SimpleEvalCallback",
    "EarlyStoppingCallback",
    "OptunaPruningCallback",
]
