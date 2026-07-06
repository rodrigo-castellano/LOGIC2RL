"""BaseAlgorithm — the generic, policy-free algorithm contract.

Owns the pieces every learning algorithm shares — ``env`` / ``config`` / ``device`` /
``evaluator`` / ``callback`` + the train→eval lifecycle — and nothing else. It has **no
policy**: not every algorithm has one (PPO has a policy network, a DP solver has value
tables, MCTS has a search tree). Subclasses build whatever components they need in
``_setup()`` and implement ``learn()``:

    class PPO(BaseAlgorithm):      # policy + rollout buffer + optimizer + compiler
    class DPAlgorithm(BaseAlgorithm):   # value tables, no policy

The builder owns the wiring: the app builder injects ``evaluator_cls`` (generic
reward eval by default; the KGE builder passes its ranking/MRR evaluator), built
as ``evaluator_cls(self)``.
"""
from typing import Any, Dict, Optional

import torch


class BaseAlgorithm:
    """Generic algorithm: env/config/evaluator + the train→eval lifecycle.

    Subclasses build their components in ``_setup()`` and implement ``learn()``.
    """

    def __init__(self, env: Any, config: Any, *, eval_only: bool = False,
                 evaluator_cls: Optional[type] = None, **kwargs: Any) -> None:
        self.config = config
        self.env = env
        device = getattr(config, "device", torch.device("cpu"))
        self.device = torch.device(device) if isinstance(device, str) else device
        self.verbose = getattr(config, "verbose", 0)
        # Eval-only ⇒ skip training-side machinery (buffer/optimizer/compile). Supplied by
        # the caller (Q_KGE passes total_timesteps<=0); PPO defaults to training-enabled.
        self.eval_only = eval_only
        self.last_metrics: Dict[str, Any] = {}  # shared snapshot (algo + callbacks write)
        self.num_timesteps = 0
        self.iteration = 0  # current learn() iteration (read by callbacks off the model)
        self.callback = None

        self._setup()  # subclass builds its components (PPO: policy + buffer + compiler)

        # The builder owns the evaluator choice; built last so components exist.
        if evaluator_cls is None:
            from logic2rl.algorithm.ppo.evaluator import Evaluator
            evaluator_cls = Evaluator
        self.evaluator = evaluator_cls(self)

    def _setup(self) -> None:
        """Build the algorithm's components. Subclass hook; default no-op."""

    def learn(self, total_timesteps: int, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def evaluate(self, queries=None, **kwargs):
        """Eval entry (the runner's eval step). Full test split when ``queries is
        None``, else the given queries. Delegates to the config-chosen evaluator."""
        return self.evaluator.evaluate(queries, config=self.config, **kwargs)
