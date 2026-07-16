"""Generic env / converter config (pillar: base).

``BaseConfig`` is DATA ONLY: the environment / grounder geometry shared by any
logic→RL setup — no dataset, KGE, or algorithm fields, and no construction
logic. All construction lives in ``base/builder.py`` (generic) and the app
builders (``kge/builder.py``); an app composes this with an algorithm config:

    class KGEConfig(BaseConfig, SearchConfig): ...   # kge/config.py

The ``__post_init__`` chain is cooperative (``super().__post_init__()``), so a
composed config runs every layer's validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BaseConfig:
    """Environment / converter / grounder parameters for any logic→RL setup."""

    # Core env geometry
    device: str = "cuda"
    seed: int = 0
    # Shuffle the training query pool once (seeded) in the data loader, so the
    # env's sequential round-robin draw sees a randomized order. Turn off for
    # exact-order parity (e.g. the SB3 parity config).
    shuffle_train_queries: bool = True
    n_envs: int = 256
    max_steps: int = 20            # proof depth (max_depth in the env)
    padding_atoms: int = 6
    padding_states: Optional[int] = None  # None ⇒ derived from the engine's max_children at build
    atom_embedding_size: int = 64
    state_embedding_size: Optional[int] = None  # derived from atom_embedding_size
    verbose: bool = True

    # Env / logic components (consumed by the builder's generic component set:
    # memory_pruning wires VisitMemoryComponent; the skip-unary trio parameterizes
    # UnaryAdvanceComponent).
    eval_max_depth: int = 20
    memory_pruning: bool = True
    skip_unary_actions: bool = True
    max_unary_iterations: int = 2
    # Refuse to unary-advance a successor that still carries a free runtime var (required by
    # Q_KGE's joint scoring; the KGE config derives it in __post_init__).
    unification_safe_skip_unary: bool = False

    # Grounder
    max_total_vars: int = 100
    enforce_runtime_var_range: bool = False  # Raise on standardized/runtime var overflow (no clamping)
    # Extra engine ctor kwargs (data, not a subclass hook) — e.g. the SB3 parity
    # suite passes the frozen reference's enumeration widths here.
    engine_extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Grounder resolution (selects the engine in build_env): 'sld' = SLD backward resolution (open
    # vars filled at the replace_candidates seam), 'enumerate' = real-fact enumerate inside derive.
    resolution: str = "sld"
    # The replace_candidates seam fill — WHICH filler the app attaches to the engine:
    #   'soft'  commit each open var to the scorer's best assignment over ALL entities
    #   'hard'  commit the best REAL-FACT assignment (no-fact states discarded)
    #   'none'  leave vars open (pure resolution; e.g. enumerate, or SLD-open scoring)
    # None ⇒ default by resolution: sld → 'soft', enumerate → 'none' (enumerate's derive IS
    # the fact resolution; 'soft' there fills only its residual). The fill itself lives in
    # the app's scorer (the engine only delegates to the attached filler).
    replace_candidates: Optional[str] = None
    # Enumerate width K (real-fact fillers per open-var state). 0 = auto (app-derived).
    enumerate_k: int = 0

    # Eval geometry
    eval_batch_size: Optional[int] = None  # Fixed batch size for evaluation (defaults to n_envs if None)

    # Reproducibility (parity with the frozen SB3 reference)
    parity: bool = False

    # Build-derived sizing — filled by build_env() once the IndexManager + engine
    # exist (the embedder reads these via config; the live IM is then discarded).
    n_constants: Optional[int] = None
    n_predicates: Optional[int] = None
    n_vars: Optional[int] = None        # branching-derived runtime-var table size (from engine)
    max_arity: Optional[int] = None

    def __post_init__(self) -> None:
        _s = super()
        if hasattr(_s, "__post_init__"):
            _s.__post_init__()
        if self.n_envs <= 0:
            raise ValueError(f"n_envs must be > 0, got {self.n_envs}")
        if self.resolution not in ("sld", "enumerate"):
            raise ValueError(
                f"resolution must be 'sld' or 'enumerate', got {self.resolution!r}.")
        if self.replace_candidates is None:  # default by resolution
            self.replace_candidates = "soft" if self.resolution == "sld" else "none"
        if self.replace_candidates not in ("none", "soft", "hard"):
            raise ValueError(f"replace_candidates must be 'none' | 'soft' | 'hard', "
                             f"got {self.replace_candidates!r}.")
        if self.replace_candidates == "hard" and self.resolution != "sld":
            raise ValueError("replace_candidates='hard' requires resolution='sld' "
                             "(Enumerate's derive IS the fact resolution).")
