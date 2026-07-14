"""Generic builders for the logic→RL converter (pillar: base).

Construction lives HERE, not on the config: the config is data only
(``base/config.py:BaseConfig``), and everything task-specific enters as an
explicit parameter — the dataset as a ``DataLoader``, the env/engine/algorithm
as injected classes, the task callbacks as prebuilt instances. Imports only
``base``; an app builder (e.g. ``kge/builder.py``) composes these with its own
pieces. A minimal task needs nothing beyond a config and a loader:

    core = build_env(config, MyDataLoader())
    algo = build_algorithm(core, config, algo_cls=MyAlgorithm)
    algo.callback = build_callbacks(config, algo, core)
"""
import logging
from typing import Any, Callable, Dict, Iterable, Optional

import torch

from logic2rl.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    MetricsCallback,
    SimpleEvalCallback,
    TorchRLCallbackManager,
)
from logic2rl.data_loader import DataLoader
from logic2rl.env import FuncEnv, GymVecEnvWrapper
from logic2rl.index_manager import IndexManager
from logic2rl.utils import seed_all

logger = logging.getLogger(__name__)


# =============================================================================
# build_env — the converter: DataLoader → IndexManager → grounder → env
# =============================================================================

def make_base_components(config) -> tuple:
    """The generic env feature components, built from config data: visit-memory
    (when ``memory_pruning``) + the unary auto-advance, layered on the env's
    invariant proof core (all-TRUE ⇒ success / any-FALSE ⇒ fail). An app builder
    extends this tuple with its own components (order note: UnaryAdvanceComponent
    must precede any stop-action component a task appends)."""
    from logic2rl.env.proof_components import UnaryAdvanceComponent
    comps = []
    if config.memory_pruning:
        from logic2rl.env.memory import VisitMemoryComponent
        comps.append(VisitMemoryComponent())
    comps.append(UnaryAdvanceComponent(
        skip_unary_actions=config.skip_unary_actions,
        max_unary_iterations=config.max_unary_iterations,
        unification_safe=config.unification_safe_skip_unary,
    ))
    return tuple(comps)


def build_env(
    config,
    data_loader: DataLoader,
    *,
    core_cls: type = FuncEnv,
    index_manager_cls: type = IndexManager,
    index_manager_kwargs: Optional[Dict[str, Any]] = None,
    engine_cls: Optional[type] = None,
    special_predicates: Iterable[str] = (),
    components: Optional[tuple] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    post_build: Optional[Callable[[Any, Any], None]] = None,
) -> "FuncEnv":
    """Turn a logic problem into a vectorized RL env — the stateless ``FuncEnv`` core.

    Generic: reads the symbolic vocabulary + rules off ``data_loader``, indexes them,
    materializes KB tensors, builds the SLD grounder, and constructs the stateless
    ``core_cls`` core with the materialized bundle. Task wiring is injected:

      * ``core_cls`` / ``index_manager_cls`` / ``index_manager_kwargs`` — the task's
        env / id-space classes (e.g. KGE's checkpoint-aligned ``KGEIndexManager``).
      * ``engine_cls`` — the grounding engine (default: the arity-general ``SLD``);
        extra ctor kwargs come from the ``config.engine_extra_kwargs`` data field.
      * ``special_predicates`` — predicate names beyond the True/False proof markers.
      * ``components`` — env feature components (default: :func:`make_base_components`).
      * ``env_kwargs`` — extra ``core_cls`` ctor kwargs beyond the generic geometry.
      * ``post_build(core, im)`` — runs while the IndexManager is still alive, so a task
        can attach build artifacts (e.g. KGE's sampler + SoftUnifier) onto the core;
        after this, nothing retains the IM.
    """
    if not isinstance(data_loader, DataLoader):
        raise TypeError(
            f"data_loader must satisfy the base.data_loader.DataLoader protocol "
            f"(constants/predicates/rules/rules_str/max_arity/padding_idx + "
            f"materialize); got {type(data_loader).__name__}."
        )
    device = torch.device(config.device)
    seed_all(config.seed, deterministic=False)

    im = index_manager_cls(
        constants=data_loader.constants,
        predicates=data_loader.predicates,
        max_arity=data_loader.max_arity,
        padding_atoms=config.padding_atoms,
        device=device,
        rules=data_loader.rules,
        rules_str=data_loader.rules_str,
        padding_idx=getattr(data_loader, "padding_idx", 0),
        extra_special_predicates=list(special_predicates),
        **(index_manager_kwargs or {}),
    )

    mat = data_loader.materialize(im=im, device=device)

    # Guard: a too-small padding_atoms silently fails proofs (a resolved rule body won't fit the
    # proof state). rules_idx is [R, 1+body_w, atom_width]; the body needs padding_atoms slots.
    if mat.rules_idx.shape[0] > 0:
        body_w = mat.rules_idx.shape[1] - 1
        assert config.padding_atoms >= body_w, (
            f"padding_atoms={config.padding_atoms} < max rule body atoms={body_w}; "
            f"raise padding_atoms so a resolved rule body fits the proof state.")

    # The engine owns the runtime-var id-space: it self-computes start/end and the packing
    # base from constant_no + the static pool size (config.max_total_vars), and the
    # branching-derived embedder n_vars once max_children is known.
    if engine_cls is None:
        from logic2rl.unification import SLD, Enumerate
        engine_cls = Enumerate if getattr(config, "resolution", "sld") == "enumerate" else SLD
    vec_engine = engine_cls(
        facts_idx=mat.facts_idx,
        rules_idx=mat.rules_idx,
        padding_idx=im.padding_idx,
        constant_no=im.constant_no,
        n_runtime_vars=config.max_total_vars,
        predicate_no=im.predicate_no,
        max_arity=im.max_arity,
        max_steps=config.max_steps,
        enforce_runtime_var_range=config.enforce_runtime_var_range,
        device=device,
        padding_atoms=config.padding_atoms,
        max_children=config.padding_states,
        **dict(config.engine_extra_kwargs),
    )
    if config.padding_states is None:
        config.padding_states = vec_engine.max_children
        logger.info("[Engine] padding_states (max_children) auto-calculated: %d", config.padding_states)
    logger.info("[Embedder] n_vars (runtime var table) = %d (max_children=%d)",
                vec_engine.n_vars, vec_engine.max_children)

    # Build-derived embedder sizing → config (the one place IM counts + engine n_vars
    # meet, while IM is alive). The embedder reads these via config; IM is then discarded.
    config.n_constants = im.constant_no
    config.n_predicates = im.predicate_no
    config.max_arity = im.max_arity
    config.n_vars = vec_engine.n_vars

    # Build the stateless core. Vocab indices: the IndexManager is the single source; the
    # core (like the engine) is a consumer of them. special_pred_ids is the name→id map
    # ({'True','False','Endf',…}).
    core = core_cls(
        vec_engine=vec_engine,
        device=device,
        data=mat,
        padding_idx=im.padding_idx,
        constant_no=im.constant_no,
        special_pred_ids=im.special_pred_ids,
        max_arity=im.max_arity,
        batch_size=config.n_envs,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_steps,
        components=components if components is not None else make_base_components(config),
        **(env_kwargs or {}),
    )

    # Stash the materialized data so a task's post_build can attach artifacts without
    # re-loading (query pools + depths/labels come from `data` in __init__).
    core._materialized = mat
    if post_build is not None:
        post_build(core, im)
    return core


def build_gym_env(config, data_loader: DataLoader, *,
                  observation_space: Optional[Any] = None, **env_seams) -> "GymVecEnvWrapper":
    """Opt-in stateful gym facade over the stateless core (for external gym / ``VectorEnv``
    consumers; the algorithms drive the bare ``build_env`` core directly). ``None``
    observation_space lets the wrapper build the default from the core dims."""
    core = build_env(config, data_loader, **env_seams)
    return GymVecEnvWrapper(core, observation_space=observation_space)


# =============================================================================
# build_algorithm — instantiate the injected algorithm class
# =============================================================================

def build_algorithm(env, config, *, algo_cls: type, **algo_kwargs):
    """Construct ``algo_cls(env, config, **algo_kwargs)``. Task pieces (an embedder,
    an ``evaluator_cls`` override) are plain kwargs supplied by the app builder;
    callbacks are attached by the caller after construction (they read the built
    algorithm's policy)."""
    return algo_cls(env, config, **algo_kwargs)


# =============================================================================
# build_callbacks — generic callback set + injected task-specific callbacks
# =============================================================================

def build_callbacks(
    config,
    algorithm,
    env,
    *,
    metrics_callback=None,
    eval_callbacks: Iterable = (),
    checkpoint_cls: type = CheckpointCallback,
    eval_metric: Optional[str] = None,
    early_stop_metric: Optional[str] = None,
    save_path=None,
    date: Optional[str] = None,
) -> Optional[TorchRLCallbackManager]:
    """Assemble the generic callback set: metrics, reward-eval, injected task eval
    callbacks, checkpoint, early stopping. Metric names are plain data — the app
    builder resolves its aliases (e.g. KGE's 'mrr' → 'mrr_mean') before calling.

    Producer/consumer order: the checkpoint + early-stop callbacks READ metrics off
    the shared dict in ``on_iteration_end``, so they are appended after the metric
    producers (metrics callback, eval callbacks)."""
    if date is None:
        import datetime
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if eval_metric is None:
        eval_metric = getattr(config, "eval_best_metric", "ep_rew_mean")

    callbacks: list = [metrics_callback if metrics_callback is not None
                       else MetricsCallback(log_interval=1, verbose=config.verbose)]

    if config.eval_freq > 0 and env is not None:
        callbacks.append(SimpleEvalCallback(
            ppo_agent=algorithm,
            eval_queries=env.valid_queries,
            eval_freq=int(config.eval_freq),
            max_steps=config.eval_max_depth,
            verbose=config.verbose,
        ))

    callbacks.extend(eval_callbacks)

    if config.save_model and save_path is not None:
        callbacks.append(checkpoint_cls(
            save_path=save_path, policy=algorithm.policy,
            train_metric="ep_rew_mean", eval_metric=eval_metric,
            verbose=True, date=date,
            restore_best=config.restore_best,
            load_best_metric=config.load_best_metric,
            load_model=config.load_model,
            save_mode=getattr(getattr(getattr(config, "logging", None), "model", None), "mode", "best"),
            single_file=True,
        ))

    # lr / ent_coef schedules are NOT callbacks: they are training semantics, owned by
    # the algorithm itself (PPO._update_schedules), so they run even with callbacks off.

    if config.early_stopping:
        es_metric = early_stop_metric or eval_metric
        # 'reward' is the generic alias for the rollout objective.
        es_metric = {"reward": "ep_rew_mean"}.get(es_metric, es_metric)
        callbacks.append(EarlyStoppingCallback(
            patience_steps=config.early_stopping_patience_steps,
            min_delta=config.early_stopping_min_delta,
            verbose=config.verbose,
            metric=es_metric,
        ))

    return TorchRLCallbackManager(callbacks=callbacks) if callbacks else None
