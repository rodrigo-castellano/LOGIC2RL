"""Generic train/eval driver (pillar: base).

``run(config, build_env=…, build_algorithm=…)`` runs the algorithm- and
dataset-agnostic train→eval loop with the app's injected builders, returning
``{results, algorithm, env, policy}``. ``run_cli`` is the ``--set`` / ``--grid`` /
seed sweep that builds one ``RunContext`` (the run bundle) per run. Everything
task-specific enters through the injected builders (e.g. ``kge/builder.py``);
this module imports only ``base``.
"""
from __future__ import annotations

import argparse
import ast
import logging
import traceback
from dataclasses import fields
from itertools import product
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Type

import torch

from logic2rl.logging import RunContext
from logic2rl.utils import seed_all

logger = logging.getLogger(__name__)


def run(config, *, build_env: Callable, build_algorithm: Callable) -> Dict[str, Any]:
    """Build env + algorithm with the injected app builders, train (if any steps),
    then evaluate; returns ``{results, algorithm, env, policy}``.

    Eval belongs to the algorithm: ``algorithm.evaluate()`` runs the task's
    full evaluation (KGE → corruption MRR; generic → reward eval). ``learn()`` owns
    the eval-only path (fires the callbacks' ``on_training_start`` then skips
    training when ``total_timesteps <= 0``).
    """
    import time as _time
    _t = _time.perf_counter()
    env = build_env(config)                       # grounder engine + checkpoint load + data
    _t_env = _time.perf_counter() - _t; _t = _time.perf_counter()
    algorithm = build_algorithm(env, config)      # policy/providers + callbacks
    _t_algo = _time.perf_counter() - _t

    logger.info("=" * 70)
    logger.info("Run: %s | envs=%d steps=%d timesteps=%d seed=%d",
                getattr(config, "dataset", "?"), config.n_envs, config.n_steps,
                config.total_timesteps, config.seed)
    logger.info("[Load] env+engine+checkpoint=%.2fs | algorithm(policy+scorer)=%.2fs", _t_env, _t_algo)
    logger.info("=" * 70)

    # Parity mode: re-seed to match the SB3 reference before model.learn.
    if config.parity:
        seed_all(config.seed, deterministic=True)

    algorithm.learn(total_timesteps=config.total_timesteps)

    eval_results = algorithm.evaluate()
    for k in ('policy_loss', 'value_loss', 'entropy'):
        eval_results["stats"].setdefault(k, algorithm.last_metrics.get(k, 0.0))

    return {
        "results": eval_results,
        "algorithm": algorithm,
        "env": env,
        "policy": algorithm.policy,
    }


# ─────────────────────────────────────────────────────────────────────
# CLI driver — parse --set / --grid, expand the grid + seed loop, and run
# one experiment per combo through a RunContext (the run bundle).
# ─────────────────────────────────────────────────────────────────────

_BOOLEAN_TRUE = {'true', 't', 'yes', 'y', 'on', '1'}
_BOOLEAN_FALSE = {'false', 'f', 'no', 'n', 'off', '0'}


def parse_scalar(text: str) -> Any:
    """Best-effort conversion of a CLI string to a typed Python value.

    ``ast.literal_eval`` handles ints / floats / lists / dicts; a fallback covers
    bare booleans / none / strings (``lr=0.001`` → float, ``save_model=false`` →
    bool, ``seed='[0,1]'`` → list, ``dataset=family`` → str).
    """
    text = text.strip()
    if not text:
        return ''
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        low = text.lower()
        if low in _BOOLEAN_TRUE:
            return True
        if low in _BOOLEAN_FALSE:
            return False
        if low in {'none', 'null'}:
            return None
    return text


def _run_one(
    overrides: Mapping[str, Any],
    *,
    config_cls: Type,
    run_experiment: Callable[[RunContext, Any], Optional[Mapping[str, Any]]],
) -> None:
    """Build the config + RunContext for one run and execute it (the bundle lifecycle).

    Unknown override keys are dropped (only ``config_cls`` init fields are kept).
    Run-bundle metadata comes from the config's duck-typed ``family()`` /
    ``signature()`` / ``logging_config()`` methods.
    """
    valid = {f.name for f in fields(config_cls) if f.init}
    # Optional per-dataset preset applied UNDER the CLI overrides (so --set always wins).
    # config_cls.preset_overrides(overrides) is duck-typed; absent → no preset.
    preset_fn = getattr(config_cls, "preset_overrides", None)
    preset = dict(preset_fn(overrides) or {}) if callable(preset_fn) else {}
    merged = {**preset, **overrides}
    cfg = config_cls(**{k: v for k, v in merged.items() if k in valid})
    ctx = RunContext(
        logging=cfg.logging_config(), family=cfg.family(), signature=cfg.signature(),
        seed=int(getattr(cfg, "seed", 0)), resolved_config=cfg,
    )
    # Reset dynamo between runs so the previous run's compiled graphs don't leak in
    # (the direct ``run()`` / ``pipeline()`` test harness keeps its warm cache).
    torch._dynamo.reset()
    try:
        ctx.log_event("run_started")
        with ctx.stdout_capture():
            result = run_experiment(ctx, cfg)
        ctx.log_event("run_completed")
        ctx.finish(status="completed", final_metrics=dict(result) if result else {})
    except Exception as exc:
        ctx.log_event("run_failed", error=str(exc))
        ctx.finish(
            status="failed",
            final_metrics={"status": "failed", "error": str(exc), "traceback": traceback.format_exc()},
            error=str(exc),
        )
        raise


def run_cli(
    *,
    config_cls: Type,
    run_experiment: Callable[[RunContext, Any], Optional[Mapping[str, Any]]],
    description: str = "",
    extras_handler: Optional[Callable[[argparse.Namespace, dict], None]] = None,
    grid_exclude: Iterable[str] = ("seed",),
) -> None:
    """Parse ``--set`` / ``--grid``, expand the grid × seed loop, and run each combo.

    ``--set key=value`` overrides (typed via :func:`parse_scalar`) become config
    kwargs; list-valued overrides (and ``--grid key=v1,v2``) become grid dimensions
    except those in ``grid_exclude`` (``seed`` iterates as its own loop).
    ``extras_handler(args, overrides)`` fires once for CLI shortcuts (``--eval`` /
    ``--profile``). The config must expose ``family()`` / ``signature()`` /
    ``logging_config()`` for run-bundle metadata.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value, e.g. --set lr=0.001 --set seed='[0,1]'.")
    parser.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2",
                        help="Grid sweep, e.g. --grid dataset=family,fb15k237.")
    parser.add_argument("--eval", action="store_true", help="Shortcut interpreted by extras_handler.")
    parser.add_argument("--profile", action="store_true", help="Shortcut interpreted by extras_handler.")
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    for entry in args.set:
        if "=" not in entry:
            raise ValueError(f"--set expects KEY=VALUE, got {entry!r}")
        key, _, raw = entry.partition("=")
        overrides[key.strip()] = parse_scalar(raw)
    if extras_handler is not None:
        extras_handler(args, overrides)

    exclude = set(grid_exclude)
    grid = {k: v for k, v in overrides.items() if isinstance(v, list) and k not in exclude}
    for entry in args.grid:
        if "=" not in entry:
            raise ValueError(f"--grid expects KEY=V1,V2, got {entry!r}")
        key, _, raw = entry.partition("=")
        values = [parse_scalar(c) for c in raw.split(",") if c.strip()]
        if not values:
            raise ValueError(f"No values supplied for grid entry {entry!r}.")
        grid[key.strip()] = values

    seed_override = overrides.get("seed")
    seeds = seed_override if isinstance(seed_override, list) else [seed_override]
    has_seed_run_i = "seed_run_i" in {f.name for f in fields(config_cls)}

    base = {k: v for k, v in overrides.items() if k not in grid and k != "seed"}
    grid_keys = sorted(grid)
    combos = (
        [dict(zip(grid_keys, vals)) for vals in product(*(grid[k] for k in grid_keys))]
        if grid_keys else [{}]
    )

    for combo in combos:
        for seed in seeds:
            run_overrides = {**base, **combo}
            if seed is not None:
                run_overrides["seed"] = seed
                if has_seed_run_i:
                    run_overrides["seed_run_i"] = seed
            _run_one(run_overrides, config_cls=config_cls, run_experiment=run_experiment)
