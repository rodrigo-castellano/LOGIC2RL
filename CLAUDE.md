# CLAUDE.md

## Scope

- Applies to the entire `logic2rl` repository — the reusable, domain-agnostic logic→RL library.
- Canonical section order for every `CLAUDE.md`: Scope, Project Overview, Architecture, Running Experiments,
  Logging Experiments, Testing, Documentation, Adding or Changing Code, Coding Standards, Technical Rules,
  Verification Checklist. Keep headings even when a section states "not applicable here".

## Project Overview

`logic2rl` turns a logic problem (KB / rules / queries) into a gym-compatible vectorized RL environment plus a
Stable-Baselines3-faithful learning stack. It is the reusable core extracted from DeepProofLog; applications
(e.g. `logic2rl-kge`) pip-install it and inject their own data loader, env components, embedders, evaluators, and
algorithm. Published as the importable package `logic2rl` (src-layout, real packages — no sys.path bootstrap).

## Architecture

- `logic2rl/env/` — stateless `FuncEnv` core + `GymVecEnvWrapper` facade + obs/state types + components.
- `logic2rl/unification/` — the single-step SLD grounding engine (`sld/`, exposed as `SLD`).
- `logic2rl/nn/` — generic neural blocks (CUDA-graph-safe attention/GRU, set encoders, `EmbedderLearnable`).
- `logic2rl/algorithm/` — SB3-faithful learning stack: `base.py` (`BaseAlgorithm`), `ppo/` (PPO + rollout + loss
  + metrics + generic `evaluator`), `policy/` (`ActorCriticPolicy` + networks/extractors/layers).
- `config.py` (`BaseConfig`), `builder.py`, `runner.py`, `logging.py`, `data_loader.py`, `dataset.py`,
  `index_manager.py`, `det_ops.py`, `utils.py`, `callbacks/`.
- **Configs are data; builders construct.** `build_env`/`build_algorithm`/`build_callbacks` take every task piece
  (core_cls / engine_cls / components / data_loader / algo_cls / evaluator_cls / …) as an explicit parameter.
- Domain-agnostic: never import application/KGE code here. Consumers inject their pieces.

## Running Experiments

Not applicable — this is a library, not an experiment runner. Experiments live in consumer repos (`logic2rl-kge`).

## Logging Experiments

`logging.py` provides run-bundle plumbing (`RunContext`, `LoggingConfig`) that consumers wire up; this repo does
not itself write run bundles.

## Testing

```bash
python -m pytest tests -q     # standalone import/instantiation smoke test
```

Deeper end-to-end coverage of this library runs in the `logic2rl-kge` consumer suite (real PPO train+eval).

## Documentation

- Update `README.md` when the public surface or install flow changes.
- Keep this `CLAUDE.md` aligned with the actual tree; remove stale inventories.

## Adding or Changing Code

- One clear responsibility per module; extend the existing owner rather than adding near-duplicate files.
- Do not add application/KGE specifics — keep the library domain-agnostic; expose injection seams instead.

## Coding Standards

- Type hints on signatures; document important tensor shapes (e.g. `[B, S, M, 3]`).
- Vectorized tensor code over Python loops in hot paths; keep compile-friendly code compile-friendly.
- Hyperparameters in config/arguments, not hardcoded. Concise comments only where they add information.

## Technical Rules

- Never revert/restore files without explicit permission. Fix bugs forward; no silent clamps/fallbacks.
- This library is consumed via a `git+https@<sha>` pin in `logic2rl-kge`. When its API changes, the consumer's
  pin must be bumped (its `check-editable-pins` hook enforces this).

## Verification Checklist

- Code change: `python -m pytest tests -q`.
- Public-surface change: also update `README.md` and confirm `from logic2rl.<x> import …` still resolves.
