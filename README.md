# logic2rl

A **logic → RL converter framework**: it turns a logic problem (KB / rules / queries) into a
gym-compatible, vectorized RL environment, and ships a Stable-Baselines3-faithful learning stack on top.

It is domain-agnostic — it holds no application/KGE specifics. Applications (e.g. `logic2rl-kge`) depend on it
and inject their own data loader, env components, embedders, and evaluators.

## Install

```bash
pip install "git+https://github.com/rodrigo-castellano/LOGIC2RL.git@<sha>"
```

Editable dev install:

```bash
git clone https://github.com/rodrigo-castellano/LOGIC2RL.git
pip install -e LOGIC2RL[dev]
```

## What's inside (`logic2rl/`)

- `env/` — the stateless `FuncEnv` core + the `GymVecEnvWrapper` facade + obs/state types + components.
- `unification/` — the vectorized single-step SLD grounding engine (`SLD`).
- `nn/` — generic neural building blocks (CUDA-graph-safe attention/GRU, set encoders, `EmbedderLearnable`).
- `algorithm/` — the SB3-faithful learning stack: `algorithm.base.BaseAlgorithm`, `algorithm.ppo` (PPO,
  rollout, loss, metrics, generic evaluator), `algorithm.policy` (`ActorCriticPolicy` + networks/extractors).
- `config.py` (`BaseConfig`), `builder.py` (`build_env`/`build_algorithm`/`build_callbacks`, all task pieces
  injected as parameters), `runner.py` (`run`/`run_cli`), `logging.py`, `data_loader.py`, `index_manager.py`, …

## Usage sketch

```python
from logic2rl.config import BaseConfig
from logic2rl.builder import build_env, build_algorithm
from logic2rl.runner import run
# An application injects core_cls / engine_cls / components / data_loader / algo_cls; see logic2rl-kge.
```

## Tests

```bash
python -m pytest tests -q
```
