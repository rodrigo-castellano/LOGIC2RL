"""Standalone smoke test for the logic2rl library.

Proves the package installs and its internal import graph resolves after the
`base` -> `logic2rl` rename and the `algorithm/` repackaging. A full training
loop is exercised by the `logic2rl-kge` consumer suite (it injects the KGE data
loader / KB that a real env build needs).
"""


def test_public_imports():
    import logic2rl  # noqa: F401
    from logic2rl.algorithm import BaseAlgorithm  # noqa: F401
    from logic2rl.algorithm.policy import ActorCriticPolicy  # noqa: F401
    from logic2rl.algorithm.ppo import PPO, RolloutBuffer  # noqa: F401
    from logic2rl.builder import build_algorithm, build_env  # noqa: F401
    from logic2rl.config import BaseConfig  # noqa: F401
    from logic2rl.env import FuncEnv, GymVecEnvWrapper, make_observation_space  # noqa: F401
    from logic2rl.nn import EmbedderLearnable  # noqa: F401
    from logic2rl.runner import run  # noqa: F401
    from logic2rl.unification import SLD  # noqa: F401


def test_base_config_constructs():
    from logic2rl.config import BaseConfig

    cfg = BaseConfig()
    assert cfg.seed == 0
    assert cfg.max_steps > 0
    assert cfg.n_envs > 0
