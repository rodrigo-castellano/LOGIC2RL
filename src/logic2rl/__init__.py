"""logic2rl — the logic->RL converter framework.

Turns a logic problem (KB / rules / queries) into a gym-compatible RL
environment plus a Stable-Baselines3-faithful learning stack: env, vectorized
unification (the vendored SLD grounder), index manager, data-loading contract,
generic neural blocks, callbacks, and the ``algorithm`` package (PPO, policy,
``BaseAlgorithm``).

Domain-agnostic: holds no application/KGE specifics; those assets are
dependency-injected by consumers (e.g. the ``logic2rl-kge`` app).
"""
