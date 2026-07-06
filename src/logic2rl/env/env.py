"""Stateful gymnasium facade over the ``FuncEnv`` core (pillar: base).

``GymVecEnvWrapper`` is a thin ``gymnasium.vector.VectorEnv``: ``reset()`` / ``step()`` hold the
state internally (``self._state``) with same-step auto-reset and expose the batched spaces. All
the work lives in the wrapped stateless ``FuncEnv`` (``self.core``); this class only adds the
state cell + spaces and translates between the two. Every other attribute/method access
(``env.step_core``, ``env.engine``, ``env.sampler``, …) is delegated to the core via
``__getattr__``, so call sites use ``env.<x>`` with no ``.core`` hop.
"""
from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import torch
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from logic2rl.utils import seed_all

from .core import EnvObs, EnvState, FuncEnv, make_observation_space

__all__ = ["GymVecEnvWrapper"]


class GymVecEnvWrapper(VectorEnv):
    """Thin stateful gymnasium adapter over a ``FuncEnv``.

    Holds the recurrent ``self._state`` and the batched spaces; ``reset()`` / ``step()`` drive the
    core's stateless ``reset_core`` / ``step_autoreset`` / ``observation`` and surface the gym
    5-tuple. Gym-``VectorEnv``-API compatible (not byte-exact ``SyncVectorEnv``).
    """

    # Done lanes are reset in-step and their terminal data surfaced under gymnasium 1.2.3's
    # SAME_STEP autoreset (terminal obs → final_obs; see step()).
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(self, core: FuncEnv, *, observation_space: Optional[gym.spaces.Space] = None) -> None:
        self.core = core
        self.closed = False
        self._state: Optional[EnvState] = None
        self.num_envs = core.batch_size
        base_space = observation_space if observation_space is not None else make_observation_space(
            core.padding_atoms, core.padding_states, int(getattr(core.engine, "total_vocab_size", 0)),
            atom_width=core.atom_width,
        )
        # Component obs keys (e.g. RuleIdComponent's derived_rule_idx) are exposed only when their
        # component is on; declare them in the space the same way, so it matches what observation()
        # emits — the space-side twin of the core's per-component obs_extra merge.
        space_extra = {k: s for c in core.components for k, s in c.obs_space_extra(core).items()}
        self.single_observation_space = (
            gym.spaces.Dict({**base_space.spaces, **space_extra}) if space_extra else base_space
        )
        self.single_action_space = gym.spaces.Discrete(core.padding_states)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[EnvObs, dict]:
        """Gym reset: initialize all envs, store the state in ``self._state``, return (obs, infos).

        Draws sequentially from the active pool (``set_queries``; defaults to the training pool
        if none was set), or from ``options['queries']`` for an explicit reset (other ``options``
        keys pass through as the components' draw context). Explicit-state callers read
        ``self._state`` after this.
        """
        if seed is not None:
            seed_all(seed)

        if options and options.get("queries") is not None:
            q = options["queries"]
            extra = {k: v for k, v in options.items() if k != "queries" and v is not None}
            self._state = self.core.reset_core(q.to(self.core.device), **extra)
            return self.core.observation(self._state), {}

        # No explicit queries → draw all envs from the active pool (the core's stateless bootstrap).
        self._state = self.core.reset_pool(self._state)
        return self.core.observation(self._state), {}

    def step(self, actions: torch.Tensor) -> Tuple[EnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Gym step over ``self._state`` (SAME_STEP auto-reset of done envs).

        Returns ``(obs, rewards, terminations, truncations, info)``: terminations / truncations are
        disjoint (natural termination vs depth-limit truncation); ``info`` carries the
        per-transition info fields + ``final_obs`` (the terminal sub_index) with its
        ``final_obs_mask`` done-lane mask. ``actions`` is coerced to a torch ``[B]`` long tensor,
        so a numpy ``action_space.sample()`` works.
        """
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.long, device=self.core.device)
        self._state, out = self.core.step_autoreset(self._state, actions)
        obs = self.core.observation(self._state)
        truncations = out.step_truncated.bool()
        terminations = out.step_dones.bool() & ~truncations          # gym: term / trunc disjoint
        info = {f: getattr(out, f) for f in out._fields               # info fields → plain keys
                if f not in ("step_rewards", "step_dones", "step_truncated", "final_observation")}
        info["final_obs"] = out.final_observation                    # terminal observation (sub_index)
        info["final_obs_mask"] = out.step_dones.bool()               # which lanes are terminal this step
        return obs, out.step_rewards, terminations, truncations, info

    def __getattr__(self, name: str):
        """Read-through to the stateless core: every attribute/method not on the facade resolves
        on ``self.core`` — so ``env.step_core(...)`` / ``env.engine`` / ``env.sampler`` need no
        ``.core`` hop (the compiled rollout resolves these at trace time). Writes are NOT forwarded:
        build-time task metadata is attached to the core directly."""
        if name == "core":                       # core not set yet (pre-__init__ / unpickle)
            raise AttributeError(name)
        return getattr(self.core, name)
