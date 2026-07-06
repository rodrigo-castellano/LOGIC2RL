"""Generic, domain-agnostic reward/rollout evaluator.

``Evaluator`` runs a deterministic policy rollout and reports a results dict
``{"metrics": {...}, "stats": {...}, "config": {...}}``. It owns its own compiled
eval step (argmax + ``env.step_core``), like the KGE ranking layer owns its pooled
step — evaluation compiles where it runs. The KGE ranking/MRR layer subclasses it
(:class:`algorithms.ppo.evaluator_kge.KGEEvaluator`).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class Evaluator:
    """Deterministic reward/rollout evaluator. Holds only the algorithm handle."""

    def __init__(self, ppo) -> None:
        self.ppo = ppo
        self._eval_step = None   # lazily compiled on first evaluate_policy

    def _ensure_eval_step(self) -> None:
        """Lazily build the deterministic eval step (argmax + env.step, no auto-reset)."""
        if self._eval_step is not None:
            return
        policy, env = self.ppo.policy, self.ppo.env

        def eval_step(obs, state):
            logits = policy.get_logits(obs)
            new_state, step_out = env.step_core(state, logits.argmax(dim=-1))
            return env.observation(new_state), new_state, step_out   # obs for the next eval step

        self._eval_step = (
            torch.compile(eval_step, mode=self.ppo.compile_mode, fullgraph=True)
            if self.ppo.config.compile else eval_step
        )

    @torch.no_grad()
    def evaluate_policy(
        self, queries: Tensor, max_steps: Optional[int] = None,
        deterministic: bool = True, return_episode_rewards: bool = False,
    ):
        """Deterministic rollout, SB3-parity returns (mirrors
        ``stable_baselines3.common.evaluation.evaluate_policy``):
        ``return_episode_rewards=False`` → ``(mean_reward, std_reward)`` floats;
        ``True`` → ``(episode_rewards, episode_lengths)`` tensors ``[B]``.
        Reward is ``success → +1 / failure → −1``; length is the rollout step count."""
        del deterministic
        self._ensure_eval_step()
        max_steps = max_steps or self.ppo.max_depth
        env = self.ppo.env
        torch.compiler.cudagraph_mark_step_begin()
        state = env.reset_core(queries)
        obs = {k: v.clone() for k, v in env.observation(state).items()}
        state = state.clone()
        for _ in range(max_steps):
            torch.compiler.cudagraph_mark_step_begin()
            obs, state, _ = self._eval_step(obs, state)
            obs = {k: v.clone() for k, v in obs.items()}
            state = state.clone()
        rewards = state.success.float() * 2.0 - 1.0   # success → +1, failure → −1
        lengths = state.depths
        if return_episode_rewards:
            return rewards, lengths
        return float(rewards.mean()), float(rewards.std(unbiased=False))   # np.std: ddof=0

    def evaluate(self, queries=None, *, config=None, max_steps=None, **_) -> dict:
        """Eval entry, returns ``{"metrics", "stats", "config"}``. Full test split when
        ``queries is None`` (reward rollout reporting success rate), else the given queries."""
        config = config or self.ppo.config
        if queries is None:
            queries = getattr(self.ppo.env, "test_queries", None)
            if queries is None:
                return {"metrics": {}, "stats": {}, "config": {}}
        max_steps = max_steps or getattr(config, "eval_max_depth", None)
        rewards, lengths = self.evaluate_policy(queries, max_steps=max_steps, return_episode_rewards=True)
        return {"metrics": {}, "config": {}, "stats": {
            "success_rate": float((rewards > 0).float().mean().item()),
            "mean_reward": float(rewards.mean().item()),
            "ep_len_mean": float(lengths.float().mean().item()),
        }}


__all__ = ["Evaluator"]
