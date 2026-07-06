"""SimpleEvalCallback — positive-only validation with SB3-style output."""

import logging
import time
from typing import Any, Dict, Optional

import torch

from ._base import BaseCallback

logger = logging.getLogger(__name__)


class SimpleEvalCallback(BaseCallback):
    """
    Runs deterministic evaluation on ALL positive validation queries (no negatives).
    Prints SB3-style eval block with episode_reward, episode_length, and success_rate.
    """

    def __init__(
        self,
        ppo_agent: Any,
        eval_queries: Any,  # [N, 3] tensor of positive validation queries
        eval_freq: int = 1,
        eval_batch_size: Optional[int] = None,
        max_steps: int = 20,
        verbose: bool = True,
    ):
        self.ppo_agent = ppo_agent
        self.eval_queries = eval_queries  # [N, 3]
        self.eval_freq = eval_freq
        self.eval_batch_size = eval_batch_size or ppo_agent.eval_batch_size
        self.max_steps = max_steps
        self.verbose = verbose

    def on_iteration_end(self) -> bool:
        iteration = self.model.iteration
        global_step = self.model.num_timesteps
        if iteration % self.eval_freq != 0:
            return True

        if self.verbose:
            time_start = time.time()
        self.ppo_agent.policy.eval()
        N = self.eval_queries.shape[0]
        B = self.eval_batch_size

        all_rewards = []
        all_depths = []

        # Process in chunks of eval_batch_size
        for start in range(0, N, B):
            end = min(start + B, N)
            chunk = self.eval_queries[start:end]
            actual = chunk.shape[0]

            # Pad to eval_batch_size if needed
            if actual < B:
                padded = torch.zeros(B, 3, dtype=chunk.dtype, device=chunk.device)
                padded[:actual] = chunk
                if actual > 0:
                    padded[actual:] = chunk[-1]
                chunk = padded

            # evaluate_policy returns ±1 episode rewards (success → +1 / failure → −1) + lengths.
            rewards, depths = self.ppo_agent.evaluator.evaluate_policy(
                chunk, max_steps=self.max_steps, return_episode_rewards=True)
            all_rewards.append(rewards[:actual].cpu())
            all_depths.append(depths[:actual].cpu())

        self.ppo_agent.policy.train()

        all_rewards = torch.cat(all_rewards)
        all_depths = torch.cat(all_depths)

        success_rate = (all_rewards > 0).float().mean().item()
        ep_reward_mean = all_rewards.mean().item()
        ep_reward_std = all_rewards.std().item()
        ep_len_mean = all_depths.float().mean().item()
        ep_len_std = all_depths.float().std().item()

        if self.verbose:
            logger.info("Eval num_timesteps=%d, episode_reward=%.3f +/- %.3f", global_step, ep_reward_mean, ep_reward_std)
            logger.info("Episode length: %.2f +/- %.2f", ep_len_mean, ep_len_std)
            logger.info("Success rate: %.2f%%", 100.0 * success_rate)
            logger.info("Evaluation time: %.2f seconds", time.time() - time_start)
            logger.info("---------------evaluation finished---------------")

        self.model.last_metrics.update({
            "eval/ep_reward_mean": ep_reward_mean,
            "eval/ep_reward_std": ep_reward_std,
            "eval/ep_len_mean": ep_len_mean,
            "eval/success_rate": success_rate,
        })
        return True
