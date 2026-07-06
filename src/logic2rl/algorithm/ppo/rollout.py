"""GPU-accelerated rollout buffer with stable memory addresses for CUDA graphs."""

from typing import Dict, Generator, Tuple

import torch


class RolloutBuffer:
    """On-policy rollout buffer: observations in flat (T*N, ...) format for direct
    batching, scalars in (T, N) for GAE. All batch tensors pre-allocated once for
    CUDA graph stability.

    Observation storage is schema-driven: ``obs_spec`` maps key → (per-env shape,
    dtype) — whatever the env's observation carries. The buffer never names an
    observation field.
    """

    def __init__(
        self,
        buffer_size: int,
        n_envs: int,
        device: torch.device,
        obs_spec: Dict[str, Tuple[tuple, torch.dtype]],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        batch_size: int = 64,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.obs_spec = dict(obs_spec)
        self.pos = 0
        self.full = False

        T, N, B = buffer_size, n_envs, batch_size
        total_size = T * N
        # Observations: flat rings + stable-address batch tensors, one pair per spec key.
        self.flat_obs = {k: torch.zeros((total_size, *shp), dtype=dt, device=device)
                         for k, (shp, dt) in self.obs_spec.items()}
        self._batch_obs = {k: torch.zeros((B, *shp), dtype=dt, device=device)
                           for k, (shp, dt) in self.obs_spec.items()}
        # Scalars in (T, N) for GAE.
        self.actions = torch.zeros((T, N), dtype=torch.long, device=device)
        self.rewards = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.values = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.episode_starts = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.returns = torch.zeros((T, N), dtype=torch.float32, device=device)
        # Flattened scalar mirrors (permuted once per get()) + stable batch scalars.
        self.flat_actions = torch.zeros(total_size, dtype=torch.long, device=device)
        self.flat_values = torch.zeros(total_size, dtype=torch.float32, device=device)
        self.flat_log_probs = torch.zeros(total_size, dtype=torch.float32, device=device)
        self.flat_advantages = torch.zeros(total_size, dtype=torch.float32, device=device)
        self.flat_returns = torch.zeros(total_size, dtype=torch.float32, device=device)
        self._scalar_flat_list = [self.flat_actions, self.flat_values, self.flat_log_probs,
                                  self.flat_advantages, self.flat_returns]
        self._batch_actions = torch.zeros(B, dtype=torch.long, device=device)
        self._batch_values = torch.zeros(B, dtype=torch.float32, device=device)
        self._batch_log_probs = torch.zeros(B, dtype=torch.float32, device=device)
        self._batch_advantages = torch.zeros(B, dtype=torch.float32, device=device)
        self._batch_returns = torch.zeros(B, dtype=torch.float32, device=device)
        self._batch_scalar_dst_list = [
            self._batch_actions, self._batch_values, self._batch_log_probs,
            self._batch_advantages, self._batch_returns,
        ]
        self._permutation = torch.zeros(total_size, dtype=torch.long, device=device)
        self._add_base_indices = torch.arange(N, device=device) * T

        # Pre-computed permutations avoid randperm overhead (~0.26s per call)
        self._num_precomputed_perms = 20
        self._precomputed_perms = torch.stack([
            torch.randperm(total_size, device=device) for _ in range(self._num_precomputed_perms)
        ])
        self._perm_index = 0

    def reset(self) -> None:
        """Reset buffer position. Skips zeroing — add() overwrites before get() reads."""
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, episode_start, value, log_prob) -> None:
        """Add a transition. ``obs`` must carry every spec key."""
        pos = self.pos
        self.actions[pos] = action
        self.rewards[pos] = reward
        self.episode_starts[pos] = episode_start
        self.values[pos] = value.flatten()
        self.log_probs[pos] = log_prob.flatten()

        indices = self._add_base_indices + pos
        for k, buf in self.flat_obs.items():
            buf[indices] = obs[k]
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: torch.Tensor) -> None:
        """Compute returns and advantages using GAE."""
        last_values = last_values.flatten().to(self.device)
        dones = dones.float().to(self.device)
        last_gae_lam = torch.zeros(self.n_envs, device=self.device)

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def get(self) -> Generator[Tuple, None, None]:
        """Yield shuffled minibatches ``(obs_dict, actions, values, log_probs,
        advantages, returns)`` of ``self.batch_size`` (the CUDA-graph size the batch
        tensors were allocated for); a smaller trailing remainder yields views."""
        if not self.full:
            raise RuntimeError("Buffer is not full. Cannot sample.")
        total_size = self.buffer_size * self.n_envs
        batch_size = self.batch_size

        self._permutation.copy_(self._precomputed_perms[self._perm_index])
        self._perm_index = (self._perm_index + 1) % self._num_precomputed_perms
        perm = self._permutation

        # Flatten scalars: (T,N) → (N,T) → (N*T) then permute+copy
        _src = (self.actions, self.values, self.log_probs, self.advantages, self.returns)
        flat_tmps = [s.transpose(0, 1).contiguous().view(-1) for s in _src]
        torch._foreach_copy_(self._scalar_flat_list, [t[perm] for t in flat_tmps])

        start_idx = 0
        while start_idx < total_size:
            end_idx = min(start_idx + batch_size, total_size)
            batch_perm = perm[start_idx:end_idx]

            if end_idx - start_idx < batch_size:
                yield (
                    {k: buf[batch_perm] for k, buf in self.flat_obs.items()},
                    self.flat_actions[start_idx:end_idx],
                    self.flat_values[start_idx:end_idx],
                    self.flat_log_probs[start_idx:end_idx],
                    self.flat_advantages[start_idx:end_idx],
                    self.flat_returns[start_idx:end_idx],
                )
            else:
                for k, buf in self.flat_obs.items():
                    torch.index_select(buf, 0, batch_perm, out=self._batch_obs[k])
                s, e = start_idx, end_idx
                torch._foreach_copy_(self._batch_scalar_dst_list, [
                    self.flat_actions[s:e], self.flat_values[s:e], self.flat_log_probs[s:e],
                    self.flat_advantages[s:e], self.flat_returns[s:e],
                ])
                yield (
                    self._batch_obs,
                    self._batch_actions, self._batch_values, self._batch_log_probs,
                    self._batch_advantages, self._batch_returns,
                )

            start_idx = end_idx
