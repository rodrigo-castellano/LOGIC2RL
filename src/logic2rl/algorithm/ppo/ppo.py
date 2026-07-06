"""PPO with CUDA graph support for neurosymbolic KG reasoning.

The whole training path lives on the class. Two methods are the compiled regions —
``rollout_step`` (policy sample + env step) and ``_loss_step`` (fused loss) — and
``_compile`` wraps them with fullgraph ``torch.compile`` (or leaves them eager for
debugging). Evaluation compiles where it runs (see the evaluators).

    _setup             hyperparams → policy → buffers (schema by example) → optimizer → _compile
    rollout_step       compiled: sample + env.step_autoreset
    _loss_step         compiled: PPOLossModule over the buffer's static batch tensors
    _warmup_gradients  pre-allocate .grad storage (stable addresses for cudagraphs)
    _warmup_rollout    one-off compile of the rollout, RNG-guarded (SB3 parity)
    collect_rollouts   the rollout loop (prepare → step → bootstrap → buffer → stats)
    train              epochs × minibatches over the buffer
    learn              driver: schedules → collect → train → callbacks
"""
import math
import time
from typing import Any, Dict

import torch
from tensordict import TensorDict

from logic2rl.algorithm import BaseAlgorithm
from logic2rl.algorithm.policy.protocol import Policy
from logic2rl.env import FuncEnv

from .loss import PPOLossModule, fused_clip_grad_norm_
from .metrics import compute_train_metrics
from .rollout import RolloutBuffer

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def _anneal(progress: float, init: float, final: float, start: float, end: float,
            transform: str, warmup: float = 0.0) -> float:
    """Scalar schedule: optional warmup ramp 0→init, then init→final over
    progress ``[start, end]`` with a cos/linear/exp transform."""
    if warmup > 0 and progress < warmup:
        return progress / warmup * init
    if progress <= start:
        return init
    if progress >= end:
        return final
    s = max(start, warmup)
    p = min(1.0, max(0.0, (progress - s) / (end - s + 1e-9)))
    if transform == 'exp' and init > 0 and final > 0:
        return init * (final / init) ** p
    if transform == 'cos':
        return final + (init - final) * 0.5 * (1.0 + math.cos(math.pi * p))
    return init + p * (final - init)   # linear (and fallback)


# Optional per-episode stats the callback manager consumes, captured into [T, B]
# rings during collect (only while a step callback is attached) and flushed with a
# single GPU→CPU sync per rollout: name → (required StepOutput field | None, extractor).
# A field the env's StepOutput doesn't declare is skipped (its kwarg stays None at the
# manager). '_query_ptrs' records the post-step pointer, from which the flush recovers
# the finished EPISODE's query-pool index: draw_queries stores ptr+1 after drawing
# query_pool[ptr], and the done step's auto-reset advances once more — so for a done
# lane the post-step per_env_ptrs is (episode query index) + 2 (mod pool). Recorded per
# done EVENT — a rollout flushes many episodes per env, so a per-env pointer would
# attribute every episode to the env's LAST query. (Must read the post-step state: the
# pre-step state is a prior cudagraph output, already overwritten at read time.)
_EPISODE_STATS = {
    'successes':           ('is_success',        lambda out, state: out.is_success),
    'step_labels':         ('step_labels',       lambda out, state: out.step_labels),
    'terminal_categories': ('terminal_category', lambda out, state: out.terminal_category),
    'predicate_indices':   ('original_queries',  lambda out, state: out.original_queries[:, 0, 0]),
    '_query_ptrs':         (None,                lambda out, state: state.per_env_ptrs),
}


class PPO(BaseAlgorithm):
    """PPO with CUDA graph support and separate train/eval compilation.

    Subclasses :class:`BaseAlgorithm` (env/config/evaluator + lifecycle — ``evaluate``
    is inherited) and adds the policy-gradient machinery. The task's embedder (the
    policy's feature extraction) is injected by the app builder (e.g. ``kge/builder``
    passes ``EmbedderLearnable``); subclasses that build their own policy (Q_KGE)
    override ``_build_policy`` and need no embedder.
    """

    def __init__(self, env, config, *, embedder=None, **kwargs) -> None:
        self._embedder = embedder
        super().__init__(env, config, **kwargs)

    # ── setup ─────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Build everything: hyperparameters, policy, rollout + stat buffers,
        optimizer, compiled steps. Called by :meth:`BaseAlgorithm.__init__`;
        ``eval_only`` stops after the policy (no training machinery)."""
        config = self.config
        # Hyperparameters — read straight off config (no kwargs duality, no re-defaults).
        self.n_envs = config.n_envs
        self.max_depth = config.max_steps
        self.padding_atoms = config.padding_atoms
        self.padding_states = config.padding_states
        self.n_steps = config.n_steps
        self.n_epochs = config.n_epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_range = config.clip_range
        self.clip_range_vf = config.clip_range_vf
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.max_grad_norm = config.max_grad_norm
        self.target_kl = config.target_kl
        self.weight_decay = config.weight_decay
        self.normalize_advantage = config.normalize_advantage
        self.normalize_returns = config.normalize_returns
        self.compile_mode = config.compile_mode
        self.eval_batch_size = config.eval_batch_size or self.n_envs  # static eval batch
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float32

        self.policy: Policy = self._build_policy(self.env, config)
        assert isinstance(self.policy, Policy), (
            f"{type(self.policy).__name__} does not satisfy the Policy contract — needs "
            "forward/get_logits/predict_values/evaluate_actions/prepare_step "
            "(see algorithms/policy/protocol.py)."
        )
        self._ent_coef_buf = torch.tensor(self.ent_coef, dtype=torch.float32, device=self.device)
        self._primed = False
        if self.eval_only:
            self.rollout_buffer = None
            self.optimizer = None
            self._cached_params = None
            return

        # Buffer schema = the env's observation, by example (one functional reference
        # reset, RNG-guarded): whatever keys the env emits — e.g. derived_rule_idx
        # under rule-idx tracking — are stored and round-tripped to the loss without
        # being named here.
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None
        with torch.no_grad():
            ref_queries = self.env.train_queries[:1].expand(self.n_envs, -1).contiguous()
            ref_obs = self.env.observation(self.env.reset_core(ref_queries))
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        obs_spec = {k: (tuple(v.shape[1:]), v.dtype) for k, v in ref_obs.items()}
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps, n_envs=self.n_envs, device=self.device,
            obs_spec=obs_spec, gamma=self.gamma, gae_lambda=self.gae_lambda,
            batch_size=self.batch_size,
        )

        # Episode-stat rings [T, B]: dones/rewards/lengths always; the optional extras
        # from _EPISODE_STATS only if the env's StepOutput declares the source field.
        T, B, device = self.n_steps, self.n_envs, self.device
        self._step_dones = torch.zeros(T, B, device=device, dtype=torch.bool)
        self._step_rewards = torch.zeros(T, B, device=device, dtype=torch.float32)
        self._step_lengths = torch.zeros(T, B, device=device, dtype=torch.long)
        self._stat_fns = {name: fn for name, (field, fn) in _EPISODE_STATS.items()
                          if field is None or field in self.env.StepOutput._fields}
        self._stat_bufs = {name: torch.zeros(T, B, device=device) for name in self._stat_fns}

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.learning_rate, eps=1e-5,
            weight_decay=self.weight_decay, fused=self.device.type == 'cuda')
        self._epoch_end_indices = []
        self._compile()
        self._cached_params = list(self.policy.parameters())

    def _build_policy(self, env: FuncEnv, config: Any) -> Policy:
        """Build the policy: injected embedder → feature extractor → ActorCriticPolicy,
        arch knobs read straight off the config data (overridable — Q_KGE builds a
        QKGEPolicy with no embedder).

        Seeding order (manual_seed -> embedder -> manual_seed -> policy nets) is fixed to
        match the SB3 reference exactly: the app builder seeds before constructing the
        embedder; here we re-seed before the policy nets. The extractor has no params,
        so building it between the seeds is RNG-neutral.
        """
        from logic2rl.algorithm.policy import ActorCriticPolicy, CustomCombinedExtractor
        if self._embedder is None:
            raise ValueError(
                "PPO needs an embedder for its default ActorCriticPolicy — the app "
                "builder constructs and injects it (e.g. kge/builder.build_algorithm).")
        device = env.device
        features_extractor = CustomCombinedExtractor(self._embedder)
        torch.manual_seed(config.seed)
        return ActorCriticPolicy(
            features_extractor=features_extractor, device=device,
            action_dim=config.padding_states,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            separate_value_network=config.separate_value_network,
            value_head_scale=config.value_head_scale,
            use_l2_norm=config.use_l2_norm,
            temperature=config.temperature,
            sqrt_scale=config.sqrt_scale,
            learnable_temperature=config.learnable_temperature,
            obs_body_layers=config.obs_body_layers,
            obs_head_layers=config.obs_head_layers,
            action_body_layers=config.action_body_layers,
            action_head_layers=config.action_head_layers,
            shared_policy_body=config.shared_policy_body,
            shared_policy_head=config.shared_policy_head,
            parity=getattr(config, "parity", False),
        ).to(device)

    # ── the two compiled regions + their wrapping ─────────────────────

    def rollout_step(self, obs, state):
        """One policy sample + env step — the rollout's compiled region.

        ``get_logits`` already masks invalid slots (-inf), so softmax+multinomial
        equals the eager dist.sample. The value head and the eager ``prepare_step``
        prelude stay outside the graph (``collect_rollouts``).
        """
        logits = self.policy.get_logits(obs)
        probs = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
        log_probs = torch.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
        new_state, step_out = self.env.step_autoreset(state, actions)   # step + same-step reset of done lanes
        new_obs = self.env.observation(new_state)                       # state → the policy's obs for next step
        return new_obs, new_state, step_out, actions, log_probs

    def _loss_step(self, batch=None):
        """The PPO loss region. Compiled: called with no ``batch`` → reads the
        buffer's static batch tensors (stable cudagraph addresses). Eager
        (compile=False): reads the yielded minibatch (handles the partial tail)."""
        if batch is None:
            rb = self.rollout_buffer
            return self.loss_module(rb._batch_obs, rb._batch_actions, rb._batch_advantages,
                                    rb._batch_returns, rb._batch_log_probs, rb._batch_values,
                                    self.clip_range, self.clip_range_vf or 0.0,
                                    self._ent_coef_buf, self.vf_coef)
        obs, actions, old_values, old_log_probs, advantages, returns = batch
        return self.loss_module(obs, actions, advantages, returns, old_log_probs, old_values,
                                self.clip_range, self.clip_range_vf or 0.0,
                                self.ent_coef, self.vf_coef)

    def _compile(self) -> None:
        """Wrap the two regions: fullgraph torch.compile (production) or eager (debug).
        ``loss_step(batch)`` is what ``train`` calls either way."""
        torch._inductor.config.fx_graph_cache = True
        self.loss_module = PPOLossModule(self.policy)
        if self.config.compile:
            self._warmup_gradients()
            self._rollout_fn = torch.compile(self.rollout_step, mode=self.compile_mode, fullgraph=True)
            compiled_loss = torch.compile(self._loss_step, mode=self.compile_mode, fullgraph=True)
            self.loss_step = lambda batch: compiled_loss()
        else:
            self._rollout_fn = self.rollout_step
            self.loss_step = self._loss_step

    def _warmup_gradients(self) -> None:
        """Backward-pass warmup: one dummy loss+backward pre-allocates every ``.grad``
        so its storage address is stable for cudagraph capture. The dummy obs comes
        from the buffer's schema — all-ones mask (an all-zeros mask would -inf every
        slot)."""
        B, dev = self.batch_size, self.device
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            d = {k: (torch.ones if k == 'action_mask' else torch.zeros)((B, *shp), dtype=dt, device=dev)
                 for k, (shp, dt) in self.rollout_buffer.obs_spec.items()}
            v, lp, ent = self.policy.evaluate_actions(
                TensorDict(d, batch_size=[B]),
                torch.zeros(B, dtype=torch.long, device=dev),
            )
            loss = v.mean() + lp.mean() + ent.mean()
        self.optimizer.zero_grad(set_to_none=False)
        loss.backward()
        self.optimizer.zero_grad(set_to_none=False)

    def _warmup_rollout(self, obs, state) -> None:
        """Forward-pass warmup: trigger the one-off torch.compile of the rollout at
        learn() start, on throwaway clones, with the RNG restored after — compiling
        must not shift the training sample stream (SB3 parity)."""
        if self._primed or not self.config.compile:
            self._primed = True
            return
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None
        obs, state = obs.clone(), state.clone()
        with torch.no_grad():
            self.policy.prepare_step(obs)
            torch.compiler.cudagraph_mark_step_begin()
            self._rollout_fn(obs, state)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        self._primed = True

    # ── rollout collection ────────────────────────────────────────────

    def collect_rollouts(self, current_state, current_obs, episode_starts, current_episode_reward,
                         current_episode_length, episode_rewards, episode_lengths,
                         on_step_callback=None):
        """Collect n_steps of experiences: per step, the eager ``prepare_step``
        prelude → the compiled ``rollout_step`` → the value head → the SB3
        truncation bootstrap → buffer add → deferred stat recording."""
        self.policy.eval()
        self.rollout_buffer.reset()
        state, obs = current_state, current_obs

        with torch.no_grad():
            for n in range(self.n_steps):
                obs_snap = obs.clone()   # snapshot: cudagraph replays overwrite outputs in place
                self.policy.prepare_step(obs_snap)
                torch.compiler.cudagraph_mark_step_begin()
                new_obs, new_state, step_out, actions, log_probs = self._rollout_fn(obs_snap, state.clone())
                values = self.policy.predict_values(obs_snap)

                # SB3-faithful truncation bootstrap (sb3_ppo): an episode cut off by the depth
                # limit (truncated, not naturally terminated) keeps a future — add
                # gamma * V(terminal_obs) to the BUFFERED reward so GAE's done-masking doesn't
                # zero the cut-off value. predict_values reads only obs['sub_index'] (= the
                # terminal current state); non-truncated envs are masked out.
                terminal_obs = TensorDict(
                    {'sub_index': step_out.final_observation.unsqueeze(1)}, batch_size=[self.n_envs])
                terminal_values = self.policy.predict_values(terminal_obs).flatten()
                buffer_rewards = step_out.step_rewards + self.gamma * terminal_values * step_out.step_truncated.float()

                self.rollout_buffer.add(
                    obs=obs_snap, action=actions, reward=buffer_rewards,
                    episode_start=episode_starts, value=values.flatten(), log_prob=log_probs,
                )

                # Episode stats: accumulate on GPU, zero finished lanes, defer the CPU sync.
                current_episode_reward += step_out.step_rewards
                current_episode_length += 1
                done_mask = step_out.step_dones.bool()
                self._step_dones[n] = done_mask
                self._step_rewards[n] = torch.where(done_mask, current_episode_reward, torch.zeros_like(current_episode_reward))
                self._step_lengths[n] = torch.where(done_mask, current_episode_length, torch.zeros_like(current_episode_length))
                if on_step_callback is not None:
                    for name, fn in self._stat_fns.items():
                        self._stat_bufs[name][n] = fn(step_out, new_state)
                current_episode_reward.masked_fill_(done_mask, 0.0)
                current_episode_length.masked_fill_(done_mask, 0)
                episode_starts = step_out.step_dones.float()
                state, obs = new_state, new_obs

            last_values = self.policy.predict_values(obs)

        self._flush_episode_stats(episode_rewards, episode_lengths, on_step_callback)
        # episode_starts carries the last step's step_dones.
        self.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=episode_starts)

        return state, obs, episode_starts, self.n_steps * self.n_envs

    def _flush_episode_stats(self, episode_rewards: list, episode_lengths: list, on_step_callback) -> None:
        """Single GPU→CPU sync to transfer the rollout's deferred episode stats;
        the optional extras reach the callback manager as keyword arrays by name."""
        idx = self._step_dones.flatten().nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return
        cols = {'rewards': self._step_rewards.flatten()[idx],
                'lengths': self._step_lengths.flatten()[idx].float()}
        if on_step_callback is not None:
            cols.update({name: buf.flatten()[idx] for name, buf in self._stat_bufs.items()})
            cols['_env'] = (idx % self.n_envs).float()
        cpu = dict(zip(cols, torch.stack(list(cols.values()), dim=1).cpu().numpy().T))
        episode_rewards.extend(cpu['rewards'].tolist())
        episode_lengths.extend(cpu['lengths'].astype(int).tolist())
        if on_step_callback is None:
            return
        env_idx = cpu.pop('_env').astype(int)
        # Per done EVENT (not per env): post-step ptr - 2 = the episode's query-pool index
        # (draw stores ptr+1; the done step's auto-reset advanced once more; mod = wrap).
        pool_size = int(self.env.query_pool.shape[0])
        query_ptrs = (cpu.pop('_query_ptrs').astype(int) - 2) % pool_size
        stats = {k: (v.astype(bool) if k == 'successes' else v.astype(int))
                 for k, v in cpu.items() if k not in ('rewards', 'lengths')}
        on_step_callback(rewards=cpu['rewards'], lengths=cpu['lengths'].astype(int),
                         done_idx_cpu=env_idx, episode_query_indices=query_ptrs,
                         **stats)

    # ── training ──────────────────────────────────────────────────────

    def train(self) -> Dict[str, Any]:
        """Update policy from rollout buffer using the compiled loss."""
        self.policy.train()
        n_batches = (self.n_steps * self.n_envs) // self.batch_size
        total = self.n_epochs * n_batches

        pg_losses = torch.zeros(total, device=self.device)
        vl_losses = torch.zeros(total, device=self.device)
        ent_losses = torch.zeros(total, device=self.device)
        clips = torch.zeros(total, device=self.device)
        kls = torch.zeros(total, device=self.device)

        batch_count = 0
        continue_training = True
        last_epoch_start = 0

        for epoch in range(self.n_epochs):
            last_epoch_start = batch_count
            for batch in self.rollout_buffer.get():
                torch.compiler.cudagraph_mark_step_begin()
                _, _, _, _, advantages, returns = batch

                if self.normalize_returns and returns.numel() > 1:
                    returns.sub_(returns.mean()).div_(returns.std() + 1e-8)
                if self.normalize_advantage and advantages.numel() > 1:
                    advantages.sub_(advantages.mean()).div_(advantages.std() + 1e-8)

                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    loss, policy_loss, value_loss, entropy_loss, approx_kl_div, clip_fraction = self.loss_step(batch)

                pg_losses[batch_count] = policy_loss.detach()
                vl_losses[batch_count] = value_loss.detach()
                ent_losses[batch_count] = entropy_loss.detach()
                kls[batch_count] = approx_kl_div.detach()
                clips[batch_count] = clip_fraction.detach()

                batch_count += 1

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    if self.verbose:
                        print(f"[PPO] Early stopping at epoch {epoch}, batch {batch_count} due to KL: {approx_kl_div.item():.4f}")
                    continue_training = False
                    break

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm:
                    fused_clip_grad_norm_(self._cached_params, self.max_grad_norm)
                self.optimizer.step()

            if not continue_training:
                break
            if self.verbose:
                self._epoch_end_indices.append((epoch, batch_count, loss.detach().clone()))

        return compute_train_metrics(
            self, pg_losses, vl_losses, ent_losses, clips, kls,
            batch_count, last_epoch_start,
        )

    def _update_schedules(self, total_timesteps: int) -> None:
        """Anneal lr / ent_coef per config (no-ops unless ``lr_decay`` / ``ent_coef_decay``).

        The algorithm owns its schedules (SB3's ``_update_learning_rate`` idiom): they
        are training semantics and must run regardless of which callbacks are attached.
        ``_ent_coef_buf`` is written in-place so the compiled loss reads the current value.
        """
        config = self.config
        progress = min(1.0, max(0.0, self.num_timesteps / total_timesteps))
        if config.lr_decay:
            warmup = config.lr_warmup_steps if config.lr_warmup else 0.0
            lr = float(_anneal(progress, config.lr_init_value, config.lr_final_value,
                               config.lr_start, config.lr_end, config.lr_transform, warmup))
            self.learning_rate = lr
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        if config.ent_coef_decay:
            val = float(_anneal(progress, config.ent_coef_init_value, config.ent_coef_final_value,
                                config.ent_coef_start, config.ent_coef_end, config.ent_coef_transform))
            self.ent_coef = val
            self._ent_coef_buf.fill_(val)

    def learn(self, total_timesteps, reset_num_timesteps=True):
        """Main loop: alternates collect_rollouts() and train() until total_timesteps."""
        if reset_num_timesteps:
            self.num_timesteps = 0
        else:
            total_timesteps += self.num_timesteps
        self.iteration = 0

        if self.callback is not None:
            self.callback.init_callback(self)
            self.callback.on_training_start(total_timesteps)

        if total_timesteps <= 0:
            # Eval-only: no training steps. on_training_start has fired (so eval
            # callbacks initialize); skip the rollout machinery entirely.
            return {}

        self.env.set_queries(self.env.train_queries)  # pool pre-shuffled in the data loader
        state = self.env.reset_pool()      # stateless bootstrap from the pool (no gym wrapper)
        obs = self.env.observation(state)

        ep_starts = torch.ones(self.n_envs, dtype=torch.float32, device=self.device)
        curr_ep_rew = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
        curr_ep_len = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        ep_rews, ep_lens = [], []
        iteration = 0

        self._warmup_rollout(obs, state)

        while self.num_timesteps < total_timesteps:
            iteration += 1
            self.iteration = iteration
            self._update_schedules(total_timesteps)

            step_cb = self.callback.prepare_batch_infos if self.callback is not None else None

            if self.callback is not None:
                self.callback.on_iteration_start()

            rollout_start_time = time.time()
            n_ep_before = len(ep_rews)
            state, obs, ep_starts, n_steps = self.collect_rollouts(
                state, obs, ep_starts, curr_ep_rew, curr_ep_len, ep_rews, ep_lens,
                on_step_callback=step_cb)
            state = state.clone()
            obs = obs.clone()

            self.num_timesteps += n_steps
            rollout_time = time.time() - rollout_start_time
            # Mean episode reward from this iteration's rollout — what PPO actually optimizes.
            # Into the shared metrics dict so callbacks/diagnostics can compare it vs MRR.
            new_rews = ep_rews[n_ep_before:]
            rollout_reward = sum(new_rews) / len(new_rews) if new_rews else 0.0
            self.last_metrics['rollout_reward'] = rollout_reward
            # Diagnosis-only timing lines — the metrics callback logs the compact
            # per-iteration summary (steps/fps/reward) either way.
            log_diag = self.verbose and getattr(self.config, "log_diagnostics", False)
            if log_diag:
                print(f"[PPO] Rollout collected in {rollout_time:.2f}s. Rollout FPS: {n_steps/rollout_time:.2f}. Reward: {rollout_reward:.3f}. Timesteps: {self.num_timesteps}")

            train_start_time = time.time()
            self.last_metrics.update(self.train())  # train/* losses into the shared dict
            if log_diag:
                print(f"[PPO] Training completed in {time.time() - train_start_time:.2f}s")

            # Callbacks read/contribute to self.last_metrics; on_iteration_end returns
            # False to stop (early-stop / pruning).
            if self.callback is not None and not self.callback.on_iteration_end():
                print(f"[PPO] Training stopped by callback at {self.num_timesteps} timesteps")
                break

        if self.callback is not None:
            self.callback.on_training_end()

        return {
            'num_timesteps': self.num_timesteps, 'episode_rewards': ep_rews,
            'episode_lengths': ep_lens, 'last_metrics': dict(self.last_metrics),
        }
