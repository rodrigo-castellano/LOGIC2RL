"""Stateless core of the vectorized SLD-reasoning environment (pillar: base).

``FuncEnv`` is the batched, compiled transition algebra the rollout threads explicit ``state``
through — no ``self._state``, no gym spaces. The stateful gymnasium facade lives next door in
``env.py`` (``GymVecEnvWrapper``), which wraps a ``FuncEnv`` by composition.

The split mirrors gymnasium's own ``FuncEnv`` + ``FunctionalJaxVectorEnv(func_env)`` pairing:
the compile boundary becomes a type boundary (you cannot touch ``_state`` here because it
isn't here), and KGE specialization is a subclass (``KGEFuncEnv``) so the facade stays generic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from logic2rl.unification import Grounder

from .component import TransitionCtx
from .unification_logic import UnificationLogic

if TYPE_CHECKING:
    from logic2rl.data_loader import MaterializedData

Tensor = torch.Tensor


class EnvState(NamedTuple):
    """Core recurrent proof-search state — the fields the rollout always needs.

    The composed ``env.State`` (built in ``_compose_state``) is this plus every component's
    ``declare_state_fields`` (memory's visit-history, the kge per-query fields / counters / rule
    id); the auto-reset splice + ``clone`` are generic over the fields, so components mix in for
    free without base knowing them.
    """
    current_states: Tensor          # [B, A, W]
    derived_states: Tensor          # [B, G, A, W]  candidate next states (actions)
    derived_counts: Tensor          # [B]  number of valid candidates
    original_queries: Tensor        # [B, A, W]  the query being proved (self-exclusion)
    next_var_indices: Tensor        # [B]  fresh runtime-var allocator
    depths: Tensor                  # [B]  step count (drives depth-limit truncation)
    done: Tensor                    # [B] uint8  cumulative: is the env currently finished
    success: Tensor                 # [B] uint8
    per_env_ptrs: Tensor            # [B]  round-robin pointer into the query pool

    def clone(self) -> "EnvState":
        """Clone every tensor (CUDA-graph safe); returns the same subclass type."""
        return type(self)(*(t.clone() for t in self))


class StepOutput(NamedTuple):
    """Per-transition outputs a step *emits* (never read back by ``step_core``).

    Returned alongside the next ``EnvState`` so the completed step survives the same-step
    auto-reset (which overwrites recurrent fields for done envs). Mirrors SB3's
    ``(reward, done, info)`` plus the gym truncation split. The composed ``self.StepOutput``
    (built in ``_compose_state``) appends each component's ``declare_trace_fields`` (e.g. a kge
    per-query label, the proof ``step_endf``); fixed at build → cudagraph-static.

    ``final_observation`` [B, A, W] is the terminal current state (obs ``sub_index`` is this with
    an inserted slot axis, [B, 1, A, W]): the truncation bootstrap reads it and the gym facade
    surfaces it as ``final_obs``.
    """
    step_rewards: Tensor            # [B]
    step_dones: Tensor              # [B] uint8  edge: did the env finish THIS step
    step_truncated: Tensor          # [B] uint8  edge: finished THIS step by depth-limit truncation
    is_success: Tensor              # [B] uint8
    original_queries: Tensor        # [B, A, W]
    final_observation: Tensor       # [B, A, W]  terminal current state (obs sub_index = this, unsqueezed to [B,1,A,W])


# The policy's view — a censored projection of EnvState (via observation): only the proof-search
# frontier the agent acts on. Excludes internal bookkeeping (component fields, history, pointers).
EnvObs = TensorDict


def make_observation_space(padding_atoms: int, padding_states: int,
                           total_vocab_size: int, atom_width: int = 3) -> "gym.spaces.Dict":
    """The single-env observation space. Built by ``GymVecEnvWrapper.__init__`` from the core's
    dims (then merged with each component's ``obs_space_extra``) so the space matches the obs.
    ``atom_width`` = max_arity + 1 (3 for binary atoms)."""
    high = int(total_vocab_size) or (2 ** 31 - 1)  # upper bound for token ids
    A, G, W = padding_atoms, padding_states, atom_width
    return gym.spaces.Dict({
        "sub_index": gym.spaces.Box(0, high, shape=(1, A, W), dtype=np.int64),
        "derived_sub_indices": gym.spaces.Box(0, high, shape=(G, A, W), dtype=np.int64),
        "action_mask": gym.spaces.Box(0, 1, shape=(G,), dtype=np.uint8),
    })


class FuncEnv:
    """Stateless, batched, compiled proof-search transition algebra (pillar: base).

    The rollout threads an explicit ``state`` through ``step_autoreset`` / ``observation``
    (CUDA-graph friendly); nothing here holds ``self._state`` — that lives on the
    ``GymVecEnvWrapper`` facade. Owns the proof-rollout flow, buffers, the base reward, SLD
    candidate generation, and the pluggable components (visit-memory, unary advance, the task's
    own) whose hooks base invokes generically. The two step outputs are the recurrent next
    ``EnvState`` and the per-transition ``StepOutput``. KGE specialization (negative sampling,
    the 4-way reward) is the ``KGEFuncEnv`` subclass; base never imports kge.
    """

    def __init__(
        self,
        vec_engine: Grounder,
        *,
        # Core dimensions
        batch_size: int = 100,
        padding_atoms: int = 6,
        padding_states: int = 120,
        max_depth: int = 20,
        max_arity: int = 2,
        device: Optional[torch.device] = None,
        # Vocab indices — received from the IndexManager (via build_env), NOT read off the engine:
        # the IM is the single source of indices; the engine stays the resolution engine.
        # ``constant_no`` fixes the runtime-var start (constant_no+1). ``special_pred_ids`` is the
        # IM's name→id map of special predicates ({'True','False','Endf',…}); the env uses
        # True/False for its termination and carries the rest for components to look up by name.
        padding_idx: int,
        constant_no: int,
        special_pred_ids: Dict[str, int],
        # Data + indexing
        data: "MaterializedData",
        # Pluggable feature components (visit-memory, unary advance, the task's own, …); each
        # owns its config and contributes recurrent state fields + hooks. Base composes the
        # state type and invokes their hooks generically, so it never hardcodes any one feature.
        components: Tuple[Any, ...] = (),
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dimensions. atom_width (W = max_arity + 1) is the per-atom literal width:
        # [predicate, arg1, …, arg_{max_arity}]. W = 3 for binary atoms (KGE); a ternary
        # task (e.g. MNIST add/3) uses W = 4. Set before the components/buffers that size on it.
        self.batch_size = batch_size
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth
        self.max_arity = max_arity
        self.atom_width = max_arity + 1

        # Engine — used for resolution OPERATIONS only, NOT as an index source.
        self.engine = vec_engine
        # Vocab indices come from the IndexManager: the env extracts the True/False it uses for the
        # invariant proof-termination (all-TRUE ⇒ success, any-FALSE ⇒ fail) and carries the rest
        # for components (e.g. the kge endf component looks up 'Endf'). Runtime vars start past constants.
        self.padding_idx = padding_idx
        self.constant_no = constant_no
        self.runtime_var_start_index = constant_no + 1
        self.special_pred_ids = dict(special_pred_ids)
        self.true_pred_idx = self.special_pred_ids.get('True')
        self.false_pred_idx = self.special_pred_ids.get('False')
        assert self.true_pred_idx is not None and self.false_pred_idx is not None, \
            "FuncEnv is a proof-search env: it requires True/False predicate ids."

        # Query pools, extracted from the materialized bundle. Each split's queries[:, 0] is the
        # query atom [N, W]; an empty valid split falls back to the test pool. Per-query metadata
        # (depths, outcomes) is a task concern — a task attaches what it needs via build_env's post_build
        # (the kge env loads depths there). The active pool + pointers are set later by set_queries.
        q = lambda split: split.queries[:, 0].to(self.device) if len(split) > 0 else None
        has_valid = len(data.valid) > 0
        self.train_queries = q(data.train)
        self.test_queries = q(data.test)
        self.valid_queries = q(data.valid) if has_valid else self.test_queries
        self.query_pool = None

        # Candidate generation (pure mechanism; feature behavior lives in the components).
        self._allocate_buffers()
        self.unification_logic = UnificationLogic(self)

        # Pluggable components: eager setup, then compose the recurrent state type from the core
        # fields + every component's declared fields (fixed at build → one specialized compiled
        # graph). Base invokes their hooks generically; it owns no feature logic.
        self.components = tuple(components)
        for c in self.components:
            c.setup(self)
        # Terminal (non-continuable) predicate ids: the env's own proof outcomes {True, False} +
        # any component contributions (e.g. the endf early-stop). Fixed at build (python ints → the
        # consult loop unrolls under torch.compile).
        self.terminal_pred_ids = tuple(
            p for p in (self.true_pred_idx, self.false_pred_idx) if p is not None
        ) + tuple(p for c in self.components for p in c.declare_terminal_pred_ids(self) if p is not None)
        self._compose_state()

    def _compose_state(self) -> None:
        """Compose the recurrent ``State`` + ``StepOutput`` types from the core fields + every
        component's ``declare_state_fields`` / ``declare_trace_fields`` (fixed at build).

        With no components ``State`` is exactly ``EnvState``; a flat NamedTuple keeps
        ``state.<field>`` accessors plus the generic auto-reset splice / ``clone`` working when
        fields are added. A new component adds a field just by declaring it (no facade edits).
        """
        specs = [fs for c in self.components for fs in c.declare_state_fields(self)]
        if specs:
            self.State = NamedTuple("EnvState", [(f, Tensor) for f in EnvState._fields]
                                    + [(fs.name, Tensor) for fs in specs])
            self.State.clone = EnvState.clone
        else:
            self.State = EnvState
        self._field_init = {fs.name: fs.init for fs in specs}
        # Names of component-contributed fields (read by the candidate-gen orchestrator to seed
        # its working ``fields`` dict from the prior state).
        self._component_fields = [fs.name for fs in specs]

        trace = [n for c in self.components for n in c.declare_trace_fields(self)]
        if trace:
            self.StepOutput = NamedTuple(
                "StepOutput", [(f, Tensor) for f in StepOutput._fields] + [(n, Tensor) for n in trace])
        else:
            self.StepOutput = StepOutput

    # =========================================================================
    # STEP (stateless; the compiled rollout calls these directly)
    # =========================================================================

    def step_core(self, state: EnvState, actions: Tensor) -> Tuple[EnvState, "StepOutput"]:
        """Bare stateless step: each env steps into its chosen candidate → (next_state, step_output).

        Applies the action, scores the transition + terminal outcome, regenerates candidates
        (the grounder), freezes done envs, and emits the per-transition ``StepOutput`` beside the
        recurrent next ``EnvState``. **No auto-reset** — used by eval / scoring (done envs stay
        done). The training rollout + gym facade use ``step_autoreset``.
        """
        assert actions.dim() == 1 and actions.shape[0] == self.batch_size, \
            f"actions must be [B={self.batch_size}], got shape {actions.shape}"

        active, new_current, new_depths = self._apply_action(state, actions)
        truncated, is_success, is_end, step_done = self._terminal(state, new_current, new_depths, active)
        rewards = self._reward(state, step_done, is_success, active)

        # Cumulative bookkeeping: edge-done this step, done-by-now, success-by-now, still-active.
        newly_done = active & step_done
        new_done = state.done.bool() | newly_done
        new_success = state.success.bool() | (active & is_success)
        still_active = ~new_done

        cand = self._derive(state, new_current, new_depths, still_active)

        # Assemble the recurrent next state (+ component-owned field updates via step_commit_fields).
        updates = dict(
            current_states=cand.current_states, derived_states=cand.derived, derived_counts=cand.counts,
            next_var_indices=cand.next_var, depths=new_depths,
            done=new_done.to(torch.uint8), success=new_success.to(torch.uint8),
        )
        updates.update(cand.fields)   # component-owned fields seeded during candidate-gen (e.g. memory)
        for c in self.components:
            updates.update(c.step_commit_fields(self, cand, state, still_active))
        new_state = state._replace(**updates)

        # Per-transition snapshot (survives the same-step auto-reset): canonical gym returns + base
        # info fields + every component's trace fields. final_observation = the terminal current
        # state (= obs sub_index), read by the truncation bootstrap / surfaced as final_obs.
        trace = dict(
            step_rewards=rewards,
            step_dones=newly_done.to(torch.uint8),
            step_truncated=(active & truncated).to(torch.uint8),
            is_success=new_success.to(torch.uint8),
            original_queries=state.original_queries,
            final_observation=cand.current_states,
        )
        ctx = TransitionCtx(actions=actions, active=active, newly_done=newly_done,
                            truncated=truncated, is_success=is_success, is_end=is_end)
        for c in self.components:
            trace.update(c.step_trace(self, state, new_state, ctx))
        return new_state, self.StepOutput(**trace)

    def _apply_action(self, state: EnvState, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Step each active env into its chosen derived candidate; advance its depth.

        Done envs are frozen (keep their current state + depth). Returns
        ``(active, new_current, new_depths)``.
        """
        B = self.batch_size
        active = ~state.done.bool()                      # was still searching when the step began
        chosen = state.derived_states[self._batch_arange, actions]            # [B, A, W]
        new_current = torch.where(active.view(B, 1, 1), chosen, state.current_states)
        new_depths = torch.where(active, state.depths + 1, state.depths)
        return active, new_current, new_depths

    def _terminal(self, state: EnvState, new_current: Tensor, new_depths: Tensor,
                  active: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-env terminal outcome of the new state. Returns ``(truncated, is_success, is_end, step_done)``.

        Proof core (env-owned): every non-pad atom TRUE ⇒ success; any FALSE atom ⇒ fail; an empty
        resolvent terminates. Components fold in their own semantics (the endf early-stop). Finally
        truncate at the depth limit (disjoint from termination).
        """
        B = self.batch_size
        non_pad = new_current[:, :, 0] != self.padding_idx
        preds = new_current[:, :, 0]                                   # [B, A] per-atom predicate
        all_true = ((preds == self.true_pred_idx) | ~non_pad).all(dim=1) & non_pad.any(dim=1)
        terminated = ~non_pad.any(dim=1) | all_true | (preds == self.false_pred_idx).any(dim=1)
        is_success = all_true
        is_end = torch.zeros(B, dtype=torch.bool, device=self.device)
        for c in self.components:
            terminated, is_success, is_end = c.step_terminal_outcome(self, new_current, non_pad, terminated, is_success, is_end)
        truncated = (new_depths >= self.max_depth) & ~terminated
        step_done = terminated | truncated
        return truncated, is_success, is_end, step_done

    def _reward(self, state: EnvState, step_done: Tensor, is_success: Tensor, active: Tensor) -> Tensor:
        """Transition reward for the active envs (zero for already-done envs).

        Base: +1 for a successful terminal transition, else 0 — no positive/negative
        distinction (a base task such as MNIST-addition has no such concept; ``state`` is
        unused). The task override seam: ``KGEFuncEnv._reward`` scores negative-labeled
        queries with the 4-way reward, reading the per-query label off ``state``.
        """
        return torch.where(active & step_done & is_success, self._reward_pos, self._reward_zero)

    def _derive(self, state: EnvState, new_current: Tensor, new_depths: Tensor,
                still_active: Tensor) -> Any:
        """Regenerate the candidate next-states for the new current state.

        Candidate-gen (visit-history update + unification + skip-unary advance) is delegated to
        the ``unification_logic`` component; done envs are frozen on their previous derived set,
        then components may transform the candidates (e.g. terminal-actions force {true,false} at
        max depth). No-op transform when no component owns one.
        """
        B = self.batch_size
        cand = self.unification_logic._compute_derived(new_current, state)
        cand = cand._replace(
            derived=torch.where(still_active.view(B, 1, 1, 1), cand.derived, state.derived_states),
            counts=torch.where(still_active, cand.counts, state.derived_counts),
            next_var=torch.where(still_active, cand.next_var, state.next_var_indices),
        )
        for c in self.components:
            cand = c.step_transform_candidates(self, cand, state, new_depths, still_active)
        return cand

    def step_autoreset(self, state: EnvState, actions: Tensor) -> Tuple[EnvState, "StepOutput"]:
        """Stateless step + same-step auto-reset of the finished envs → (next_state, step_output).

        The training / gym-facade step: runs ``step_core``, then re-initializes the envs that
        finished this step. The reset draw reads the PRE-step context (the cursor + the kge
        corruption counters the step itself advances); its fresh initial state is spliced into the
        post-step state — finished envs take the reset, active envs keep theirs. The splice is
        generic over the fields, so subclass fields (kge per-query / counters) ride along;
        ``step_output`` is the completed step's output (it survives the recurrent-state splice).
        """
        next_state, step_output = self.step_core(state, actions)
        done_mask = step_output.step_dones.bool()
        # Draw fresh queries for the finished envs (round-robin + component corruption) and build
        # their initial state. The draw gets only the context fields it reads (the kge corruption
        # counters), not the whole state. reset_core takes the advanced cursor.
        context = {f: getattr(state, f) for f in self._component_fields}
        queries, new_ptrs, extra = self.draw_queries(self.query_pool, state.per_env_ptrs, done_mask, context=context)
        reset_state = self.reset_core(queries, per_env_ptrs=new_ptrs, **extra)
        spliced = {}
        for field in type(next_state)._fields:
            r_val, n_val = getattr(reset_state, field), getattr(next_state, field)
            m = done_mask
            for _ in range(r_val.ndim - 1):
                m = m.unsqueeze(-1)
            spliced[field] = torch.where(m, r_val, n_val)
        return type(next_state)(**spliced), step_output

    # =========================================================================
    # RESET (stateless)
    # =========================================================================

    def reset_core(self, queries: Tensor, *,
                   per_env_ptrs: Optional[Tensor] = None, **extra) -> EnvState:
        """Build the initial state for fresh ``queries`` [B, W].

        Generates the initial candidate set (owned by the unification component) and seeds
        component state from the candidate bundle + the draw context ``extra`` (e.g. the kge
        per-query fields / counters). Returns the state; project to obs via ``observation(state)``.
        """
        device = self.device
        A, pad = self.padding_atoms, self.padding_idx
        B = queries.shape[0]
        assert B == self.batch_size, f"reset expects queries for all {self.batch_size} envs, got {B}."

        W = self.atom_width
        padded = torch.full((B, A, W), pad, dtype=torch.long, device=device)
        padded[:, 0, :] = queries.to(device)  # [B, W] → first atom slot
        queries = padded  # [B, A, W]

        cand = self.unification_logic._compute_initial(queries)

        zeros_long = torch.zeros(B, dtype=torch.long, device=device)
        core = dict(
            current_states=cand.current_states,
            derived_states=cand.derived.clone(),
            derived_counts=cand.counts,
            original_queries=queries,
            next_var_indices=cand.next_var,
            depths=zeros_long,
            done=torch.zeros(B, dtype=torch.uint8, device=device),
            success=torch.zeros(B, dtype=torch.uint8, device=device),
            per_env_ptrs=per_env_ptrs if per_env_ptrs is not None else zeros_long,
        )
        core.update(cand.fields)   # component-owned fields seeded during candidate-gen (e.g. memory)
        for c in self.components:
            core.update(c.reset_commit_fields(self, core, cand, extra))
        # Safety net: any declared component field a component didn't seed above falls back to its
        # FieldSpec.init default — so a component can declare a state field without hand-seeding it.
        for name, init in self._field_init.items():
            if name not in core:
                core[name] = init(self, B)
        return self.State(**core)

    def reset_pool(self, prev_state: "Optional[EnvState]" = None) -> EnvState:
        """Reset ALL envs from the active query pool → the initial bootstrap state.

        Restarts the round-robin cursor at the top of the pool (defaulting to the training pool if
        ``set_queries`` was never called). ``prev_state`` carries the draw context (e.g. the kge
        corruption counters) — ``None`` on a cold start. The stateless twin of the gym facade's
        ``reset()``: the rollout starts here and threads the state through ``step_autoreset``.
        """
        if self.query_pool is None:
            if self.train_queries is None:
                raise RuntimeError("No query pool: call set_queries(queries) first.")
            self.set_queries(self.train_queries)
        B = self.batch_size
        cursor = torch.arange(B, dtype=torch.long, device=self.device)
        all_envs = torch.ones(B, dtype=torch.bool, device=self.device)
        context = {} if prev_state is None else {f: getattr(prev_state, f) for f in self._component_fields}
        queries, new_ptrs, extra = self.draw_queries(self.query_pool, cursor, all_envs, context=context)
        return self.reset_core(queries, per_env_ptrs=new_ptrs, **extra)

    # =========================================================================
    # OBSERVATION (the policy's censored view of the state)
    # =========================================================================

    def observation(self, state) -> EnvObs:
        """Project ``EnvState`` to the policy's TensorDict obs: the proof frontier the agent acts on.

        current state + candidate next-states + action mask (+ component keys via ``obs_extra``);
        internal bookkeeping (component fields, history, depth, pointers) is never exposed.
        """
        B = state.current_states.shape[0]
        mask = self._slot_arange.unsqueeze(0) < state.derived_counts.unsqueeze(1)
        obs = {
            'sub_index': state.current_states.unsqueeze(1),
            'derived_sub_indices': state.derived_states,
            'action_mask': mask.to(torch.uint8),
        }
        for c in self.components:
            obs.update(c.obs_extra(self, state))   # component obs keys (e.g. kge rule id)
        return TensorDict(obs, batch_size=[B])

    # =========================================================================
    # QUERY DRAW
    # =========================================================================

    def set_queries(self, queries: Tensor) -> None:
        """Set the active query pool [N, W] (callers pre-shuffle); the next reset draws from it.

        The round-robin draw position lives in the state (``per_env_ptrs``), not on the env.
        """
        self.query_pool = queries.to(self.device)

    def draw_queries(self, query_pool: Tensor, per_env_ptrs: Tensor, reset_mask: Tensor, *,
                    context: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Tensor, dict]:
        """Draw the next queries for the envs being (re)initialized → ``(queries, new_ptrs, extra)``.

        Base: sequential round-robin over ``query_pool`` via ``per_env_ptrs`` (callers pre-shuffle);
        only ``reset_mask`` envs advance and carry a real triple (others padded). ``context`` is the
        OPTIONAL draw-context the components may read (e.g. the kge corruption counters carried from
        the previous state; ``None`` on a fresh reset). ``extra`` is the OUTPUT draw context — reset
        seeds the components accumulate, read back by their ``reset_commit_fields``. Base adds nothing.
        """
        pool_size = query_pool.shape[0]
        W = self.atom_width
        safe_idx = per_env_ptrs % pool_size
        next_ptrs = (per_env_ptrs + 1) % pool_size
        queries = query_pool[safe_idx]
        padding = torch.full((W,), self.padding_idx, dtype=torch.long, device=self.device)
        queries = torch.where(reset_mask.unsqueeze(-1).expand(-1, W), queries, padding)  # [B, W]
        new_ptrs = torch.where(reset_mask, next_ptrs, per_env_ptrs)
        extra: dict = {}
        for c in self.components:
            queries, seeds = c.reset_draw_queries(self, queries, reset_mask, context)
            extra.update(seeds)
        return queries, new_ptrs, extra

    # =========================================================================
    # BUFFERS
    # =========================================================================

    def _allocate_buffers(self):
        """Pre-allocate the env's persistent tensors (cudagraph-stable addresses): the base
        reward scalars and the generic index helpers. Components and the candidate generation
        allocate their own extra buffers (e.g. the compaction scratch)."""
        B, G = self.batch_size, self.padding_states
        device = self.device

        # Base reward scalars: success → _reward_pos, else _reward_zero (KGE adds its own).
        self._reward_pos = torch.tensor(1.0, device=device)
        self._reward_zero = torch.tensor(0.0, device=device)

        # Index helpers reused every step. _slot_arange spans the G derived slots; the
        # _batch_* helpers span the B envs.
        self._slot_arange = torch.arange(G, device=device)
        self._batch_arange = torch.arange(B, device=device)
        self._batch_ones = torch.ones(B, dtype=torch.long, device=device)
        self._batch_zeros = torch.zeros(B, dtype=torch.long, device=device)
