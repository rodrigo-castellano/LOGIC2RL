"""Env components — the extension API (pillar: base).

A *component* is a pluggable env feature that OWNS ITS CONFIG (constructor args). The env
core (`FuncEnv`) is a thin compiled frame; features like visit-memory, the unary advance,
and the KGE components (endf, rule-id, terminal-actions, negatives) are added as components,
so the base never hardcodes them and a user enables one by listing it
(the builder's component set, e.g. ``base.builder.make_base_components``).

Recurrent STATE fields (carried step→step, live in the composed EnvState) are resolved
ONCE at construction (so torch.compile sees a single fixed state type + a fixed hook order
→ one specialized graph, cudagraph-safe): declared by ``declare_state_fields``, seeded by
``reset_commit_fields``, updated by ``step_commit_fields``. (The engine's per-slot
``derived_rule_idx`` rides ``derived``'s compaction in the unification component; the
RuleIdComponent freezes it into a state field.)

All hooks except ``setup`` run INSIDE the compiled step and must be Dynamo-traceable:
fixed-shape tensor ops, no ``.item()``, no python branch on tensor values (use
``torch.where``), no ``torch.cond``. "Optional" = component present-or-absent (a
compile-time fact), never a runtime conditional or a None-valued tensor field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, NamedTuple, Tuple

import torch

Tensor = torch.Tensor


class TransitionCtx(NamedTuple):
    """The completed transition, as ``step_core`` saw it — the read-only context handed to
    ``step_trace`` so a component can classify the step without re-deriving core outcomes
    (or reverse-engineering them from state diffs). Intra-step only: built fresh each step,
    never stored, never part of ``StepOutput``.
    """
    actions: Tensor          # [B] long  the candidate slot each env stepped into
    active: Tensor           # [B] bool  was still searching when the step began
    newly_done: Tensor       # [B] bool  edge: finished THIS step
    truncated: Tensor        # [B] bool  raw depth-limit truncation (pre active-mask)
    is_success: Tensor       # [B] bool  raw step-level success (pre cumulative-fold)
    is_end: Tensor           # [B] bool  agent chose an early-stop action (e.g. endf)


@dataclass(frozen=True)
class FieldSpec:
    """Declares one recurrent state tensor a component adds to the composed EnvState.

    ``init(env, batch_size) -> Tensor`` builds the field's initial value (the reset
    default). It must ALWAYS return a real fixed-shape tensor (zeros when 'inactive') —
    never None — so the cudagraph address/shape is static. ``eval_roundtrip`` marks the
    field as one the compiled eval scorer must thread through its ``step_core`` loop;
    train-only fields (e.g. the kge corruption counters — eval has no auto-reset draw)
    set it False and the scorer ignores them.
    """
    name: str
    init: Callable[[Any, int], Tensor]
    eval_roundtrip: bool = True


class EnvComponent:
    """A pluggable env feature: extra recurrent state fields + traceable lifecycle hooks.

    Override only what the feature needs; every hook is a no-op by default.

    Hook names are ``<phase>_<action>`` — the prefix says WHEN the hook fires:
      ``declare_*``    build-time declarations (run once at construction, not traced)
      ``reset_*``      reset body only
      ``step_*``       step body only
      ``candidate_*``  the candidate-gen seam — runs in BOTH reset and step
    (``setup`` also runs once at construction; ``obs_extra`` builds the observation
    in both reset and step.)
    """

    name: str = "component"          #: stable id; components run in the order passed to the env

    # ---- eager, once, at construction (NOT traced) ----
    def setup(self, env: Any) -> None:
        """Allocate static buffers / cache constants off the env.

        ``setup`` = constants/buffers computed once (vs ``declare_state_fields`` = tensors that
        change every step and must ride the compiled state)."""

    # ---- recurrent state fields ----
    def declare_state_fields(self, env: Any) -> Tuple[FieldSpec, ...]:
        """The recurrent tensors this component adds to the composed EnvState (fixed at build).

        ``declare_state_fields`` = tensors that change every step and must ride the compiled state
        (vs ``setup`` = constants/buffers computed once)."""
        return ()

    # ---- candidate-generation hooks (called by the unification orchestrator) ----
    def reset_seed_fields(self, env: Any, queries: Tensor) -> Dict[str, Tensor]:
        """Seed this component's state fields at reset, BEFORE candidate generation (so the
        seed is available to ``candidate_keep_mask``) — e.g. memory seeds the visit-history
        ring from the initial queries. Returns {field_name: tensor}."""
        return {}

    def step_update_fields(self, env: Any, current: Tensor, fields: Dict[str, Tensor],
                      mask: Tensor) -> Dict[str, Tensor]:
        """Update this component's state fields from the just-entered ``current``, for the
        envs in ``mask``, BEFORE candidates are derived — e.g. memory appends the current
        state's hash. ``fields`` is the working field dict (seeded from the prior state);
        returns the updated entries. Invoked once per step (``mask=active``); the unary-advance
        re-derive updates memory directly via ``env.memory``, not through this hook."""
        return {}

    def candidate_keep_mask(self, env: Any, derived_raw: Tensor, fields: Dict[str, Tensor],
                            state) -> "Tensor | None":
        """Keep-mask [B, S] over the RAW (uncompacted) candidates — e.g. memory drops
        already-visited states. ANDed into the validity mask before the single compaction.
        ``fields`` holds this step's updated fields (from ``step_update_fields``/``reset_seed_fields``).
        Return None to keep all."""
        return None

    def candidate_refine(self, env: Any, cand, state):
        """Refine the compacted candidate bundle in the candidate-gen seam (pre-freeze), in
        component order — e.g. the unary component auto-advances forced single-successors, then
        the endf-action component appends the end-proof action. Returns a ``_Cand``. ``state`` is the
        pre-step state (``None`` at reset). Distinct from ``step_transform_candidates``, which runs
        post-freeze in the step (e.g. max-depth forcing)."""
        return cand

    # ---- traced hooks (inside the compiled step) ----
    def reset_draw_queries(self, env: Any, queries: Tensor, reset_mask: Tensor,
                        context: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Transform the freshly-drawn ``queries`` — e.g. KGE negative-sampling corrupts them
        — and return this component's reset seeds (its own per-query fields + counters).
        Returns ``(queries, seeds)``; the env merges all components' seeds into the draw
        context handed to ``reset_commit_fields`` as ``seed``. ``context`` is the optional
        draw-context — the previous state's per-env component fields (empty on the first
        reset, so carried counters default to zero)."""
        return queries, {}

    def reset_commit_fields(self, env: Any, core: Dict[str, Tensor], cand, seed: Dict[str, Tensor]
                 ) -> Dict[str, Tensor]:
        """Seed this component's state fields for a (re)initialized env → {name: tensor}.
        ``core`` is the base field dict, ``cand`` the initial candidate bundle (carries the
        per-candidate derivation ``tags``), ``seed`` the draw context (the ``reset_draw_queries``
        extras — generic; a component reads its own keys)."""
        return {}

    def step_transform_candidates(self, env: Any, cand, state, new_depths: Tensor,
                             still_active: Tensor):
        """Optionally rewrite the (frozen) candidate bundle — e.g. terminal-actions
        replace the candidate set at max depth. Returns a ``_Cand`` (same shapes)."""
        return cand

    def step_commit_fields(self, env: Any, cand, state, still_active: Tensor) -> Dict[str, Tensor]:
        """Produce recurrent state-field updates from the final candidates — e.g. rule-id
        freezes ``cand.derived_rule_idx`` into its state field for done envs → {name: tensor}."""
        return {}

    def declare_terminal_pred_ids(self, env: Any) -> Tuple[int, ...]:
        """Predicate ids this component treats as TERMINAL (non-continuable). The env unions
        these with its own proof outcomes ({True, False}) into ``env.terminal_pred_ids``, which
        ``UnaryAdvanceComponent`` (don't auto-advance into a terminal) and ``VisitMemoryComponent``
        (a terminal atom hashes canonically) consult — so neither names a predicate. Fixed at
        build. Example: the endf early-stop returns ``(endf_pred_idx,)``."""
        return ()

    def step_terminal_outcome(self, env: Any, states: Tensor, non_pad: Tensor,
                         terminated: Tensor, is_success: Tensor, is_end: Tensor):
        """Refine the transition's terminal outcome → ``(terminated, is_success, is_end)`` [B] bool.

        The base step seeds the proof-core outcome (``terminated`` = empty resolvent OR any-FALSE
        OR all-TRUE; ``is_success`` = all-TRUE; ``is_end`` = ``False``) and folds each component's
        ``step_terminal_outcome`` in turn. A component adds ITS OWN terminal semantics on top — e.g.
        the endf early-stop sets ``is_end`` and clears its own ``is_success``. ``non_pad`` is
        ``states[:, :, 0] != padding`` [B, A] (precomputed once)."""
        return terminated, is_success, is_end

    def obs_extra(self, env: Any, state) -> Dict[str, Tensor]:
        """Extra fixed-shape keys to expose to the policy observation."""
        return {}

    def obs_space_extra(self, env: Any) -> Dict[str, Any]:
        """Gym obs-space entries this component contributes — the space-side twin of
        ``obs_extra``: declare the same keys under the same gate, so the declared
        observation space matches what ``observation()`` actually emits. Default: none."""
        return {}

    # ---- per-transition StepOutput trace fields (emitted each step, not carried) ----
    def declare_trace_fields(self, env: Any) -> Tuple[str, ...]:
        """Names of per-transition fields this component adds to the composed StepOutput
        (e.g. a kge component adds a per-query outcome field). Fixed at build; the values
        are produced each step by ``step_trace``. Consumers read them by name (the rollout
        buffer / the eval scorer), defaulting gracefully when a field is absent."""
        return ()

    def step_trace(self, env: Any, state, new_state, ctx: TransitionCtx
                   ) -> Dict[str, Tensor]:
        """Per-step values for this component's ``declare_trace_fields`` → {name: [B,...]}. ``state``
        is the pre-step state (the completed episode's metadata, pre auto-reset); ``ctx`` is the
        transition as the core saw it (actions, done/truncation/success edges)."""
        return {}
