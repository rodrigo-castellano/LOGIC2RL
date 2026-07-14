"""BaseEngine — the vectorized single-step SLD derivation engine over a fixed program (any arity).

The shared substrate; the concrete engines (``sld.SLD`` soft / ``enumerate.Enumerate`` real-fact) extend it
and differ only in ``resolve_soft_facts``.


The Prolog resolution core minus the control strategy. A classical Prolog engine couples
resolution to a fixed search rule (leftmost selection + chronological backtracking); here the
search is owned by the caller — the RL env presents the successor states and a learned policy
picks the branch. The engine therefore exposes the two Prolog verbs that remain:

  __init__   ≈ consult/1 — index the program (facts + rules), fix the runtime-variable
             id-space and the per-step branching budgets. The program is immutable
             (no assert/retract): budgets and shapes are compile-time constants.
  derive     the SLD successor function — ONE backward resolution step from each goal state:
             select the leftmost atom → resolve against facts ∥ rule heads → pack the
             children densely → prune subgoals that are known facts → standardize variables.
             Open vars are committed separately by ``resolve_soft_facts`` (soft unification when a
             ``soft_scorer`` is attached — each free variable is unified with its most likely
             neural filler; :class:`Enumerate` overrides it to ground with a real KB fact) —
             invoked ONCE per final candidate set by the env's candidate generation, at the end
             of ``UnificationLogic``'s candidate pipeline, so delivered candidates leave ground.
  prove      ≈ solve/1 — reference exhaustive breadth-first search over ``derive``
             (tests/debugging only; the RL path never calls it).

``derive`` is vocab-free: a fully-resolved child is a zero-atom valid slot, a dead end is
``counts == 0`` — the env marks proofs True / dead ends False. Atom width W is read off the
program tensors (W = max_arity + 1); at W=3 every op is BIT-EXACT with the historic binary
engine (fingerprint-locked, gated by the SB3 parity suite).

Usage:
    from logic2rl.unification import SLD

    engine = SLD(
        facts_idx=mat.facts_idx, rules_idx=mat.rules_idx,   # rules_idx = combined head+body
        padding_idx=im.padding_idx, constant_no=im.constant_no,
        n_runtime_vars=cfg.max_total_vars, device=im.device,
    )
    derived, counts, next_var, rule_idx = engine.derive(states, next_var, excluded)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from logic2rl.unification.base.kb import KB
from logic2rl.unification.base.soft import resolve_soft_facts as _resolve_soft_facts


# ==========================================================================
# The engine
# ==========================================================================

class BaseEngine(nn.Module):
    """Single-step SLD derivation engine over a fixed program (see the module docstring).

    The shared substrate for the concrete engines: :class:`~logic2rl.unification.sld.SLD`
    (soft open-var resolution) and :class:`~logic2rl.unification.enumerate.Enumerate` (real-fact enumerate) —
    siblings that both extend this and differ ONLY in :meth:`resolve_soft_facts`. Not
    instantiated directly.

    Sizing attributes read by the builder/env (the ``Grounder`` contract): ``max_children``
    (branching budget → padding_states), ``total_vocab_size`` (token-id ceiling + hash base),
    ``n_vars`` (runtime-var embedder table), ``num_rules``. The sizing keyword-only knobs
    (``derived_cap``, ``full_fact_slots``, ``n_vars``) have production defaults; the SB3
    parity suite overrides them to mirror the frozen reference's enumeration widths."""

    def __init__(
        self,
        facts_idx: Tensor,             # [F, W] ground atoms
        rules_idx: Tensor,             # combined head+body per rule: [R, 1+max_body, W], head at slot 0
        *,
        padding_idx: int,
        constant_no: int,
        n_runtime_vars: int,
        device: torch.device,
        predicate_no: Optional[int] = None,
        padding_atoms: Optional[int] = None,
        max_arity: int = 2,
        max_steps: int = 10,
        enforce_runtime_var_range: bool = False,
        max_children: Optional[int] = None,   # branching cap (default 550)
        derived_cap: Optional[int] = None,    # G: output slots per derive (default 256)
        full_fact_slots: bool = False,        # facts get the full budget (SB3 enumeration width)
        n_vars: Optional[int] = None,         # runtime-var table size (default: derived formula)
        soft: bool = False,                   # soft open-var grounding on (SLD hook / Enumerate derive)
    ):
        super().__init__()
        assert facts_idx.shape[-1] == rules_idx.shape[-1], (
            f"facts W={facts_idx.shape[-1]} != rules W={rules_idx.shape[-1]}")

        # max body width = combined rule width minus the head slot (bodies are padded to it).
        body_width = int(rules_idx.shape[1]) - 1 if rules_idx.numel() > 0 else 1
        if padding_atoms is None:
            padding_atoms = body_width + 1
        self._M_rl = body_width + padding_atoms + 1            # RL state width

        # ── Runtime var id-space (engine-owned) ──
        # Layout: constants [1..constant_no], runtime vars [start..end], padding 0. The pool
        # size ``n_runtime_vars`` (= config.max_total_vars) fixes the standardizer ceiling +
        # the packing base — both knowable from ``constant_no`` alone.
        self.n_runtime_vars = int(n_runtime_vars)
        self.runtime_var_start_index = int(constant_no) + 1
        self.runtime_var_end_index = self.runtime_var_start_index + self.n_runtime_vars - 1
        self.total_vocab_size = int(constant_no) + self.n_runtime_vars + 1
        self._max_steps = int(max_steps)
        self._body_width = body_width
        self._enforce = bool(enforce_runtime_var_range)

        self.kb = KB(
            facts_idx, rules_idx,
            constant_no=constant_no, predicate_no=predicate_no,
            padding_idx=padding_idx, device=device,
            pack_base=self.total_vocab_size,
            max_children=550 if max_children is None else int(max_children),
            full_fact_slots=full_fact_slots)
        self.num_rules = self.kb.rule_index.num_rules
        self.max_children = self.kb.max_children

        # Soft open-var grounding: on/off flag (``_soft``) + the app-attached scorer (nn.Module). When on,
        # the scorer MUST be attached (SLD grounds at the hook, Enumerate fills its residual); else off.
        self._soft = bool(soft)
        self.soft_scorer = None

        # Goal-tape width L and the pack output cap G.
        self.max_atoms = max(self._M_rl, self.kb.rule_index.M)
        self.G = 256 if derived_cap is None else int(derived_cap)

        # ── Public attributes — KB metadata for env / policy consumers ──
        self.constant_no = self.kb.constant_no
        self.padding_idx = self.kb.padding_idx
        self.pack_base = self.kb.fact_index.pack_base
        self.M = self._M_rl
        self.device = self.kb.device_
        # Embedder runtime-var table size: each step can introduce up to
        # ``max_children * padding_atoms * max_arity`` fresh vars (the offset standardizer
        # only renumbers upward, never compacts), so the max var id climbs by the full
        # derived-state width per step. Size the table to ``max_steps`` of that + a base.
        self.n_vars = (int(n_vars) if n_vars is not None
                       else self._max_steps * (int(self.max_children) * int(padding_atoms) * int(max_arity) + 1) + 64)

    # ==================================================================
    # derive — the successor function (COMPULSORY; each engine implements it)
    # ==================================================================

    @torch.no_grad()
    def derive(self, current_states: Tensor, next_var_indices: Tensor,
               excluded_queries: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """One backward resolution step (the successor function) → ``(derived [B, G, L, W],
        counts [B], next_var [B], derived_rule_idx [B, G])``. COMPULSORY — each concrete engine
        implements it: :class:`~logic2rl.unification.sld.SLD` resolves the leftmost atom against
        facts ∥ rules (free vars stay open, committed at the ``resolve_soft_facts`` hook);
        :class:`~logic2rl.unification.enumerate.Enumerate` enumerates the real-fact groundings of each rule
        body. ``prove`` and the RL env drive it polymorphically.

        Semantic contract (what the env reads off the output): a successor with ZERO non-padding
        atoms marks a completed proof; ``counts[b] == 0`` means a dead end; ``excluded [B,1,W]`` is
        the episode's root query atom (don't re-derive it — cycle prevention); ``next_var`` is the
        fresh-runtime-var allocator, advanced and returned; atoms are FLAT ``(pred, arg1, …)``."""
        raise NotImplementedError("derive: use a concrete engine (SLD / Enumerate), not BaseEngine.")

    def resolve_soft_facts(self, states: Tensor, counts: Tensor) -> Tensor:
        """Post-derive SOFT-fact resolution — commit each remaining free variable with its neural
        argmax filler (the ``base.soft.resolve_soft_facts`` primitive), when ``soft`` is on. Invoked
        ONCE per final candidate set by the env's candidate generation (the end of
        ``UnificationLogic``'s pipeline, after the unary refine). For SLD these are ALL the free
        vars; for :class:`Enumerate` (soft variant) only the RESIDUAL its real-fact resolution left open
        (pure enumerate / pure sld leave it a no-op — no free vars remain).

        The soft [S,E] GEMM lives at this seam — NOT inside ``derive`` — on purpose: the unary
        auto-advance re-derives up to ``max_unary_iterations`` times per env step, so running the
        GEMM in ``derive`` pays it on every intermediate candidate set (~2.4x slower — measured),
        whereas here it runs exactly once. Enumerate's real-fact resolution is cheap (a fact lookup, no
        GEMM), so THAT lives in ``derive`` as a proper resolution method — see :class:`Enumerate`."""
        if not self._soft:
            return states
        return _resolve_soft_facts(states, counts, self.soft_scorer,
                                   self.kb.constant_no, self.kb.padding_idx)

    # ==================================================================
    # prove — reference exhaustive search (tests / debugging)
    # ==================================================================

    @torch.no_grad()
    def prove(
        self,
        queries: Tensor,               # [B, W] atoms or [B, A, W] goal states
        max_depth: Optional[int] = None,
        excluded_queries: Optional[Tensor] = None,   # [B, 1, W]
        max_frontier: int = 4096,
    ) -> Tensor:
        """Exhaustive breadth-first proof search over ``derive`` → ``proven [B]`` bool.

        The Prolog ``solve`` verb as a reference oracle: expands EVERY successor instead of
        following a policy. Eager Python loop, no dedup of alpha-variant states — worst case
        exponential in depth, bounded by ``max_frontier`` live states (excess is truncated
        breadth-first) and ``max_depth`` (default: the engine's ``max_steps``). For program
        debugging and tests; the RL path drives ``derive`` directly."""
        states = queries.unsqueeze(1) if queries.dim() == 2 else queries
        B = states.shape[0]
        dev = states.device
        pad = self.padding_idx

        proven = torch.zeros(B, dtype=torch.bool, device=dev)
        origin = torch.arange(B, device=dev)
        frontier = states
        next_var = torch.full((B,), self.runtime_var_start_index, dtype=torch.long, device=dev)
        excl = excluded_queries

        for _ in range(max_depth if max_depth is not None else self._max_steps):
            derived, counts, next_var, _ = self.derive(frontier, next_var, excl)
            atom_counts = (derived[:, :, :, 0] != pad).sum(dim=2)            # [N, G]
            slot_valid = (torch.arange(self.G, device=dev).unsqueeze(0)
                          < counts.unsqueeze(1))                             # [N, G]
            proven[origin[(slot_valid & (atom_counts == 0)).any(dim=1)]] = True
            if bool(proven.all()):
                break
            # Expand every open successor of a still-unproven origin.
            open_ = slot_valid & (atom_counts > 0) & ~proven[origin].unsqueeze(1)
            n_idx, s_idx = open_.nonzero(as_tuple=True)
            if n_idx.numel() == 0:
                break
            n_idx, s_idx = n_idx[:max_frontier], s_idx[:max_frontier]
            frontier = derived[n_idx, s_idx]                                 # [N', L, W]
            next_var = next_var[n_idx]
            origin = origin[n_idx]
            excl = excl[n_idx] if excl is not None else None
        return proven


__all__ = ["BaseEngine"]
