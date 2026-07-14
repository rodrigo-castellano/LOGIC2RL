"""BaseEngine — the vectorized single-step SLD derivation engine over a fixed program (any arity).

The shared substrate; the concrete engines (``sld.SLD`` soft / ``join.Join`` real-fact) extend it
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
             neural filler; :class:`Join` overrides it to ground with a real KB fact) —
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

from logic2rl.unification.base.kb import KB, fact_contains
from logic2rl.unification.base.resolution import (resolve_facts, resolve_rules,
                                                  standardize_vars)
from logic2rl.unification.base.soft import resolve_soft_facts as _resolve_soft_facts

# ==========================================================================
# Dense pack / prune / compact (BIT-EXACT, fingerprint-locked)
#
# _pack_children (children axis) and _compact_atoms (atom axis) share the same
# scatter-to-cumsum-rank idiom but are kept as separate fused implementations ON PURPOSE:
# factoring them into one generic primitive was tried and, while bit-exact, its extra
# cumsum/reshape work regressed the compiled train step's warm warmup past the speed gate
# (kge/tests test_qkge[geom]: 0.39s → 0.69s, gate 0.59s).
# ==========================================================================

def _pack_children(
    fact_goals: Tensor,     # [B, K_f, L, W]
    fact_success: Tensor,   # [B, K_f]
    rule_goals: Tensor,     # [B, K_r, L, W]
    rule_success: Tensor,   # [B, K_r]
    sub_rule_idx: Tensor,   # [B, K_r]
    G: int,
    pad: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compact the successful children into ``G`` output slots, facts first then rules →
    ``(goals [B, G, L, W], counts [B], rule_idx [B, G])``.

    Each child scatters straight to its cumsum rank (facts fill ``[0, n_valid_f)``, rules
    after); invalid and overflow children target one discarded trash slot ``G``, so the first
    G children win. One scatter over the concatenated sources per output buffer (the two
    target regions are disjoint — half the kernel launches). Fact slots carry rule id -1,
    rule slots their top-level rule id, empty slots 0."""
    B, K_f = fact_success.shape
    L, W = rule_goals.shape[2], rule_goals.shape[3]
    dev = rule_goals.device
    trash = torch.tensor(G, dtype=torch.long, device=dev)

    cs_f = fact_success.long().cumsum(dim=1)                       # [B, K_f]
    n_valid_f = cs_f[:, -1:]                                       # [B, 1]
    cs_r = rule_success.long().cumsum(dim=1) + n_valid_f
    target_f = torch.where(fact_success, cs_f - 1, trash).clamp_(min=0, max=G)
    target_r = torch.where(rule_success, cs_r - 1, trash).clamp_(min=0, max=G)
    counts = cs_r[:, -1].clamp(max=G)

    out_goals = torch.full((B, G + 1, L, W), pad, dtype=torch.long, device=dev)
    out_rid = torch.zeros(B, G + 1, dtype=torch.long, device=dev)

    tgt = torch.cat([target_r, target_f], dim=1)                   # [B, K_r + K_f]
    out_goals.scatter_(1, tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, W),
                       torch.cat([rule_goals, fact_goals], dim=1))
    f_neg1 = torch.full((B, K_f), -1, dtype=torch.long, device=dev)
    out_rid.scatter_(1, tgt, torch.cat([sub_rule_idx, f_neg1], dim=1))

    slot_valid = torch.arange(G, device=dev).unsqueeze(0) < counts.unsqueeze(1)
    rule_idx = torch.where(slot_valid, out_rid[:, :G], torch.zeros((), dtype=torch.long, device=dev))
    return out_goals[:, :G], counts, rule_idx


def _prune_ground_facts(
    candidates: Tensor,         # [B, K, M, W]
    fact_hashes: Tensor,        # [F]
    pack_base: int,
    constant_no: int,
    pad: int,
    excluded: Optional[Tensor] = None,   # [B, 1, W]
) -> Tensor:
    """Keep-mask ``[B, K, M]`` dropping ground atoms that are known KB facts (a proven
    subgoal resolves deterministically — remove it from the goal). The ``excluded`` root
    query atom is NOT prunable (cycle prevention)."""
    B, K, M, W = candidates.shape
    preds = candidates[:, :, :, 0]
    valid_atom = preds != pad
    ground = (candidates[:, :, :, 1:] <= constant_no).all(dim=-1) & valid_atom
    is_fact = fact_contains(candidates.reshape(-1, W), fact_hashes, pack_base).reshape(B, K, M)
    is_fact = is_fact & ground
    if excluded is not None:
        excl = excluded[:, 0, :].unsqueeze(1).unsqueeze(1)                   # [B, 1, 1, W]
        is_fact = is_fact & ~((candidates == excl).all(dim=-1) & ground)
    return valid_atom & ~is_fact


def _compact_atoms(states: Tensor, pad: int, valid: Tensor) -> Tensor:
    """Left-align the ``valid`` atoms within each ``[..., M, W]`` slice.

    Scatter-based: kept atoms write to their cumsum rank (unique by construction); dropped
    atoms all write to one discarded trash slot ``M``."""
    if states.numel() == 0:
        return states
    *leading, M, W = states.shape
    flat = states.reshape(-1, M, W)
    keep = valid.reshape(-1, M)
    pos = torch.cumsum(keep, dim=1, dtype=torch.long) - 1
    tgt = torch.where(keep, pos, M)
    out = flat.new_full((flat.shape[0], M + 1, W), pad)
    out.scatter_(1, tgt.unsqueeze(-1).expand(-1, -1, W), flat)
    return out[:, :M].reshape(*leading, M, W)


# ==========================================================================
# The engine
# ==========================================================================

class BaseEngine(nn.Module):
    """Single-step SLD derivation engine over a fixed program (see the module docstring).

    The shared substrate for the concrete engines: :class:`~logic2rl.unification.sld.SLD`
    (soft open-var resolution) and :class:`~logic2rl.unification.join.Join` (real-fact join) —
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
        soft: bool = False,                   # soft open-var grounding on (SLD hook / Join derive)
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

        # Soft open-var grounding: whether it is on (``_soft``) and the scorer ``(states, counts)
        # -> v* [B, G]`` — an nn.Module (a KGE SoftUnifier), so the compiled step traces it.
        # Attached by the app post-build (the scorer needs the trained model). When ``soft`` is on
        # the scorer MUST be attached: SLD grounds with it at the ``resolve_soft_facts`` seam, Join
        # from its ``derive`` (filling the residual the real-fact join left open). Pure sld / pure
        # join leave ``_soft`` off (no scorer needed).
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
    # derive — the SLD successor function
    # ==================================================================

    @torch.no_grad()
    def derive(
        self,
        current_states: Tensor,        # [B, A, W] goal states (first atom = selected goal)
        next_var_indices: Tensor,      # [B] fresh-variable allocator
        excluded_queries: Optional[Tensor] = None,   # [B, 1, W] root query (cycle prevention)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """One backward SLD step → ``(derived [B, G, L, W], counts [B], next_var [B],
        derived_rule_idx [B, G])``.

        Vocab-free: returns raw successors — a fully-resolved proof is a zero-atom valid
        slot, a dead end is ``counts == 0`` — and the *env* marks proofs True / dead ends
        False. ``derived_rule_idx`` is the top-level rule id per slot (-1 for fact-derived
        children, 0 on padding slots)."""
        kb = self.kb
        pad = kb.padding_idx
        B, A_in, W = current_states.shape
        L = self.max_atoms
        dev = current_states.device

        # ── SELECT: seed the goal tape; leftmost atom is the goal, the rest is the tail ──
        goal = torch.full((B, L, W), pad, dtype=torch.long, device=dev)
        goal[:, :A_in, :] = current_states
        queries = goal[:, 0, :]                                              # [B, W]
        active = queries[:, 0] != pad
        queries = queries * active.unsqueeze(-1).to(queries.dtype)           # zero padded queries
        remaining = goal.clone()
        remaining[:, 0, :] = pad

        # ── RESOLVE: facts ∥ rules (dense) ──
        fact_goals, fact_success = resolve_facts(
            queries, remaining, kb.fact_index, kb.constant_no, pad, kb.K_f,
            active, excluded=excluded_queries)
        rule_goals, rule_success, sub_rule_idx = resolve_rules(
            queries, remaining, kb.rule_index, kb.constant_no, pad, kb.K_r,
            active, next_var_indices)

        # ── PACK: compact the K_f + K_r children into G slots ──
        derived, counts, rule_idx = _pack_children(
            fact_goals, fact_success, rule_goals, rule_success, sub_rule_idx, self.G, pad)

        # ── PRUNE + COMPACT (fused): drop subgoals that are known facts, left-align ──
        keep = _prune_ground_facts(
            derived, kb.fact_index.fact_hashes, kb.fact_index.pack_base,
            kb.constant_no, pad, excluded=excluded_queries)
        derived = _compact_atoms(derived, pad, valid=keep)

        # ── STANDARDIZE: rename output vars past the live range ──
        derived, new_next_var = standardize_vars(
            derived, next_var_indices, kb.constant_no, self.runtime_var_end_index, pad,
            input_states=current_states, extra_new_vars=self._body_width + 2,
            enforce_runtime_range=self._enforce)

        # ── zero invalid slots; mask the per-slot rule id to the valid ones ──
        slot_valid = torch.arange(self.G, device=dev).unsqueeze(0) < counts.unsqueeze(1)
        derived = torch.where(slot_valid.unsqueeze(-1).unsqueeze(-1), derived, pad)
        rule_idx = torch.where(slot_valid, rule_idx, torch.zeros_like(rule_idx))
        return derived, counts, new_next_var, rule_idx

    def resolve_soft_facts(self, states: Tensor, counts: Tensor) -> Tensor:
        """Post-derive SOFT-fact resolution — commit each remaining free variable with its neural
        argmax filler (the ``base.soft.resolve_soft_facts`` primitive), when ``soft`` is on. Invoked
        ONCE per final candidate set by the env's candidate generation (the end of
        ``UnificationLogic``'s pipeline, after the unary refine). For SLD these are ALL the free
        vars; for :class:`Join` (soft variant) only the RESIDUAL its real-fact resolution left open
        (pure join / pure sld leave it a no-op — no free vars remain).

        The soft [S,E] GEMM lives at this seam — NOT inside ``derive`` — on purpose: the unary
        auto-advance re-derives up to ``max_unary_iterations`` times per env step, so running the
        GEMM in ``derive`` pays it on every intermediate candidate set (~2.4x slower — measured),
        whereas here it runs exactly once. Join's real-fact resolution is cheap (a fact lookup, no
        GEMM), so THAT lives in ``derive`` as a proper resolution method — see :class:`Join`."""
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


__all__ = ["BaseEngine", "_pack_children", "_prune_ground_facts", "_compact_atoms"]
