"""SLD engine — SLD backward resolution (facts ∥ rules) with open-var fill at the seam.

:class:`SLD` extends :class:`~logic2rl.unification.base.engine.BaseEngine` with the SLD-specific
resolution: one step resolves the leftmost goal atom against facts AND rule heads *in parallel*
(``resolve_facts`` ∥ ``resolve_rules``); free variables stay open and are committed by the
``replace_candidates`` seam — ``soft_fill_vars`` (joint-scorer argmax over all entities) or
:meth:`SLD.fact_fill_vars` (argmax over REAL-FACT fillers, no-fact states discarded to FALSE),
per ``var_fill``. With ``var_fill='none'`` vars persist (the KGE app rejects pure SLD). Sibling
of :class:`~logic2rl.unification.enumerate.Enumerate` (the real-fact enumerate engine); both
extend ``BaseEngine``, neither inherits the other. ``resolve_facts`` lives here because only SLD's
step resolves the leftmost atom against facts — Enumerate's real-fact resolution is the body enumerate.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from logic2rl.unification.base.engine import BaseEngine
from logic2rl.unification.base.resolution import (_compact_atoms, _pack_children,
                                                  _prune_ground_facts, apply_substitutions,
                                                  resolve_rules, standardize_vars, unify_atoms)
from logic2rl.unification.base.soft import fill_vars


def resolve_facts(
    queries: Tensor,               # [B, W]
    remaining: Tensor,             # [B, L, W]
    fact_index,
    constant_no: int,
    pad: int,
    K_f: int,
    active: Tensor,                # [B]
    excluded: Optional[Tensor] = None,   # [B, 1, W]
) -> Tuple[Tensor, Tensor]:
    """Fact resolution: targeted lookup → unify → substitute the remaining goals.

    Returns ``(fact_goals [B, K_f, L, W], success [B, K_f])``. ``excluded`` masks out the
    episode's root query atom (cycle prevention). SLD-exclusive — only SLD's ``derive`` resolves
    the leftmost atom against facts in parallel with rules."""
    B, W = queries.shape
    L = remaining.shape[1]
    facts_idx = fact_index.facts_idx
    fact_item_idx, fact_valid = fact_index.targeted_lookup(queries, K_f)     # [B, K_f]
    F = facts_idx.shape[0]
    safe_idx = fact_item_idx.clamp(0, max(F - 1, 0))
    fact_atoms = facts_idx[safe_idx.view(-1)].view(B, K_f, W)
    q_exp = queries.unsqueeze(1).expand(-1, K_f, -1)
    ok, subs = unify_atoms(q_exp, fact_atoms, constant_no=constant_no, pad=pad)
    success = ok & fact_valid & active.unsqueeze(1)
    if excluded is not None:
        excl = excluded[:, 0, :].unsqueeze(1)                                # [B, 1, W]
        success = success & ~(fact_atoms == excl).all(dim=-1)

    subs_flat = subs.reshape(B * K_f, W - 1, 2)
    rem_exp = remaining.unsqueeze(1).expand(-1, K_f, -1, -1).reshape(B * K_f, L, W)
    fact_goals = apply_substitutions(rem_exp, subs_flat, pad).view(B, K_f, L, W)
    pad_t = torch.tensor(pad, dtype=torch.long, device=queries.device)
    fact_goals = torch.where(success.view(B, K_f, 1, 1), fact_goals, pad_t)
    return fact_goals, success


class SLD(BaseEngine):
    """SLD backward resolution — see the module docstring."""

    def fact_fill_vars(self, states: Tensor, counts: Tensor) -> Tensor:
        """FACT fill — commit each state's free variable to the joint scorer's best REAL-FACT
        assignment (``joint_scorer.topk_fact_assignments`` k=1); a state with NO real-fact
        filler is discarded to a FALSE terminal (``no_fact`` + the app-attached ``false_pred``)
        instead of committing a garbage entity. The fact-unification analogue of soft fill."""
        assert self.joint_scorer is not None, "fact_fill_vars requires an attached joint_scorer"
        assert self.false_pred is not None, "fact_fill_vars requires false_pred (the discard target)"
        vstar, _, no_fact = self.joint_scorer.topk_fact_assignments(states, counts, k=1)
        return fill_vars(states, vstar[..., 0], self.kb.constant_no, self.kb.padding_idx,
                         no_fact=no_fact, false_pred=self.false_pred)

    @torch.no_grad()
    def derive(self, current_states: Tensor, next_var_indices: Tensor,
               excluded_queries: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """One backward SLD step → ``(derived [B, G, L, W], counts [B], next_var [B],
        derived_rule_idx [B, G])``. Resolve the leftmost goal atom against facts ∥ rules, pack the
        children densely, prune subgoals that are known facts, standardize. Vocab-free: a
        fully-resolved proof is a zero-atom valid slot, a dead end is ``counts == 0``; free vars stay
        open (committed later at the ``replace_candidates`` seam)."""
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


__all__ = ["SLD", "resolve_facts"]
