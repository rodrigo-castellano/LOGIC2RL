"""Enumerate engine — SLD derivation with **enumerate** open-var resolution.

:class:`Enumerate` is a sibling of :class:`~logic2rl.unification.sld.SLD` (both extend
:class:`~logic2rl.unification.base.engine.BaseEngine`, neither inherits the other). It reuses the
whole shared machinery — consult / pack / prune / ``prove`` / the ``replace_candidates`` seam —
and overrides ``derive`` to ground each free variable with a REAL KB fact (the
:class:`~logic2rl.unification.base.joint.FactJoint`) *inside* the resolution, instead of a neural
filler. In the soft variant (``var_fill='soft'``) it leaves the residual (no-real-fact) vars open
for ``soft_fill_vars`` to fill once at the seam. Configure the enumerate width ``K``, the
compaction width ``S`` (= q_kge_joint_s) and the exactness ``stats`` buffer once via
:meth:`set_enumerate`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from logic2rl.unification.base.engine import BaseEngine
from logic2rl.unification.base.joint import FactJoint
from logic2rl.unification.base.resolution import (_compact_atoms, _pack_children,
                                                  _prune_ground_facts, resolve_rules,
                                                  standardize_vars)


class Enumerate(BaseEngine):
    """SLD derivation with **enumerate** as a RESOLUTION METHOD — see the module docstring.

    Sibling of :class:`~logic2rl.unification.sld.SLD`; inherits ``BaseEngine`` but overrides
    ``derive`` with the pbc-style pipeline: ``resolve_rules → FactJoint (all real-fact groundings
    of each body, branching) → pack into G → prune → standardize`` (rule→join, NO parallel
    ``resolve_facts`` — the join *is* the fact resolution). Pure enumerate emits only grounded
    children; the ``soft`` variant leaves a body's residual (no-real) var open for
    ``soft_fill_vars`` at the seam (top-1). Configure ``K``/``S``/``stats`` via :meth:`set_enumerate`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._var_fill == "fact":
            raise ValueError("Enumerate has no fact_fill_vars — its derive IS the fact "
                             "resolution; use var_fill 'none' (pure) or 'soft' (residual fill).")
        self.fact_joint: Optional[FactJoint] = None
        self._proven_body: Optional[Tensor] = None   # opt-in per-env [B, L, W] proven-body scratch

    def set_enumerate(self, k: int, s: int, stats: Tensor,
                      proven_body: Optional[Tensor] = None) -> None:
        """Wire the enumerate width ``K``, compaction width ``S``, and the running-max
        exactness ``stats`` (int64[2]) — called once by the app post-build; builds the
        :class:`FactJoint`. ``proven_body`` (opt-in static [B, L, W] buffer) receives each
        env's first fully-ground body per ``derive`` (for a body-path score); ``None`` disables it."""
        self.fact_joint = FactJoint(self.kb.fact_index, self.kb.constant_no, self.kb.padding_idx,
                                    K=int(k), S=int(s), stats=stats)
        self._proven_body = proven_body

    def _stash_proven_body(self, derived: Tensor, keep: Tensor, counts: Tensor) -> None:
        """Stash each env's FIRST fully-ground body — every atom a real fact (all pruned by
        ``keep`` → the slot collapses to TRUE downstream) — into the opt-in ``_proven_body``
        buffer, pre-compaction, so a policy can score the proof path at the TRUE accept."""
        B, dev, pad = derived.shape[0], derived.device, self.padding_idx
        present = derived[..., 0] != pad                                     # [B, G, L]
        within = torch.arange(self.G, device=dev).unsqueeze(0) < counts.unsqueeze(1)
        proven = within & present.any(-1) & ~(keep & present).any(-1)        # [B, G] all-fact slot
        first = proven.to(torch.int32).argmax(1)                             # [B] first proven slot
        pb = derived[torch.arange(B, device=dev), first]                     # [B, L, W]
        self._proven_body.copy_(torch.where(proven.any(1).view(B, 1, 1), pb,
                                            torch.full_like(pb, pad)))

    @torch.no_grad()
    def derive(self, current_states: Tensor, next_var_indices: Tensor,
               excluded_queries: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """One backward step: rule resolution → FactJoint (all real-fact groundings of each body)
        → pack → prune → standardize. No parallel fact resolution (the join is the fact resolution)."""
        if self.fact_joint is None:
            raise RuntimeError("Enumerate engine used before set_enumerate(K, S, stats).")
        kb = self.kb
        pad = kb.padding_idx
        B, A_in, W = current_states.shape
        L = self.max_atoms
        dev = current_states.device

        # ── SELECT: goal tape; leftmost atom is the goal, the rest is the tail ──
        goal = torch.full((B, L, W), pad, dtype=torch.long, device=dev)
        goal[:, :A_in, :] = current_states
        queries = goal[:, 0, :]
        active = queries[:, 0] != pad
        queries = queries * active.unsqueeze(-1).to(queries.dtype)
        remaining = goal.clone()
        remaining[:, 0, :] = pad

        # ── RESOLVE: rules only → each rule's substituted body (with free vars) ──
        rule_goals, rule_success, sub_rule_idx = resolve_rules(
            queries, remaining, kb.rule_index, kb.constant_no, pad, kb.K_r, active, next_var_indices)

        # ── JOIN: enumerate all real-fact groundings of each body (branch) ──
        Y_r = max(1, self.G // max(int(kb.K_r), 1))     # per-rule grounding cap; K_r*Y_r ≈ G
        ground_goals, ground_success, ground_rid = self.fact_joint.all_groundings(
            rule_goals, rule_success, sub_rule_idx, Y_r, leave_open=self._var_fill == "soft")

        # ── PACK the groundings into G (no fact children; the join replaced them) ──
        empty_f = torch.full((B, 1, L, W), pad, dtype=torch.long, device=dev)
        empty_fs = torch.zeros(B, 1, dtype=torch.bool, device=dev)
        derived, counts, rule_idx = _pack_children(
            empty_f, empty_fs, ground_goals, ground_success, ground_rid, self.G, pad)

        # ── PRUNE + COMPACT: discharge subgoals that are known facts, left-align ──
        keep = _prune_ground_facts(
            derived, kb.fact_index.fact_hashes, kb.fact_index.pack_base,
            kb.constant_no, pad, excluded=excluded_queries)
        if self._proven_body is not None:
            self._stash_proven_body(derived, keep, counts)
        derived = _compact_atoms(derived, pad, valid=keep)

        # ── STANDARDIZE: rename any remaining (residual / tail) vars ──
        derived, new_next_var = standardize_vars(
            derived, next_var_indices, kb.constant_no, self.runtime_var_end_index, pad,
            enforce_runtime_range=self._enforce)

        slot_valid = torch.arange(self.G, device=dev).unsqueeze(0) < counts.unsqueeze(1)
        derived = torch.where(slot_valid.unsqueeze(-1).unsqueeze(-1), derived, pad)
        rule_idx = torch.where(slot_valid, rule_idx, torch.zeros_like(rule_idx))
        return derived, counts, new_next_var, rule_idx


__all__ = ["Enumerate"]
