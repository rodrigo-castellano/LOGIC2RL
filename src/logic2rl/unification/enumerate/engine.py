"""Enumerate engine — SLD derivation with **enumerate** open-var resolution.

:class:`Enumerate` is a sibling of :class:`~logic2rl.unification.sld.SLD` (both extend
:class:`~logic2rl.unification.base.engine.BaseEngine`, neither inherits the other). It reuses the
whole shared machinery — consult / pack / prune / ``prove`` / the ``resolve_soft_facts`` hook — and
overrides ``derive`` to ground each free variable with a REAL KB fact (``enumerate_groundings``)
*inside* the resolution, instead of a neural filler. In the soft variant it leaves the residual
(no-real-fact) vars open for the shared ``resolve_soft_facts`` hook to fill once. Configure the
enumerate width ``K``, the compaction width ``S`` (= q_kge_joint_s) and the exactness ``stats``
buffer once via :meth:`set_enumerate`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from logic2rl.unification.base.engine import BaseEngine
from logic2rl.unification.base.kb import is_const, is_var
from logic2rl.unification.base.resolution import (_compact_atoms, _pack_children,
                                                  _prune_ground_facts, resolve_rules,
                                                  standardize_vars)


@torch.no_grad()
def enumerate_groundings(rule_goals: Tensor, rule_success: Tensor, sub_rule_idx: Tensor, fact_index,
                   constant_no: int, pad: int, K: int, Y_r: int, S: int, leave_open: bool = False,
                   stats: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """Enumerate ALL real-fact groundings of each rule body — the branching enumerate.

    ``rule_goals [B, K_r, L, 3]`` are the substituted rule bodies (from ``resolve_rules``); each has
    ≤1 shared free var over ≤2 soft atoms. For every body we enumerate up to ``Y_r`` bindings of that
    var that make the body a real KB fact (the semijoin: enumerate the more-selective atom's ≤K
    fillers, membership-filter the other), and emit ONE grounded body per binding. Returns
    ``(ground_goals [B, K_r*Y_r, L, 3], ground_success [B, K_r*Y_r], ground_rid [B, K_r*Y_r])``.

    Memory-lean: the ≤``S`` bodies with a free var are compacted first (``topk``), so the ``[·, K]``
    enumerate is ``O(S·K)`` not ``O(B·K_r·K)`` — bounded on yago-scale KBs. A rule body already ground
    (no free var) passes through as one child. With ``leave_open`` (soft variant) a body with NO real
    binding emits ONE child with its var left open (for the hook); without it that body dies. Fixed
    shapes (static ``S``, ``K``, ``Y_r``) → CUDA-graph safe. ≤1-var / ≤2-atom case; general V-var /
    M-atom enumerate is the next extension."""
    B, K_r, L, W = rule_goals.shape
    assert W == 3, "enumerate_groundings: binary predicates only (W=3)"
    N = B * K_r
    dev = rule_goals.device
    bodies = rule_goals.reshape(N, L, W)
    succ = rule_success.reshape(N)
    preds, a1, a2 = bodies[:, :, 0], bodies[:, :, 1], bodies[:, :, 2]          # [N, L]
    pt = is_const(a1, constant_no) & is_var(a2, constant_no, pad)              # p(e, V)
    ph = is_var(a1, constant_no, pad) & is_const(a2, constant_no)              # p(V, e)
    soft = pt | ph
    n_soft = soft.long().sum(1)
    active = succ & (n_soft >= 1)                                             # bodies to enumerate
    ground_pass = succ & (n_soft == 0)                                        # already-ground bodies

    # ── pack each body's ≤2 soft atoms + the shared var id [N] ──
    posn = soft.long().cumsum(1) - 1
    tk0, tk1 = (soft & (posn == 0)).long(), (soft & (posn == 1)).long()
    sp, sbe, shf = preds.long(), torch.where(pt, a1, a2).long(), ph.long()
    var_arg = torch.where(pt, a2, a1).long()                                  # free-var arg per soft atom
    p0, e0, h0 = (sp * tk0).sum(1), (sbe * tk0).sum(1), (shf * tk0).sum(1).bool()
    p1, e1, h1 = (sp * tk1).sum(1), (sbe * tk1).sum(1), (shf * tk1).sum(1).bool()
    varY = (var_arg * tk0).sum(1)                                             # [N] shared var id
    has2 = n_soft >= 2

    # ── COMPACT the ≤S active bodies (fixed shape) → the enumerate is O(S·K), not O(N·K) ──
    S_ = min(S, N)
    idx = torch.topk(active.float(), k=S_).indices                           # [S_]
    sa = active[idx]
    P0, E0, H0, P1, E1, H1 = p0[idx], e0[idx], h0[idx], p1[idx], e1[idx], h1[idx]
    HAS2, VARY, BODY = has2[idx], varY[idx], bodies[idx]                      # [S_], [S_], [S_, L, W]

    var = torch.full_like(E0, constant_no + 1)

    def _look(p: Tensor, e: Tensor, hf: Tensor) -> Tensor:
        return torch.stack([p, torch.where(hf, var, e), torch.where(hf, e, var)], dim=1)

    l0, l1 = _look(P0, E0, H0), _look(P1, E1, H1)
    c0, c1 = fact_index.targeted_count(l0), fact_index.targeted_count(l1)
    swap = HAS2 & (c1 < c0)
    lE, hE = torch.where(swap.unsqueeze(1), l1, l0), torch.where(swap, H1, H0)
    pM, eM, hM = torch.where(swap, P0, P1), torch.where(swap, E0, E1), torch.where(swap, H0, H1)
    cE = torch.where(swap, c1, c0)

    fidx, fvalid = fact_index.targeted_lookup(lE, K)                          # [S_, K]
    col = torch.where(hE, torch.ones_like(hE, dtype=torch.long),
                      torch.full_like(hE, 2, dtype=torch.long))
    fill = fact_index.facts_idx.reshape(-1).index_select(
        0, (fidx * W + col.unsqueeze(1)).reshape(-1)).reshape(S_, K)          # [S_, K] filler ids
    base = int(fact_index.pack_base)
    hcol = torch.where(hM.unsqueeze(1), fill, eM.unsqueeze(1))
    tcol = torch.where(hM.unsqueeze(1), eM.unsqueeze(1), fill)
    key = (pM.unsqueeze(1) * base + hcol) * base + tcol                       # [S_, K]
    fh = fact_index.fact_hashes
    F = fh.shape[0]
    ins = torch.searchsorted(fh, key.reshape(-1))
    mM = ((ins < F) & (fh[ins.clamp(max=F - 1)] == key.reshape(-1))).reshape(S_, K)
    okm = fvalid & (mM | ~HAS2.unsqueeze(1)) & sa.unsqueeze(1)                # [S_, K] valid bindings

    # ── take the first Y_r valid bindings (topk on the validity mask) ──
    Y_ = min(Y_r, K)
    sel = torch.topk(okm.float(), k=Y_, dim=1).indices                       # [S_, Y_]
    fillY, validY = fill.gather(1, sel), okm.gather(1, sel)                   # [S_, Y_]
    if Y_ < Y_r:
        fillY = torch.cat([fillY, fillY.new_zeros(S_, Y_r - Y_)], dim=1)
        validY = torch.cat([validY, validY.new_zeros(S_, Y_r - Y_)], dim=1)

    open0 = torch.zeros(S_, dtype=torch.bool, device=dev)
    if leave_open:
        open0 = sa & ~validY.any(1)                                          # no real binding → keep open
        validY = validY.clone()
        validY[:, 0] = validY[:, 0] | open0

    # ── substitute VARY → each binding into the compacted bodies ([S_, Y_r, L, 3]) ──
    g = BODY.unsqueeze(1).expand(S_, Y_r, L, W)
    is_varY = (g[..., 1:] == VARY.view(S_, 1, 1, 1))
    subst = torch.where(is_varY, fillY.view(S_, Y_r, 1, 1), g[..., 1:])
    if leave_open:
        slot0 = torch.arange(Y_r, device=dev).view(1, Y_r, 1, 1) == 0
        subst = torch.where(open0.view(S_, 1, 1, 1) & slot0 & is_varY, g[..., 1:], subst)
    grounded_s = torch.cat([g[..., :1], subst], dim=-1)                      # [S_, Y_r, L, 3]

    # ── scatter the groundings back to the N body slots ──
    grounded = torch.full((N, Y_r, L, W), pad, dtype=torch.long, device=dev)
    grounded.index_copy_(0, idx, grounded_s)
    gsucc = torch.zeros(N, Y_r, dtype=torch.bool, device=dev)
    gsucc.index_copy_(0, idx, validY & sa.unsqueeze(1))
    # already-ground rule bodies (no free var) pass through as one child (slot 0)
    grounded[:, 0, :, :] = torch.where(ground_pass.view(N, 1, 1), bodies, grounded[:, 0, :, :])
    gsucc = gsucc.clone()
    gsucc[:, 0] = gsucc[:, 0] | ground_pass

    if stats is not None:
        need = torch.where(sa & (cE > Y_r), cE, torch.zeros_like(cE))
        stats.copy_(torch.maximum(stats, torch.stack([active.sum(), need.amax()])))

    ground_goals = grounded.reshape(B, K_r * Y_r, L, W)
    ground_success = gsucc.reshape(B, K_r * Y_r)
    ground_rid = sub_rule_idx.reshape(N, 1).expand(N, Y_r).reshape(B, K_r * Y_r)
    return ground_goals, ground_success, ground_rid


class Enumerate(BaseEngine):
    """SLD derivation with **enumerate** as a RESOLUTION METHOD — see the module docstring.

    Sibling of :class:`~logic2rl.unification.sld.SLD`; inherits ``BaseEngine`` but overrides
    ``derive`` with the pbc-style pipeline: ``resolve_rules → enumerate all real-fact groundings of
    each body (branch) → pack into G → prune → standardize`` (rule→enumerate, NO parallel
    ``resolve_facts`` — the enumerate *is* the fact resolution). Pure enumerate emits only grounded children;
    the ``soft`` variant leaves a body's residual (no-real) var open for the shared
    ``resolve_soft_facts`` hook (top-1). Configure ``K``/``S``/``stats`` via :meth:`set_enumerate`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._enumerate: Optional[Tuple[int, int, Tensor]] = None   # (K, S, stats)

    def set_enumerate(self, k: int, s: int, stats: Tensor) -> None:
        """Wire the enumerate width ``K``, compaction width ``S``, and the running-max
        exactness ``stats`` (int64[2]) — called once by the app post-build."""
        self._enumerate = (int(k), int(s), stats)

    @torch.no_grad()
    def derive(self, current_states: Tensor, next_var_indices: Tensor,
               excluded_queries: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """One backward step: rule resolution → enumerate all real-fact groundings of each body →
        pack → prune → standardize. No parallel fact resolution (the enumerate is the fact resolution)."""
        if self._enumerate is None:
            raise RuntimeError("Enumerate engine used before set_enumerate(K, S, stats).")
        K, S, stats = self._enumerate
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
        ground_goals, ground_success, ground_rid = enumerate_groundings(
            rule_goals, rule_success, sub_rule_idx, kb.fact_index, kb.constant_no, pad,
            K, Y_r, S, leave_open=self._soft, stats=stats)

        # ── PACK the groundings into G (no fact children; the enumerate replaced them) ──
        n_ground = ground_goals.shape[1]
        empty_f = torch.full((B, 1, L, W), pad, dtype=torch.long, device=dev)
        empty_fs = torch.zeros(B, 1, dtype=torch.bool, device=dev)
        derived, counts, rule_idx = _pack_children(
            empty_f, empty_fs, ground_goals, ground_success, ground_rid, self.G, pad)

        # ── PRUNE + COMPACT: discharge subgoals that are known facts, left-align ──
        keep = _prune_ground_facts(
            derived, kb.fact_index.fact_hashes, kb.fact_index.pack_base,
            kb.constant_no, pad, excluded=excluded_queries)
        derived = _compact_atoms(derived, pad, valid=keep)

        # ── STANDARDIZE: rename any remaining (residual / tail) vars ──
        derived, new_next_var = standardize_vars(
            derived, next_var_indices, kb.constant_no, self.runtime_var_end_index, pad,
            enforce_runtime_range=self._enforce)

        slot_valid = torch.arange(self.G, device=dev).unsqueeze(0) < counts.unsqueeze(1)
        derived = torch.where(slot_valid.unsqueeze(-1).unsqueeze(-1), derived, pad)
        rule_idx = torch.where(slot_valid, rule_idx, torch.zeros_like(rule_idx))
        return derived, counts, new_next_var, rule_idx


__all__ = ["Enumerate", "enumerate_groundings"]
