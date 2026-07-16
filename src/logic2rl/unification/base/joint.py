"""FactJoint — enumerate ALL real-fact groundings of open-var states (the fact joint).

The fact-side joint unifier: given states whose atoms share ≤1 free variable, enumerate every
binding that makes ALL the var's atoms real KB facts (the semijoin: enumerate the more-selective
atom's ≤K fillers, membership-filter the other) — no top-k, no scoring; bindings that fulfil the
condition are emitted, the rest are dropped. This is a RESOLUTION-level primitive (it expands one
state into many grounded children), so it runs inside ``derive`` — the shape-preserving
``replace_candidates`` seam cannot host an expansion (proof-marking/prune/compaction run before
it). :class:`~logic2rl.unification.enumerate.Enumerate` is the engine built on it; SLD's
``derive`` could adopt it for fact-expansion variants. The neural counterpart (top-k SCORED
assignments over all entities / facts) is the app-side KGE joint attached as
``engine.joint_scorer``."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from logic2rl.unification.base.kb import is_const, is_var


class FactJoint:
    """All-real-fact groundings of open-var rule bodies — see the module docstring.

    Holds the fact index and the compile-time widths (``K`` fillers per lookup, ``S``
    open-var-state compaction, the running-max exactness ``stats`` int64[2]); per-call
    tensors are the ``resolve_rules`` outputs. Fixed shapes → CUDA-graph safe."""

    def __init__(self, fact_index, constant_no: int, pad: int, *,
                 K: int, S: int, stats: Optional[Tensor] = None) -> None:
        self.fact_index = fact_index
        self.constant_no = int(constant_no)
        self.pad = int(pad)
        self.K = int(K)
        self.S = int(S)
        self.stats = stats

    @torch.no_grad()
    def all_groundings(self, rule_goals: Tensor, rule_success: Tensor, sub_rule_idx: Tensor,
                       Y_r: int, leave_open: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """Enumerate ALL real-fact groundings of each rule body — the branching enumerate.

        ``rule_goals [B, K_r, L, 3]`` are the substituted rule bodies (from ``resolve_rules``);
        each has ≤1 shared free var over ≤2 soft atoms. For every body we enumerate up to ``Y_r``
        bindings of that var that make the body a real KB fact, and emit ONE grounded body per
        binding. Returns ``(ground_goals [B, K_r*Y_r, L, 3], ground_success [B, K_r*Y_r],
        ground_rid [B, K_r*Y_r])``.

        Memory-lean: the ≤``S`` bodies with a free var are compacted first (``topk``), so the
        ``[·, K]`` enumerate is ``O(S·K)`` not ``O(B·K_r·K)`` — bounded on yago-scale KBs. A rule
        body already ground (no free var) passes through as one child. With ``leave_open`` (soft
        variant) a body with NO real binding emits ONE child with its var left open (for the
        ``replace_candidates`` fill); without it that body dies. Fixed shapes (static ``S``,
        ``K``, ``Y_r``) → CUDA-graph safe. ≤1-var / ≤2-atom case; general V-var / M-atom joint
        is the next extension."""
        fact_index, constant_no, pad = self.fact_index, self.constant_no, self.pad
        K, S, stats = self.K, self.S, self.stats
        B, K_r, L, W = rule_goals.shape
        assert W == 3, "FactJoint.all_groundings: binary predicates only (W=3)"
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


__all__ = ["FactJoint"]
