"""Join engine — SLD derivation with **join** open-var resolution.

:class:`Join` is a sibling of :class:`~logic2rl.unification.sld.SLD` (both extend
:class:`~logic2rl.unification.base.engine.BaseEngine`, neither inherits the other). It reuses the
whole shared machinery — consult / pack / prune / ``prove`` / the ``resolve_soft_facts`` hook — and
overrides ``derive`` to ground each free variable with a REAL KB fact (``resolve_join_facts``)
*inside* the resolution, instead of a neural filler. In the soft variant it leaves the residual
(no-real-fact) vars open for the shared ``resolve_soft_facts`` hook to fill once. Configure the
enumerate width ``K``, the compaction width ``S`` (= q_kge_joint_s) and the exactness ``stats``
buffer once via :meth:`set_join`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from logic2rl.unification.base.engine import BaseEngine
from logic2rl.unification.base.kb import is_const, is_var


@torch.no_grad()
def resolve_join_facts(
    states: Tensor,                # [B, G, A, 3] derived states
    counts: Tensor,                # [B] valid slots per row
    fact_index,
    constant_no: int,
    pad: int,
    K: int,                        # enumerate width: real-fact fillers per state
    S: int,                        # open-var state compaction width
    stats: Optional[Tensor] = None,   # int64 [2] running-max (active states, needed K)
    leave_open: bool = False,      # SOFT variant: leave residual (no real fact) OPEN for soft to fill
) -> Tensor:
    """Join fact resolution: ground each derived state's free variable with a REAL
    KB fact — a resolution method of its own, the symbolic alternative to
    ``resolve_soft_facts`` (a one-var enumerate of the grounder's PBC join scheme).

    ``leave_open`` (the SOFT variant of the join): a state whose free var has NO real fact is
    left OPEN (its var uncommitted) so the caller's ``resolve_soft_facts`` fills it — real facts
    are preferred, soft only fills where the KB is incomplete. Default (pure join): the no-real
    state grounds to a non-fact slot-0 → V=0 (fails).
    A state's ≤2 soft atoms (one bound constant, one free var) share the var; the MORE
    SELECTIVE atom (fewer matching facts — ``targeted_count``) enumerates its ≤K fact
    groundings and the other atom filters them by hash membership; the FIRST surviving
    filler is committed. No scorer, no per-entity work: peak extra memory is O(S·K)
    int64 (fill column gathered directly, membership via arithmetic key packing — the
    [S, K, 3] triples are never materialized). A state with no real joint grounding
    commits enumerate slot 0, which grounds ≥1 atom to a NON-fact (sound: the caller's
    ground scoring maps non-facts to 0 = failure — see the case analysis in the body).
    States without a soft atom pass through unchanged."""
    B, G, A, W = states.shape
    assert W == 3, "resolve_join_facts: binary predicates only (W=3)"
    BS = B * G
    dev = states.device
    flat = states.reshape(BS, A, W)
    preds, a1, a2 = flat[:, :, 0], flat[:, :, 1], flat[:, :, 2]
    slot_ok = (torch.arange(G, device=dev).unsqueeze(0)
               < counts.unsqueeze(1)).reshape(BS, 1).expand(BS, A)
    pt = is_const(a1, constant_no) & is_var(a2, constant_no, pad) & slot_ok   # p(e, V)
    ph = is_var(a1, constant_no, pad) & is_const(a2, constant_no) & slot_ok   # p(V, e)
    soft = pt | ph                                                            # [BS, A]
    n_soft = soft.long().sum(1)
    active = n_soft >= 1

    # ── pack each state's ≤2 soft atoms as (pred, bound entity, head-free) ──
    posn = soft.long().cumsum(1) - 1
    tk0, tk1 = (soft & (posn == 0)).long(), (soft & (posn == 1)).long()
    sp, sbe, shf = preds.long(), torch.where(pt, a1, a2).long(), ph.long()
    p0, e0, h0 = (sp * tk0).sum(1), (sbe * tk0).sum(1), (shf * tk0).sum(1).bool()
    p1, e1, h1 = (sp * tk1).sum(1), (sbe * tk1).sum(1), (shf * tk1).sum(1).bool()

    # ── compact the ≤S active states (fixed shape; overflow tripwired via stats) ──
    S_ = min(S, BS)
    idx = torch.topk(active.float(), k=S_).indices                            # [S_]
    sa = active[idx]
    has2 = (n_soft >= 2)[idx]
    P0, E0, H0 = p0[idx], e0[idx], h0[idx]
    P1, E1, H1 = p1[idx], e1[idx], h1[idx]

    var = torch.full_like(E0, constant_no + 1)      # free-slot marker for the lookup

    def _look(p: Tensor, e: Tensor, hf: Tensor) -> Tensor:
        """[S_] (pred, bound entity, head-free) → [S_, 3] lookup atom
        ``p(e, V)`` tail-free / ``p(V, e)`` head-free."""
        return torch.stack([p, torch.where(hf, var, e), torch.where(hf, e, var)], dim=1)

    # ── join order: enumerate the smaller-span atom, membership-check the other ──
    l0, l1 = _look(P0, E0, H0), _look(P1, E1, H1)
    c0, c1 = fact_index.targeted_count(l0), fact_index.targeted_count(l1)
    swap = has2 & (c1 < c0)
    lE = torch.where(swap.unsqueeze(1), l1, l0)
    hE, cE = torch.where(swap, H1, H0), torch.where(swap, c1, c0)
    pM = torch.where(swap, P0, P1)
    eM = torch.where(swap, E0, E1)
    hM = torch.where(swap, H0, H1)

    fidx, fvalid = fact_index.targeted_lookup(lE, K)                          # [S_, K]
    # fill = the enumerated fact's free-slot column, gathered fused (no [S,K,3]).
    col = torch.where(hE, torch.ones_like(hE, dtype=torch.long),
                      torch.full_like(hE, 2, dtype=torch.long))               # head↦1, tail↦2
    fill = fact_index.facts_idx.reshape(-1).index_select(
        0, (fidx * W + col.unsqueeze(1)).reshape(-1)).reshape(S_, K)          # [S_, K]

    # member-atom hash membership via arithmetic key packing ([S_, K] int64 only).
    base = int(fact_index.pack_base)
    hcol = torch.where(hM.unsqueeze(1), fill, eM.unsqueeze(1))
    tcol = torch.where(hM.unsqueeze(1), eM.unsqueeze(1), fill)
    key = (pM.unsqueeze(1) * base + hcol) * base + tcol                       # [S_, K]
    fh = fact_index.fact_hashes
    F = fh.shape[0]
    ins = torch.searchsorted(fh, key.reshape(-1))
    mM = ((ins < F) & (fh[ins.clamp(max=F - 1)] == key.reshape(-1))).reshape(S_, K)

    # first surviving filler. Soundness of the no-survivor slot-0 commit: with a member
    # atom, it grounds to a non-fact (else the slot would survive); without one,
    # no-survivor ⟺ span 0 ⟺ NO fact matches the enum atom, so ANY fill is a non-fact.
    okm = fvalid & (mM | ~has2.unsqueeze(1))
    pos = torch.arange(K, device=dev).unsqueeze(0)
    first = torch.where(okm, pos, torch.full_like(pos, K)).amin(1)            # [S_]
    vstar_s = fill.gather(1, first.clamp(max=K - 1).unsqueeze(1)).squeeze(1)

    if stats is not None:
        # [0] peak active states (S overflow ⇒ uncommitted fillers — inexact);
        # [1] max span of a FAILED truncated enumeration (span > K ⇒ maybe a false fail).
        failed = sa & ~okm.any(1)
        need = torch.where(failed & (cE > K), cE, torch.zeros_like(cE))
        stats.copy_(torch.maximum(stats, torch.stack([active.sum(), need.amax()])))

    # ── scatter back and commit (identical to the resolve_soft_facts commit) ──
    vstar = torch.zeros(BS, dtype=torch.long, device=dev)
    vstar.scatter_(0, idx, torch.where(sa, vstar_s, torch.zeros_like(vstar_s)))
    vstar = vstar.view(B, G)
    args = states[..., 1:]
    softm = is_var(args, constant_no, pad) & is_const(args, constant_no).any(dim=-1, keepdim=True)
    if leave_open:
        # SOFT variant: commit only states with a REAL filler; leave the residual open var
        # untouched so the caller's resolve_soft_facts fills it (real facts preferred).
        real_found = torch.zeros(BS, dtype=torch.bool, device=dev)
        real_found.scatter_(0, idx, sa & okm.any(1))
        softm = softm & real_found.view(B, G, 1, 1)
    filled = torch.where(softm, vstar.unsqueeze(-1).unsqueeze(-1), args)
    return torch.cat([states[..., :1], filled], dim=-1)


class Join(BaseEngine):
    """SLD derivation with **join** as a RESOLUTION METHOD — see the module docstring.

    Sibling of :class:`~logic2rl.unification.sld.SLD`; inherits all of ``BaseEngine`` and overrides
    ``derive`` to ground the free variables with real KB facts *inside* the resolution, right after
    the base facts∥rules step — a fact lookup is cheap enough to run every derive (incl. the unary
    re-derives), unlike the soft GEMM. In the ``soft`` variant it leaves the RESIDUAL (free vars
    with no real fact) OPEN; those are filled ONCE by the shared ``resolve_soft_facts`` at
    ``BaseEngine.resolve_soft_facts`` (the hook, after the unary refine — NOT 3× in the loop). Pure
    join has no residual (no-real ⇒ non-fact slot-0 ⇒ V=0). Configure ``K``/``S``/``stats`` via
    :meth:`set_join` once at post-build."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._join: Optional[Tuple[int, int, Tensor]] = None   # (K, S, stats)

    def set_join(self, k: int, s: int, stats: Tensor) -> None:
        """Wire the enumerate width ``K``, compaction width ``S``, and the running-max
        exactness ``stats`` (int64[2]) — called once by the app post-build."""
        self._join = (int(k), int(s), stats)

    @torch.no_grad()
    def derive(self, current_states: Tensor, next_var_indices: Tensor,
               excluded_queries: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """One backward step: the shared facts∥rules resolution, then ground the derived free vars
        with real KB facts (``resolve_join_facts``). In the soft variant the residual (no real
        fact) is left OPEN for the ``resolve_soft_facts`` hook's ``resolve_soft_facts`` to fill once."""
        derived, counts, new_var, rule_idx = super().derive(
            current_states, next_var_indices, excluded_queries)
        if self._join is None:
            raise RuntimeError("Join engine used before set_join(K, S, stats).")
        k, s, stats = self._join
        derived = resolve_join_facts(derived, counts, self.kb.fact_index,
                                     self.kb.constant_no, self.kb.padding_idx, k, s, stats,
                                     leave_open=self._soft)
        return derived, counts, new_var, rule_idx


__all__ = ["Join", "resolve_join_facts"]
