"""KB — a logic program (facts + rules) indexed for one-step SLD resolution.

Semantically a fact is just a bodyless clause (as in Prolog); the fact/rule split here is
*clause indexing*, the vectorized analog of Prolog's first-argument indexing: ground unit
clauses admit a targeted ``(pred, arg)`` composite-key index (``FactIndex``) that bounds the
per-goal fact fan-out by the largest (pred, arg) group instead of the largest predicate,
while non-ground clauses fall back to a predicate-segment index (``RuleIndex``). The sorted
fact hash doubles as the O(log F) ground-membership test used to prune proven subgoals.

Contents:

  is_const / is_var    id convention (var boundary at constant_no+1; pad inert)
  pack_atoms           int64 Horner hash of ``[pred, arg1, …]`` atoms
  fact_contains        sorted-hash ground membership
  FactIndex            per-arg-column CSR lookup for resolve_facts (any arity)
  RuleIndex            predicate→rule segment lookup for resolve_rules
  KB                   the indexed program + the per-step branching budgets (K_f / K_r)

Read-only after construction. Scalar metadata (constant_no / predicate_no / padding_idx)
flows in from the IndexManager via the engine.

BIT-EXACT recipes (fingerprint-locked): pack key = Horner fold ``(((pred*base)+arg1)*base+…``
(at W=3 exactly the historic ``((pred*base)+arg0)*base+arg1``); CSR build = argsort(stable)
→ unique_consecutive → scatter cumulative counts at key+1 → cummax forward-fill.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# ==========================================================================
# Id-boundary predicates (the const/var/pad convention, defined once)
# ==========================================================================

def is_const(ids: Tensor, constant_no: int) -> Tensor:        # [...] -> [...] bool
    """An id is a CONSTANT iff ``id <= constant_no`` (the first variable id is
    ``constant_no + 1``; padding 0 reads as const)."""
    return ids < constant_no + 1


def is_var(ids: Tensor, constant_no: int, pad: int) -> Tensor:   # [...] -> [...] bool
    """An id is a VARIABLE iff ``id > constant_no and id != pad``."""
    return (ids >= constant_no + 1) & (ids != pad)


# ==========================================================================
# Atom hashing (sorted-hash membership)
# ==========================================================================

@torch.no_grad()
def pack_atoms(atoms: Tensor, base: int) -> Tensor:
    """Pack ``[.., W]`` atoms ``[pred, arg1, …]`` into int64 keys via a Horner fold over the W
    columns. ``base`` must exceed every id and satisfy ``base**W < 2**63`` (else distinct
    atoms collide — ``FactIndex`` asserts this at build)."""
    if atoms.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=atoms.device)
    a = atoms.long()
    key = a[..., 0]
    for j in range(1, a.shape[-1]):
        key = key * base + a[..., j]
    return key


def fact_contains(atoms: Tensor, fact_hashes: Tensor, pack_base: int) -> Tensor:
    """``[N, W] -> [N]`` bool ground membership via binary search on sorted hashes.

    BIT-EXACT: searchsorted then clamp the insertion point and test equality (the clamp keeps
    out-of-range keys — vars/padding — safe and reports them absent)."""
    N = atoms.shape[0]
    if N == 0 or fact_hashes.numel() == 0:
        return torch.zeros(N, dtype=torch.bool, device=atoms.device)
    keys = pack_atoms(atoms.long(), pack_base)
    F = fact_hashes.shape[0]
    insert = torch.searchsorted(fact_hashes, keys)
    return (insert < F) & (fact_hashes[insert.clamp(max=F - 1)] == keys)


# ==========================================================================
# Fact index (sorted-hash membership + per-arg targeted lookup, any arity)
# ==========================================================================

class FactIndex(nn.Module):
    """Sorted-hash fact set + O(1) targeted ``(pred, arg_j)`` composite-key lookup.

    One CSR segment table per argument column keyed by ``pred*key_scale + arg_j`` plus a
    predicate-only table for the all-variable case. ``targeted_lookup`` binds a goal by its
    LOWEST-index constant argument (arg0 wins over arg1 …); ``exists`` is the base sorted-hash
    membership. The atom width W comes from the facts tensor."""

    pack_base: int

    def __init__(self, facts_idx: Tensor, *, constant_no: int, padding_idx: int,
                 device: torch.device, pack_base: Optional[int] = None) -> None:
        super().__init__()
        if facts_idx.numel() == 0:
            raise ValueError("facts_idx is empty — cannot build a fact index without facts")
        self._constant_no = constant_no
        self._padding_idx = padding_idx
        self._W = int(facts_idx.shape[1])
        self._n_args = self._W - 1
        # BIT-EXACT: base must clear both the largest entity id and the padding sentinel so no
        # two distinct atoms collide to the same packed key.
        self.pack_base = (pack_base if pack_base is not None
                          else max(int(constant_no), int(padding_idx)) + 2)
        # Build-time overflow guard: the Horner fold must stay injective in int64.
        assert int(self.pack_base) ** self._W < (1 << 63), (
            f"atom hash overflow: pack_base={self.pack_base}^W={self._W} >= 2^63; "
            f"this arity×vocab is too large for the int64 membership hash.")

        facts = facts_idx.long().to(device)
        hashes = pack_atoms(facts, self.pack_base)
        sort_order = hashes.argsort()
        self.register_buffer("facts_idx", facts[sort_order])
        self.register_buffer("fact_hashes", hashes[sort_order])
        self._build_tables(device)

    @staticmethod
    def _segment_table(keys: Tensor, num_slots: int,
                       device: torch.device) -> Tuple[Tensor, Tensor]:
        """CSR segment index over composite keys → ``(order, offsets)``.

        BIT-EXACT: stable argsort, then place each unique key's cumulative count at
        ``offsets[key+1]`` and ``cummax`` to forward-fill empty keys."""
        order = keys.argsort(stable=True)
        unique, counts = torch.unique_consecutive(keys[order], return_counts=True)
        offsets = torch.zeros(num_slots + 1, dtype=torch.long, device=device)
        offsets[unique + 1] = counts.cumsum(0)
        offsets = offsets.cummax(0).values
        return order, offsets

    @staticmethod
    def _max_span(offsets: Tensor) -> int:
        if offsets.numel() < 2:
            return 1
        return max(int((offsets[1:] - offsets[:-1]).max().item()), 1)

    def _build_tables(self, device: torch.device) -> None:
        facts = self.facts_idx
        preds = facts[:, 0].long()
        ks = max(int(self._constant_no), int(self._padding_idx)) + 2
        self._key_scale = ks

        spans = []
        for j in range(self._n_args):
            argj = facts[:, 1 + j].long()
            order, off = self._segment_table(
                preds * ks + argj, int((preds * ks + argj).max()) + 2, device)
            self.register_buffer(f"_a{j}_order", order)
            self.register_buffer(f"_a{j}_offsets", off)
            spans.append(self._max_span(off))
        p_order, p_off = self._segment_table(preds, int(preds.max()) + 2, device)
        self.register_buffer("_p_order", p_order)
        self.register_buffer("_p_offsets", p_off)
        self._max_fact_pairs = max(max(spans, default=1), 1)

    @property
    def num_facts(self) -> int:
        return self.facts_idx.shape[0]

    @property
    def max_fact_pairs(self) -> int:
        return self._max_fact_pairs

    def exists(self, atoms: Tensor) -> Tensor:
        """``[N, W] -> [N]`` bool ground membership."""
        return fact_contains(atoms, self.fact_hashes, self.pack_base)

    def targeted_lookup(self, query_atoms: Tensor,
                        max_results: int) -> Tuple[Tensor, Tensor]:
        """Bind the free args of each goal. ``[B, W] -> (fact_idx [B, K], valid [B, K])``.

        BIT-EXACT precedence: index by the LOWEST-index constant argument; if every arg is a
        variable, fall back to the predicate-only table. ``valid`` requires an in-range slot
        AND the chosen argument actually being a constant (resp. a real predicate)."""
        B = query_atoms.shape[0]
        dev = query_atoms.device
        cno, pad, ks = self._constant_no, self._padding_idx, self._key_scale
        F = self._p_order.shape[0]
        clamp_max = max(F - 1, 0)
        preds = query_atoms[:, 0]
        pos = torch.arange(max_results, device=dev).unsqueeze(0)

        def _lookup(order, offsets, keys, ok):
            safe = keys.clamp(0, offsets.shape[0] - 2)
            left = offsets[safe]
            cnt = (offsets[safe + 1] - left).clamp(max=max_results)
            gi = (left.unsqueeze(1) + pos).clamp(0, clamp_max)
            valid = (pos < cnt.unsqueeze(1)) & ok.unsqueeze(1)
            return order[gi.reshape(-1)].reshape(B, max_results), valid

        # Base: predicate-only (used where all args are variables; pred must be real).
        all_var = (preds != pad)
        for j in range(self._n_args):
            argj = query_atoms[:, 1 + j]
            all_var = all_var & ~((argj <= cno) & (argj != pad))
        fact_idx, valid = _lookup(self._p_order, self._p_offsets, preds, all_var)

        # Override in REVERSE column order so the lowest-index constant arg wins.
        for j in reversed(range(self._n_args)):
            order = getattr(self, f"_a{j}_order")
            offsets = getattr(self, f"_a{j}_offsets")
            argj = query_atoms[:, 1 + j]
            is_cj = (argj <= cno) & (argj != pad)
            fij, vj = _lookup(order, offsets, preds * ks + argj, is_cj)
            use = is_cj.unsqueeze(1)
            fact_idx = torch.where(use, fij, fact_idx)
            valid = torch.where(use, vj, valid)
        return fact_idx, valid

    def __repr__(self) -> str:
        return f"FactIndex(F={self.num_facts}, W={self._W}, K={self._max_fact_pairs})"


# ==========================================================================
# Rule index (predicate→rule segment lookup)
# ==========================================================================

class RuleIndex(nn.Module):
    """Sorted rules with predicate→rule segment lookup (width-agnostic on the atom axis)."""

    def __init__(
        self,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        *,
        predicate_no: Optional[int] = None,
        padding_idx: int = 0,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        R = rules_heads_idx.shape[0]
        if R == 0:
            raise ValueError("rules_heads_idx is empty — cannot build a rule index without rules")

        sort_perm = torch.argsort(rules_heads_idx[:, 0], stable=True)
        heads = rules_heads_idx.index_select(0, sort_perm).to(device)
        bodies = rules_bodies_idx.index_select(0, sort_perm).to(device)
        lens = rule_lens.index_select(0, sort_perm).to(device)

        preds = heads[:, 0]
        uniq, cnts = torch.unique_consecutive(preds, return_counts=True)
        num_pred = (predicate_no + 1 if predicate_no is not None
                    else int(preds.max().item()) + 2)
        # BIT-EXACT CSR offsets: ``offsets[p+1]-offsets[p]`` is predicate p's rule count
        # (0 for absent predicates via the trailing cummax).
        seg = torch.zeros(num_pred + 1, dtype=torch.long, device=device)
        mask = uniq < num_pred
        seg[uniq[mask] + 1] = cnts.cumsum(0)[mask]
        seg_offsets = seg.cummax(0).values
        self._max_rule_pairs = int(cnts.max().item())
        self._M = int(lens.max().item())   # max body width — cached once (read every derive step)

        self.register_buffer("rules_heads_sorted", heads)
        self.register_buffer("rules_bodies_sorted", bodies)
        self.register_buffer("rules_idx_sorted", sort_perm.to(device))
        self.register_buffer("rule_lens_sorted", lens)
        self.register_buffer("_seg_offsets", seg_offsets)

    @property
    def num_rules(self) -> int:
        return self.rules_heads_sorted.shape[0]

    @property
    def max_rule_pairs(self) -> int:
        return self._max_rule_pairs

    @property
    def M(self) -> int:
        """Max rule-body width (the goal-tape needs room for the widest body); cached once."""
        return self._M

    @torch.no_grad()
    def lookup(self, query_preds: Tensor,
               max_pairs: int) -> Tuple[Tensor, Tensor]:
        """Predicate→rule segment lookup.

        ``[B] -> (item_idx [B, K], valid [B, K])`` where item_idx are rows of the sorted rule
        arrays and valid masks the live slots."""
        dev = query_preds.device
        qp = query_preds.long().clamp(0, self._seg_offsets.shape[0] - 2)
        starts = self._seg_offsets[qp]
        lens = (self._seg_offsets[qp + 1] - starts).clamp(max=max_pairs)
        pos = torch.arange(max_pairs, device=dev).unsqueeze(0)
        return starts.unsqueeze(1) + pos, pos < lens.unsqueeze(1)


# ==========================================================================
# KB — the indexed program + branching budgets
# ==========================================================================

class KB(nn.Module):
    """The logic program, indexed: ``fact_index`` + ``rule_index`` + the id boundary + the
    per-step branching budgets.

    ``K_f`` / ``K_r`` are the fact/rule slot budgets of one SLD step and ``max_children``
    their effective sum: ``K = min(K_f_raw + K_r, max_children)``; rules always fit, facts get
    the rest (floored at min(10, available facts)). ``full_fact_slots`` gives facts the full
    ``K`` instead (the SB3 reference's enumeration width — parity suites set it)."""

    def __init__(
        self,
        facts_idx: Tensor,
        rules_idx: Tensor,              # combined head+body per rule: [R, 1+max_body, W], head at slot 0
        *,
        constant_no: int,
        predicate_no: Optional[int],
        padding_idx: int,
        device: torch.device,
        pack_base: Optional[int] = None,
        max_children: int = 550,
        full_fact_slots: bool = False,
    ) -> None:
        super().__init__()
        if facts_idx.numel() == 0:
            raise ValueError("facts_idx is empty — a KB must have at least one fact")
        if rules_idx.shape[0] == 0:
            raise ValueError("rules_idx is empty — a KB must have at least one rule")

        self.constant_no = int(constant_no)
        self.predicate_no = int(predicate_no) if predicate_no is not None else int(constant_no)
        self.padding_idx = int(padding_idx)
        self.device_ = device

        facts_idx = facts_idx.to(device=device, dtype=torch.long)
        rules_idx = rules_idx.to(device=device, dtype=torch.long)
        # Split the combined rules tensor: head at slot 0, body after; the rule length is the
        # count of non-pad body atoms.
        rules_heads_idx = rules_idx[:, 0, :]                               # [R, W]
        rules_bodies_idx = rules_idx[:, 1:, :]                             # [R, max_body, W]
        rule_lens = (rules_bodies_idx[:, :, 0] != padding_idx).sum(dim=1)  # [R]

        self.fact_index = FactIndex(
            facts_idx, constant_no=constant_no, padding_idx=padding_idx,
            device=device, pack_base=pack_base)
        # Index tables are sized predicate_no + 1; padding appears in predicate slots of
        # inactive states, so the table must cover padding_idx.
        self.rule_index = RuleIndex(
            rules_heads_idx, rules_bodies_idx, rule_lens,
            predicate_no=max(self.predicate_no, self.padding_idx),
            padding_idx=padding_idx, device=device)

        # ── branching budgets ──
        K_f, K_r = self.fact_index.max_fact_pairs, self.rule_index.max_rule_pairs
        K = min(K_f + K_r, int(max_children))
        min_facts = min(10, K_f)
        if K_r > K - min_facts:
            raise ValueError(f"K_r={K_r} leaves fewer than {min_facts} fact slots in K={K}.")
        K_f_budget = max(K - K_r, min_facts)
        if K_f > K_f_budget:
            warnings.warn(f"[KB] K_f capped {K_f}->{K_f_budget} (K_r={K_r}, K={K}).",
                          stacklevel=2)
            K_f = K_f_budget
        self.K_f = K if full_fact_slots else K_f
        self.K_r = K_r
        self.max_children = K

    def __repr__(self) -> str:
        return (f"KB(facts={self.fact_index.num_facts}, rules={self.rule_index.num_rules}, "
                f"entities={self.constant_no}, predicates={self.predicate_no})")


__all__ = [
    "is_const", "is_var", "pack_atoms", "fact_contains",
    "FactIndex", "RuleIndex", "KB",
]
