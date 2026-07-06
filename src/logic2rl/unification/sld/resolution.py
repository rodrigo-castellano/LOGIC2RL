"""SLD resolution — unification primitives + fact/rule resolution + var standardization.

The resolve stage of one backward SLD step, for atoms of any width W (``[pred, arg1, …]``):

  unify_atoms              pairwise MGU of two atom tensors
  apply_substitutions      sequential substitution of ``(from → to)`` slots
  resolve_facts            targeted fact lookup → unify → substitute the remaining goals
  resolve_rules            rule segment lookup → standardize-apart → unify head → body + tail
  resolve_soft_facts       neural fact lookup: unify each free variable with its top filler
  standardize_vars         derived-state variable renaming (terminal output)

Facts and rules are resolved independently and side by side (dense); the engine packs the
two child sets afterwards. All ops are fixed-shape / CUDA-graph-safe (no ``.item()``, no
data-dependent branching); id classification goes through ``is_const`` / ``is_var``. The
recipes are BIT-EXACT at W=3 with the historic binary engine (fingerprint-locked): the
standardize-apart base, the ``id > constant_no & id != pad`` var test, the offset renaming.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch import Tensor

from logic2rl.unification.sld.kb import is_const, is_var

# ==========================================================================
# Unification primitives
# ==========================================================================

@torch.no_grad()
def unify_atoms(a: Tensor, b: Tensor, *, constant_no: int, pad: int) -> Tuple[Tensor, Tensor]:
    """Pairwise MGU of two ``[..., W]`` atoms → (ok ``[...]`` bool, subs ``[..., W-1, 2]``
    per-arg ``(from, to)``, ``pad`` = no binding). Fails on predicate mismatch, constant
    clash, or one variable bound to two different targets (checked over all arg pairs)."""
    pad_t = torch.tensor(pad, dtype=a.dtype, device=a.device)
    n = a.shape[-1] - 1

    pred_ok = a[..., 0] == b[..., 0]                             # [...]
    qa, ta = a[..., 1:], b[..., 1:]                              # [..., n]
    qc, tc = is_const(qa, constant_no), is_const(ta, constant_no)
    qv, tv = is_var(qa, constant_no, pad), is_var(ta, constant_no, pad)

    const_clash = (qc & tc & (qa != ta)).any(dim=-1)             # [...]
    ok = pred_ok & ~const_clash

    bind_q = qv & tc                                             # a-var ↦ b-const
    bind_t = tv & (qa != pad)         # b-var ↦ any non-pad a-term ((qc|qv) ≡ a != pad)
    frm = torch.where(bind_q, qa, torch.where(bind_t, ta, pad_t))   # [..., n]
    to_ = torch.where(bind_q, ta, torch.where(bind_t, qa, pad_t))   # [..., n]
    subs = torch.stack([frm, to_], dim=-1)                       # [..., n, 2]

    # Same-var clash over all arg pairs: one var bound to two different targets.
    clash = torch.zeros_like(ok)
    for i in range(n):
        for j in range(i + 1, n):
            same = (subs[..., i, 0] == subs[..., j, 0]) & (subs[..., i, 0] != pad)
            clash = clash | (same & (subs[..., i, 1] != subs[..., j, 1]))
    ok = ok & ~clash

    subs = torch.where((~ok).unsqueeze(-1).unsqueeze(-1), pad_t, subs)
    return ok, subs


@torch.no_grad()
def apply_substitutions(atoms: Tensor, subs: Tensor, pad: int) -> Tensor:
    """Apply substitution slots ``(from → to)`` to atom args: atoms ``[N, M, W]``,
    subs ``[N, S, 2]``.

    SEQUENTIAL over the S slots (slot s sees slot s-1's output) so chained var→var→const
    bindings resolve. A ``pad → -1`` sentinel folds the validity guard into the tiny operand
    (-1 is absent from real arg ids)."""
    if atoms.numel() == 0:
        return atoms
    N = atoms.shape[0]
    S = subs.shape[1]
    preds = atoms[:, :, 0:1]                          # [N, M, 1] (view)
    out = atoms[:, :, 1:]                             # [N, M, W-1] (view; first where allocates)
    neg = torch.tensor(-1, dtype=out.dtype, device=out.device)
    for s in range(S):
        frm = torch.where(subs[:, s, 0] != pad, subs[:, s, 0], neg).view(N, 1, 1)
        out = torch.where(out == frm, subs[:, s, 1].view(N, 1, 1), out)
    return torch.cat([preds, out], dim=2)


# ==========================================================================
# Fact / rule resolution (one goal slot per batch row)
# ==========================================================================

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
    episode's root query atom (cycle prevention)."""
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


def resolve_rules(
    queries: Tensor,               # [B, W]
    remaining: Tensor,             # [B, L, W]
    rule_index,
    constant_no: int,
    pad: int,
    K_r: int,
    active: Tensor,                # [B]
    next_var: Tensor,              # [B]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Resolve goals against rule heads via MGU + standardization-apart.

    Returns ``(rule_goals [B, K_r, L, W], success [B, K_r], sub_rule_idx [B, K_r])``;
    a rule child = substituted body at ``[:Bmax]``, then the substituted remaining tail."""
    B, W = queries.shape
    L = remaining.shape[1]
    dev = queries.device
    E = constant_no + 1
    Bmax = rule_index.rules_bodies_sorted.shape[1]

    # ── segment rule lookup ──
    sorted_pos, mask = rule_index.lookup(queries[:, 0], K_r)                 # [B, K_r]
    R = rule_index.rules_heads_sorted.shape[0]
    safe_pos = sorted_pos.clamp(0, max(R - 1, 0))
    sub_rule_idx = rule_index.rules_idx_sorted[safe_pos].view(B, K_r)
    flat_pos = safe_pos.reshape(-1)
    sub_heads = rule_index.rules_heads_sorted[flat_pos]                      # [N_r, W]
    sub_bodies = rule_index.rules_bodies_sorted[flat_pos]                    # [N_r, Bmax, W]
    sub_lens = rule_index.rule_lens_sorted[flat_pos]                         # [N_r]
    N_r = B * K_r

    # ── standardization apart: each (row, rule) gets a fresh var namespace ──
    rule_var_base = next_var.view(B, 1).expand(B, K_r).reshape(N_r)
    is_var_h = sub_heads[:, 1:] >= E
    std_heads = torch.cat([sub_heads[:, 0:1],
                           torch.where(is_var_h, sub_heads[:, 1:] - E + rule_var_base.unsqueeze(1),
                                       sub_heads[:, 1:])], dim=1)
    is_var_b = sub_bodies[:, :, 1:] >= E
    std_bodies = torch.cat([sub_bodies[:, :, 0:1],
                            torch.where(is_var_b, sub_bodies[:, :, 1:] - E + rule_var_base.view(N_r, 1, 1),
                                        sub_bodies[:, :, 1:])], dim=2)

    # ── unify head with goal ──
    flat_q = queries.unsqueeze(1).expand(B, K_r, W).reshape(N_r, W)
    ok_flat, subs_flat = unify_atoms(flat_q, std_heads, constant_no=constant_no, pad=pad)
    rule_success = ok_flat.view(B, K_r) & mask & active.unsqueeze(-1)

    # ── apply subs to body and the LIVE remaining slice SEPARATELY ──
    body_subst = apply_substitutions(std_bodies, subs_flat, pad).view(B, K_r, Bmax, W)
    n_rem = L - Bmax                                    # ≥ 0 (Bmax ≤ L invariant)
    # DROP, never truncate: a goal whose live remaining beyond n_rem won't fit body+tail in
    # L atoms would silently lose those atoms in EVERY child — invalidate its children.
    if n_rem < L:
        overflow = (remaining[:, n_rem:, 0] != pad).any(-1)                  # [B]
        rule_success = rule_success & ~overflow.unsqueeze(-1)
    rem_exp = remaining[:, :n_rem, :].unsqueeze(1).expand(B, K_r, n_rem, W).reshape(N_r, n_rem, W)
    rule_remaining = apply_substitutions(rem_exp, subs_flat, pad).view(B, K_r, n_rem, W)

    # ── mask body atoms beyond each rule's length; assemble body then remaining ──
    atom_idx = torch.arange(Bmax, device=dev).view(1, 1, Bmax)
    inactive = atom_idx >= sub_lens.view(B, K_r).unsqueeze(-1)               # [B, K_r, Bmax]
    body_subst = torch.where(inactive.unsqueeze(-1), pad, body_subst)
    rule_goals = torch.cat([body_subst, rule_remaining], dim=2)              # [B, K_r, L, W]
    return rule_goals, rule_success, sub_rule_idx


@torch.no_grad()
def resolve_soft_facts(
    states: Tensor,                # [B, G, A, W] derived states
    counts: Tensor,                # [B] valid slots per row
    score_soft_facts,              # (states, counts) -> v* [B, G] best filler per state
    constant_no: int,
    pad: int,
) -> Tensor:
    """Soft-fact resolution: unify each derived state's free variable with its most
    likely filler. A *soft* atom has a bound constant arg and a free variable — no KB
    fact matches it exactly, but a neural scorer can rank every candidate binding.
    ``score_soft_facts`` returns the top filler entity per state (the joint argmax
    over that state's soft atoms); committing it makes the state ground. States
    without a soft atom pass through unchanged."""
    args = states[..., 1:]                                       # [B, G, A, W-1]
    soft = is_var(args, constant_no, pad) & is_const(args, constant_no).any(dim=-1, keepdim=True)
    vstar = score_soft_facts(states, counts)                     # [B, G]
    filled = torch.where(soft, vstar.unsqueeze(-1).unsqueeze(-1), args)
    return torch.cat([states[..., :1], filled], dim=-1)


# ==========================================================================
# Variable standardization (terminal output renaming)
# ==========================================================================

# Hot-path var-id range guards: each runs a `.all()` reduction + a launched assert kernel per
# step. Active when the env's enforce_runtime_var_range is set or the GROUNDER_STD_ASSERTS=1
# env override; off by default to keep the compiled step host-sync-free.
_STD_ASSERTS = os.environ.get("GROUNDER_STD_ASSERTS") == "1"


def standardize_vars(
    states: Tensor,                       # [B, K, M, W]
    next_var: Tensor,                     # [B]
    constant_no: int,
    runtime_var_end_index: Optional[int],
    padding_idx: int,
    input_states: Optional[Tensor] = None,  # unused (kept for call-site compat)
    extra_new_vars: int = 15,               # unused (kept for call-site compat)
    enforce_runtime_range: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Renumber derived-state variables down to the fixed runtime base (out of place).

    Per batch row, the smallest live var id is translated to ``constant_no + 1`` (uniform
    shift — preserves distinctness and order, so it is substitution-safe), and
    ``new_next_var = max renumbered var + 1``. Anchoring at the FIXED base every step keeps
    ids linear in the per-episode distinct-var count; the previous scheme lifted the live
    set above ``next_var`` each step, ratcheting ids quadratically with depth until they
    overran the runtime range / embedder vocabulary on deep episodes.
    Returns ``(standardized [B, K, M, W], new_next_var [B])``."""
    device = states.device
    B, K, M, _ = states.shape
    pad = padding_idx
    if B == 0 or states.numel() == 0:
        return states, next_var

    LARGE = 1_000_000
    start = constant_no + 1
    args = states[:, :, :, 1:]
    is_var_out = (args > constant_no) & (args != pad)
    has_vars = is_var_out.any(dim=(1, 2, 3))                                 # [B]
    large_t = torch.tensor(LARGE, dtype=args.dtype, device=device)
    min_var = torch.where(is_var_out, args, large_t).amin(dim=(1, 2, 3))     # [B]
    offset = torch.where(has_vars, start - min_var,
                         torch.zeros_like(next_var))
    std_args = torch.where(is_var_out, args + offset.view(B, 1, 1, 1), args)

    if (enforce_runtime_range or _STD_ASSERTS) and runtime_var_end_index is not None:
        lo = ((~is_var_out) | (std_args >= start)).all()
        hi = ((~is_var_out) | (std_args <= runtime_var_end_index)).all()
        torch._assert_async(lo, "standardize_vars: var id below runtime range")
        torch._assert_async(hi, "standardize_vars: var id above runtime range")

    standardized = states.clone()
    standardized[:, :, :, 1:] = std_args

    # next_var MUST exceed every variable in the output: it is the base for the NEXT step's
    # rule standardize-apart. If it only equals the max live var, a fresh rule head var
    # aliases a live var in the remaining goals and the head's var↦const unification leaks
    # that constant into them.
    max_var_out = torch.where(is_var_out, std_args,
                              torch.zeros_like(std_args)).amax(dim=(1, 2, 3))   # [B]
    new_next_var = torch.maximum(max_var_out + 1,
                                 torch.full_like(next_var, start))

    if (enforce_runtime_range or _STD_ASSERTS) and runtime_var_end_index is not None:
        torch._assert_async((new_next_var <= runtime_var_end_index).all(),
                            "standardize_vars: next-var beyond runtime range")
    return standardized, new_next_var


__all__ = [
    "unify_atoms", "apply_substitutions",
    "resolve_facts", "resolve_rules", "resolve_soft_facts", "standardize_vars",
]
