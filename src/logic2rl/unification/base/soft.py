"""Soft unification — the neural open-var grounding shared by the SLD and Enumerate engines.

``resolve_soft_facts`` commits each derived state's free variable to its most likely neural
filler (the joint argmax over the state's soft atoms, returned by an attached scorer). It is
the base ``SLD.resolve_soft_facts`` and the fallback filler in ``Enumerate`` when ``soft`` is on — one
implementation, shared, so both engines score soft steps identically. Fixed-shape /
CUDA-graph-safe (no ``.item()``, no data-dependent branching)."""
from __future__ import annotations

import torch
from torch import Tensor

from logic2rl.unification.base.kb import is_const, is_var


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
    without a soft atom pass through unchanged. A scorer is always required (the caller only
    invokes this when soft grounding is on)."""
    assert score_soft_facts is not None, "resolve_soft_facts requires a soft_scorer"
    args = states[..., 1:]                                       # [B, G, A, W-1]
    soft = is_var(args, constant_no, pad) & is_const(args, constant_no).any(dim=-1, keepdim=True)
    vstar = score_soft_facts(states, counts)                     # [B, G]
    filled = torch.where(soft, vstar.unsqueeze(-1).unsqueeze(-1), args)
    return torch.cat([states[..., :1], filled], dim=-1)


__all__ = ["resolve_soft_facts"]
