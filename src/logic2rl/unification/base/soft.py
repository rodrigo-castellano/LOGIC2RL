"""Soft-fill commit — write a chosen variable assignment into derived states.

``fill_vars`` is the pure commit primitive shared by the engines' ``*_fill_vars`` methods
(``BaseEngine.soft_fill_vars`` / ``SLD.fact_fill_vars``): given the filler entity ``v*`` per
state (chosen by an app-attached joint scorer), it writes ``v*`` into every free-variable slot
of each state's soft atoms. With ``no_fact`` + ``false_pred`` it instead discards states that
have NO real-fact filler (fact fill) by replacing them with a FALSE terminal. Fixed-shape /
CUDA-graph-safe (no ``.item()``, no data-dependent branching)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from logic2rl.unification.base.kb import is_const, is_var


@torch.no_grad()
def fill_vars(
    states: Tensor,                # [B, G, A, W] derived states
    vstar: Tensor,                 # [B, G] filler entity per state (the joint argmax)
    constant_no: int,
    pad: int,
    no_fact: Optional[Tensor] = None,    # [B, G] states with NO real-fact filler (fact fill)
    false_pred: Optional[int] = None,    # FALSE-terminal pred; with no_fact set, discard those
) -> Tensor:
    """Commit the chosen filler: unify each derived state's free variable with ``v*``.
    A *soft* atom has a bound constant arg and a free variable — no KB fact matches it
    exactly, but a joint scorer can rank every candidate binding. Committing ``v*`` makes
    the state ground; states without a soft atom pass through unchanged.

    **Discard (fact fill).** A ``no_fact`` state is an open-var state with NO real-KB-fact
    filler. With ``false_pred`` set, such a state is replaced by a single-atom FALSE terminal
    (the env's ``any-FALSE ⇒ fail``) — the branch is discarded instead of committing a
    garbage entity. Without ``no_fact``/``false_pred`` (soft fill) every state commits."""
    args = states[..., 1:]                                       # [B, G, A, W-1]
    soft = is_var(args, constant_no, pad) & is_const(args, constant_no).any(dim=-1, keepdim=True)
    filled = torch.where(soft, vstar.unsqueeze(-1).unsqueeze(-1), args)
    result = torch.cat([states[..., :1], filled], dim=-1)
    if no_fact is not None and false_pred is not None:
        B, G, A, W = result.shape
        false_state = torch.full((A, W), pad, dtype=result.dtype, device=result.device)
        false_state[0, 0] = int(false_pred)                     # [A, W] = FALSE atom + padding
        result = torch.where(no_fact.view(B, G, 1, 1), false_state.view(1, 1, A, W), result)
    return result


__all__ = ["fill_vars"]
