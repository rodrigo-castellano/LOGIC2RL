"""Vectorized unification — the single-step SLD derivation engine, for logic programs of any arity.

Three packages: ``base/`` is the shared substrate (``BaseEngine``: consult ≈ Prolog
``consult``, ``derive`` = one backward step the RL env drives, ``prove`` = the reference solver;
plus ``kb`` / ``resolution`` / ``soft``). ``sld/`` and ``join/`` are the two concrete engines —
siblings that both extend ``BaseEngine`` and differ ONLY in ``resolve_soft_facts`` (soft neural
filler vs real-fact join). Atom width W = max_arity + 1 is read off the program tensors; KGE
runs W=3, the MNIST example W=6.

The env drives the engine through the :class:`Grounder` contract below; a config picks the
engine via ``build_env``'s ``engine_cls`` parameter (default :class:`SLD`). Implement :class:`Grounder`
to swap in another resolution strategy — the builder/env need nothing else.
"""
from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

from torch import Tensor

from logic2rl.unification.base import (KB, BaseEngine, FactIndex, RuleIndex,
                                       fact_contains, is_const, is_var)
from logic2rl.unification.join import Join
from logic2rl.unification.sld import SLD


@runtime_checkable
class Grounder(Protocol):
    """The contract the env/builder rely on — one verb + the sizing read-outs that can only
    be known after the program is indexed. :class:`SLD` implements it."""

    max_children: int      # max successors a state can derive in one step → builder sets padding_states
    total_vocab_size: int  # token-id ceiling AND collision-free hash base (the KB's pack_base)
    n_vars: int            # runtime-var embedder table size
    num_rules: int         # program size (the rule-bias H provider's table size)

    def derive(self, current_states: Tensor, next_var: Tensor,
               excluded: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """One backward step: current_states ``[B, A, W]``, next_var ``[B]`` →
        ``(derived [B, G, A, W], counts [B], next_var [B], derived_rule_idx [B, G])`` (W = max_arity+1).

        Semantic contract (what the env reads off the output — a custom engine must honor it):
          * a successor with ZERO non-padding atoms marks a completed proof — the env collapses
            that batch entry to a single TRUE state (its termination marker);
          * ``counts[b] == 0`` means no successor exists — the env falls back to a single FALSE
            state (dead end);
          * ``excluded`` ``[B, 1, W]`` (optional) is the episode's root query atom — don't unify
            against it as a KB fact (cycle prevention);
          * ``next_var`` is the fresh-runtime-var allocator, advanced and returned;
          * atoms are FLAT: ``(predicate, arg1, …, arg_max_arity)`` over constant / runtime-var
            ids — no function symbols / nested terms (Datalog-style programs).
        """
        ...


__all__ = ["Grounder", "BaseEngine", "SLD", "Join",
           "KB", "FactIndex", "RuleIndex", "fact_contains", "is_const", "is_var"]
