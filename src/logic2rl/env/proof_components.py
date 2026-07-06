"""Generic proof-MDP env components (pillar: base).

Optional proof-search behavior layered on the env's invariant proof core (the env owns
all-TRUE ⇒ success / any-FALSE ⇒ fail). Base owns only the task-agnostic unary auto-advance
(``UnaryAdvanceComponent``), wired via ``base.builder.make_base_components`` alongside visit-memory
(``base/env/memory.py``). The KGE app's proof components — the endf early-stop action, the
per-slot rule id, the max-depth accept/reject forcing, the terminal-outcome taxonomy — live in
``kge/env/proof_components.py``, wired by the KGE config; base never imports them.
"""
from __future__ import annotations

from typing import Any

import torch

from logic2rl.env.component import EnvComponent

Tensor = torch.Tensor


class UnaryAdvanceComponent(EnvComponent):
    """Auto-advance through forced single-successor (unary) proof states.

    While a state's only candidate is a single non-terminal successor, step into it for free
    (no agent action), up to ``max_unary_iterations`` — matching the SB3 reference. The
    component owns its whole config (passed at construction from the task config, like the kge
    endf action's ``enabled``): ``skip_unary_actions`` gates the feature (inert when off);
    ``unification_safe`` refuses to advance a successor that still carries a free runtime var
    (would leave two distinct free vars after splicing the next body — required by Q_KGE's
    joint scoring). Implemented as a ``candidate_refine`` hook ordered BEFORE any stop-action
    component's append (e.g. the kge endf action), so it sees end-free candidates (and the stop
    is appended once after). Re-derives via the candidate-gen ``_derive_step`` primitive and
    updates the visit-history via ``env.memory`` when memory is present.
    """
    name = "unary_advance"

    def __init__(self, skip_unary_actions: bool = False, max_unary_iterations: int = 2,
                 unification_safe: bool = False) -> None:
        self.skip_unary_actions = bool(skip_unary_actions)
        self.max_unary_iterations = int(max_unary_iterations)
        self.unification_safe = bool(unification_safe)

    def candidate_refine(self, env: Any, cand, state):
        if not self.skip_unary_actions:
            return cand
        bsz, dev = env.batch_size, env.device
        # The episode's root query (cycle-prevention exclusion): the carried original_queries
        # at a step; at reset (state is None) the bundle's current state IS the root query.
        original_queries = state.original_queries if state is not None else cand.current_states
        excluded = original_queries[:, 0:1, :]
        cur, derived, counts, var = cand.current_states, cand.derived, cand.counts, cand.next_var
        rule_idx, fields = cand.derived_rule_idx, dict(cand.fields)
        hh, hc = fields.get("history_hashes"), fields.get("history_count")
        # Unification-safe mode: don't advance a successor with a free var. None ⇒ no var check.
        var_floor = env.runtime_var_start_index if self.unification_safe else None

        for _ in range(self.max_unary_iterations):
            successor = derived[:, 0]                              # [B, A, W]  the sole candidate (slot 0)
            first_pred = successor[:, 0, 0]
            # Never auto-advance into a terminal successor — that's a real outcome the agent must
            # own, not a forced step. The terminal predicate set is contributed by the env +
            # components (``env.terminal_pred_ids``), so unary names no predicate.
            is_terminal = torch.zeros(bsz, dtype=torch.bool, device=dev)
            for p in env.terminal_pred_ids:
                is_terminal = is_terminal | (first_pred == p)
            advance = (counts == 1) & ~is_terminal                # advance iff exactly one non-terminal candidate
            if var_floor is not None:
                advance = advance & ~(successor[:, :, 1:] >= var_floor).any(dim=(1, 2))

            cur = torch.where(advance.view(bsz, 1, 1), successor, cur)   # step the advancing envs in
            if hh is not None:                                    # memory present → record the new state's hash
                hh, hc = env.memory._update_history(cur, hh, hc, advance)
            # Re-derive from the advanced state. The candidate keep-masks read the working fields
            # (e.g. the updated visit-history) — thread the current hh/hc.
            step_fields = fields if hh is None else {**fields, "history_hashes": hh, "history_count": hc}
            new_derived, new_counts, new_var, new_rid = env.unification_logic._derive_step(
                cur, var, step_fields, state, excluded=excluded)
            # Commit the re-derived candidates ONLY for the advancing envs; others keep theirs.
            derived = torch.where(advance.view(bsz, 1, 1, 1), new_derived, derived)
            counts = torch.where(advance, new_counts, counts)
            var = torch.where(advance, new_var, var)
            rule_idx = torch.where(advance.view(bsz, 1), new_rid, rule_idx)   # rule id follows the same mask

        if hh is not None:                                        # repack updated history into fields
            fields = {**fields, "history_hashes": hh, "history_count": hc}
        return cand._replace(current_states=cur, derived=derived, counts=counts,
                             next_var=var, derived_rule_idx=rule_idx, fields=fields)
