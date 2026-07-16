"""Candidate generation (pillar: base).

Turns the current proof state into the next set of candidate states: run the engine one
step, mark proofs / dead ends (the grounder is vocab-free, so the env owns this), drop
invalid/visited candidates, and compact to static shapes. Pure mechanism — feature behavior
(the unary auto-advance, stop-action appends, visit-memory pruning) comes from the
components, invoked here through the ``candidate_*`` hooks.

The engine returns ``derived_rule_idx`` ([B, G] — which top-level rule produced each
candidate) alongside the derived states. This component threads it in lock-step with
``derived`` (same compaction) so it stays slot-aligned; the kge ``RuleIdComponent`` carries
it as recurrent state and exposes it to the policy.

All state is read explicitly off ``self.env`` (dimensions, vocab ids, index helpers); the
only tensors owned here are the TRUE/FALSE marker states and the compaction scratch.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch

Tensor = torch.Tensor


class _Cand(NamedTuple):
    """Candidate next-states from the candidate-gen seam → consumed by the env step's
    assembly. Built/destructured inside the compiled step; never persisted. ``fields`` carries
    component-owned recurrent-state updates (e.g. memory's visit-history); ``derived_rule_idx``
    is the engine's per-candidate top-level rule id ([B, G], 0 on padding)."""
    current_states: Tensor          # [B, A, W]  post-unary committed current
    derived: Tensor                 # [B, G, A, W]
    counts: Tensor                  # [B]
    next_var: Tensor                # [B]
    derived_rule_idx: Tensor        # [B, G] which top-level rule produced each candidate (0 = padding)
    fields: dict                    # component state-field updates (e.g. history_hashes/count)


class UnificationLogic:
    """Derived-state construction: ``_derive_step`` (raw derive → keep-mask → compact) is the
    single pipeline primitive; ``_compute_derived`` / ``_compute_initial`` wrap it with the
    step / reset field handling and run the component ``candidate_refine`` seam (where the
    unary advance and the stop-action appends live)."""

    def __init__(self, env: Any) -> None:
        self.env = env
        # Proof / dead-end marker states, built from the IM's True/False ids (the env asserts
        # both exist):
        #   _false_state — the no-survivor FALSE fallback ([G, A, W], single False atom);
        #   _true_state  — the proof TRUE atom ([A, W]) written into a fully-resolved slot.
        G, A, W, pad = env.padding_states, env.padding_atoms, env.atom_width, env.padding_idx
        fs = torch.full((G, A, W), pad, dtype=torch.long, device=env.device)
        fs[0, 0, 0] = env.false_pred_idx
        self._false_state = fs
        ts = torch.full((A, W), pad, dtype=torch.long, device=env.device)
        ts[0, 0] = env.true_pred_idx
        self._true_state = ts
        # Compaction scratch (cudagraph-stable address), owned by the compaction below.
        self._compact_scratch = torch.zeros(env.batch_size, G, A * W, dtype=torch.long, device=env.device)

    def _compute_derived(self, new_current: Tensor, state) -> "_Cand":
        """Candidate next-states for the post-action ``new_current`` (the step path).

        Components update their working fields (``step_update_fields`` — e.g. memory appends
        the current hash) → ``_derive_step`` → components refine (``candidate_refine`` — e.g.
        unary auto-advance, stop-action appends), in component order.
        """
        env = self.env
        active = ~state.done.bool()
        fields = {k: getattr(state, k) for k in env._component_fields}
        for c in env.components:
            fields.update(c.step_update_fields(env, new_current, fields, active))
        derived, counts, new_var, rule_idx = self._derive_step(
            new_current, state.next_var_indices, fields, state,
            excluded=state.original_queries[:, 0:1, :])
        cand = _Cand(current_states=new_current, derived=derived, counts=counts, next_var=new_var,
                     derived_rule_idx=rule_idx, fields=fields)
        for c in env.components:
            cand = c.candidate_refine(env, cand, state)
        return cand._replace(derived=env.engine.replace_candidates(cand.derived, cand.counts))

    def _compute_initial(self, queries: Tensor) -> "_Cand":
        """Candidate generation for a freshly-reset env — the reset twin of
        ``_compute_derived``: components seed their fields (``reset_seed_fields`` — e.g. memory
        seeds the visit-history), then the same ``_derive_step`` → ``candidate_refine`` flow
        (``state=None`` marks the reset context)."""
        env = self.env
        fields: Dict[str, Tensor] = {}
        for c in env.components:
            fields.update(c.reset_seed_fields(env, queries))
        var_idx = torch.full((queries.shape[0],), env.runtime_var_start_index,
                             dtype=torch.long, device=env.device)
        derived, counts, new_var, rule_idx = self._derive_step(
            queries, var_idx, fields, None, excluded=queries[:, 0:1, :])
        cand = _Cand(current_states=queries, derived=derived, counts=counts, next_var=new_var,
                     derived_rule_idx=rule_idx, fields=fields)
        for c in env.components:
            cand = c.candidate_refine(env, cand, None)
        return cand._replace(derived=env.engine.replace_candidates(cand.derived, cand.counts))

    def _derive_step(self, current_states: Tensor, next_var_indices: Tensor,
                     fields: Dict[str, Tensor], state,
                     excluded: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """One full candidate-gen step from ``current_states`` → compacted candidates (static
        shapes): ``_derive_raw`` (engine step + shape) → keep-mask (validity AND every
        component's ``candidate_keep_mask``, e.g. memory not-visited) → ``_finalize`` (compact
        + FALSE fallback). The pipeline primitive shared by ``_compute_derived`` /
        ``_compute_initial`` and the unary advance's re-derive. ``fields`` carries the component
        working state (e.g. the updated visit-history) the keep-masks read. Returns
        ``(derived, counts, next_var, rule_idx)``."""
        env = self.env
        derived_raw, raw_counts, atom_counts, new_var, rule_idx = self._derive_raw(
            current_states, next_var_indices, excluded)
        # Keep-mask [B, S]: a slot is real if within the engine count and atom count ∈ [1, A],
        # AND no component drops it.
        keep = ((env._slot_arange.unsqueeze(0) < raw_counts.unsqueeze(1))
                & (atom_counts > 0) & (atom_counts <= env.padding_atoms))
        for c in env.components:
            m = c.candidate_keep_mask(env, derived_raw, fields, state)
            if m is not None:
                keep = keep & m
        derived, counts, rule_idx = self._finalize(derived_raw, keep, rule_idx)
        return derived, counts, new_var, rule_idx

    def _derive_raw(self, current_states: Tensor, next_var_indices: Tensor,
                    excluded: Optional[Tensor] = None):
        """Engine step → raw (uncompacted) derived states [B, S, A, W] in fixed buffers
        (CUDA-graph safe; engine K×M truncated to S×A), the fresh var allocator, the
        per-candidate ``derived_rule_idx`` [B, S], and the raw / atom counts (for the
        validity keep-mask)."""
        env = self.env
        bsz, S, A, W = env.batch_size, env.padding_states, env.padding_atoms, env.atom_width
        pad = env.padding_idx
        # One backward SLD step: the engine returns up to K successors of width ≤ M each, the
        # per-env successor count, the advanced free-var allocator, and the per-slot rule id.
        derived_raw, raw_counts, new_var, rule_idx_raw = env.engine.derive(
            current_states, next_var_indices, excluded)
        # Atoms-per-successor (non-padding) — used by the validity keep-mask (a real successor
        # has between 1 and A atoms).
        atom_counts_raw = (derived_raw[:, :, :, 0] != pad).sum(dim=2)
        # Copy the engine's K×M output into THIS env's fixed [S, A] buffers (cudagraph needs
        # static shapes); K/M may exceed S/A, so truncate to what fits, padding the rest.
        ksz, msz = min(derived_raw.shape[1], S), min(derived_raw.shape[2], A)
        derived = torch.full((bsz, S, A, W), pad, dtype=torch.long, device=env.device)
        derived[:, :ksz, :msz, :] = derived_raw[:, :ksz, :msz, :]
        atom_counts = torch.zeros((bsz, S), dtype=torch.long, device=env.device)
        atom_counts[:, :ksz] = atom_counts_raw[:, :ksz]
        # Per-slot rule id, padded the same way (0 beyond the engine's K successors).
        rule_idx = torch.zeros((bsz, S), dtype=torch.long, device=env.device)
        rule_idx[:, :ksz] = rule_idx_raw[:, :ksz]

        # PROOF marking (the grounder is vocab-free, so the env marks): a fully-resolved
        # successor is a zero-atom valid slot. When any exists the proof is complete, so collapse
        # the batch to a single TRUE atom at slot 0 (count 1). Dead ends (raw_counts==0) are left
        # for _finalize's FALSE fallback.
        within = env._slot_arange.unsqueeze(0) < raw_counts.unsqueeze(1)     # [B, S]
        proof = (within & (atom_counts == 0)).any(dim=1)                     # [B]
        slot0 = (env._slot_arange == 0).view(1, S, 1, 1)
        derived = torch.where(proof.view(bsz, 1, 1, 1) & slot0,
                              self._true_state.view(1, 1, A, W), derived)
        raw_counts = torch.where(proof, torch.ones_like(raw_counts), raw_counts)
        atom_counts = torch.where(proof.unsqueeze(1),
                                  (env._slot_arange == 0).long().unsqueeze(0), atom_counts)
        return derived, raw_counts, atom_counts, new_var, rule_idx

    def _finalize(self, derived: Tensor, keep: Tensor, rule_idx: Tensor
                  ) -> Tuple[Tensor, Tensor, Tensor]:
        """Turn the raw derive + keep-mask into the final compacted candidate set.

        1. Compact the kept slots to the front in ONE fused pass — ``derived`` and
           ``derived_rule_idx`` move together (cumsum gives each kept slot its target index;
           dropped slots scatter to the last slot then get overwritten by padding).
        2. FALSE fallback: a state with no surviving candidate gets a single FALSE atom (the
           proof fails there). Stop actions are appended later by components
           (``candidate_refine``), not here.
        """
        env = self.env
        bsz, S, A, W = env.batch_size, env.padding_states, env.padding_atoms, env.atom_width
        pad = env.padding_idx
        new_counts = keep.sum(dim=1)

        # 1. Single fused compaction (fresh allocations keep captured cudagraph buffers immutable).
        flat = A * W
        target = torch.cumsum(keep.long(), dim=1) - 1                    # kept slot -> its front index
        target = torch.where(keep, target.clamp(min=0, max=S - 1), env._batch_ones.unsqueeze(1) * (S - 1))
        src = torch.where(keep.unsqueeze(-1), derived.reshape(bsz, S, flat), self._compact_scratch)
        compact = torch.full((bsz, S, flat), pad, dtype=torch.long, device=env.device)
        compact.scatter_(1, target.unsqueeze(-1).expand(bsz, S, flat), src)
        derived = compact.view(bsz, S, A, W)
        packed_rid = torch.zeros((bsz, S), dtype=torch.long, device=env.device)   # rule id rides the same compaction
        packed_rid.scatter_(1, target, torch.where(keep, rule_idx, torch.zeros_like(rule_idx)))
        rule_idx = packed_rid

        # 2. No survivors -> a single FALSE state (the proof fails here).
        empty = new_counts == 0
        derived = torch.where(empty.view(-1, 1, 1, 1),
                              self._false_state.unsqueeze(0).expand(bsz, -1, -1, -1), derived)
        new_counts = torch.where(empty, env._batch_ones, new_counts)
        rule_idx = torch.where(empty.view(-1, 1), torch.zeros_like(rule_idx), rule_idx)

        return derived, new_counts, rule_idx
