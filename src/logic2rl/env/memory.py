"""Visit-memory component (pillar: base).

Dedups proof-search states so the agent doesn't re-explore. Each env keeps a fixed-size
ring of ``int64`` state hashes (``history_hashes`` / ``history_count`` in the composed
EnvState); a derived state is pruned when its hash matches a visited one. Cheap and
CUDA-graph-stable, with a (tiny) false-positive collision rate.

Expressed as an ``EnvComponent`` so the base env carries no memory fields or logic when
it's omitted: it owns the visit-history state fields, seeds/updates the ring each step,
and supplies the hash-path keep-mask to the candidate-gen orchestrator (fused into its
single compaction — no second pass). Enabled by the builder's component set (``base.builder.make_base_components``) when
``memory_pruning``; omit it and the env has no history fields and pays no hashing cost.

Owns its one derived dimension, ``max_history_size`` (= ``max_depth + 1``, one slot per
step plus the initial query), like the unification component owns ``padding_states``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from logic2rl.env.component import EnvComponent, FieldSpec

Tensor = torch.Tensor

_HASH_PACK_OFFSET = 1001   # constant_no offset for the state-hash pack base (fallback)


class VisitMemoryComponent(EnvComponent):
    """State hashing + visited-state pruning, as a pluggable component.

    Caches its hash constants off the env in ``setup`` and publishes itself as
    ``env.memory`` so the unary re-derive (``UnaryAdvanceComponent``) can reuse
    ``_update_history``."""

    name = "visit_memory"

    # ---- eager, once, at construction (NOT traced) ----
    def setup(self, env: Any) -> None:
        # Publish the hashing on env.memory so the unary re-derive reuses _update_history;
        # absence of the component ⇒ no env.memory.
        env.memory = self
        self.env = env
        self.engine = env.engine
        self.device = env.device
        self.padding_idx = env.padding_idx
        self.max_history_size = env.max_depth + 1
        self._hash_mix_const = 0x9E3779B97F4A7C15
        self._hash_mask63 = (1 << 63) - 1
        self._hash_pack_base = getattr(
            self.engine, "pack_base", self.env.constant_no + _HASH_PACK_OFFSET)

    # ---- Axis A: recurrent state fields ----
    def declare_state_fields(self, env: Any) -> Tuple[FieldSpec, ...]:
        H = env.max_depth + 1
        return (
            FieldSpec("history_hashes", lambda e, B: torch.zeros((B, H), dtype=torch.int64, device=e.device)),
            FieldSpec("history_count", lambda e, B: torch.zeros((B,), dtype=torch.long, device=e.device)),
        )

    # ---- candidate-generation hooks ----
    def reset_seed_fields(self, env: Any, queries: Tensor) -> Dict[str, Tensor]:
        """Reset: seed the visit-history ring from the initial queries."""
        h, c = self.init_history(queries)
        return {"history_hashes": h, "history_count": c}

    def step_update_fields(self, env: Any, current: Tensor, fields: Dict[str, Tensor],
                      mask: Tensor) -> Dict[str, Tensor]:
        """Append ``current``'s hash to the ring for the envs in ``mask`` (active at step,
        advancing at unary)."""
        h, c = self._update_history(current, fields["history_hashes"], fields["history_count"], mask)
        return {"history_hashes": h, "history_count": c}

    def candidate_keep_mask(self, env: Any, derived_raw: Tensor, fields: Dict[str, Tensor],
                            state) -> "Tensor | None":
        """Hash-path keep-mask [B, S]: True where a candidate's state hash is NOT in the visit
        history (drop already-visited). Fused into the candidate-gen single compaction.
        Presence of the component IS the enable switch (wired iff ``config.memory_pruning``)."""
        history_hashes, history_count = fields["history_hashes"], fields["history_count"]
        bsz, S = derived_raw.shape[0], derived_raw.shape[1]
        hist = history_hashes.shape[1]
        hashes = self._compute_hash(derived_raw)                                    # [B, S]
        h_valid = torch.arange(hist, device=env.device).view(1, 1, hist) < history_count.view(bsz, 1, 1)
        seen = ((hashes.unsqueeze(-1) == history_hashes.unsqueeze(1).expand(-1, S, -1)) & h_valid).any(dim=-1)
        return ~seen

    # ---- hashing -------------------------------------------------------------
    def init_history(self, queries: Tensor) -> Tuple[Tensor, Tensor]:
        """Seed per-env visit history from the initial queries → (hashes [B, H], count [B])."""
        B, H = queries.shape[0], self.max_history_size
        h_hashes = torch.zeros((B, H), dtype=torch.int64, device=self.device)
        h_hashes[:, 0] = self._compute_hash(queries)
        h_count = torch.ones(B, dtype=torch.long, device=self.device)
        return h_hashes, h_count

    def _compute_hash(self, states: Tensor) -> Tensor:
        """Order-invariant per-state hash. ``[B, A, W] -> [B]`` (or ``[B, K, A, W] -> [B, K]``).

        Terminal/padding atoms are excluded so a state hashes the same regardless of
        how it terminated; the per-atom packs are mixed and summed (commutative)."""
        if states.dim() == 4:
            bsz, ksz, atoms, dims = states.shape
            return self._compute_hash(states.view(bsz * ksz, atoms, dims)).view(bsz, ksz)
        s = states.long()
        preds = s[:, :, 0]
        valid = preds != self.padding_idx
        # Terminal atoms (true/false/endf) hash out so a state hashes the same regardless of how
        # it terminated; the canonical terminal predicate set is owned by the env.
        for p in self.env.terminal_pred_ids:
            valid = valid & (preds != p)
        base = self._hash_pack_base
        # Horner fold over all W atom columns (pred + args): identical to the binary
        # ((p*base + a1)*base + a2) at W=3. base^W < 2^63 keeps the fold injective (small-vocab
        # N-ary tasks; the binary KGE path never approaches it).
        packed = s[:, :, 0]
        for j in range(1, s.shape[2]):
            packed = packed * base + s[:, :, j]
        packed = packed & self._hash_mask63
        mixed = (packed * self._hash_mix_const) & self._hash_mask63
        return torch.where(valid, mixed, torch.zeros_like(mixed)).sum(dim=1) & self._hash_mask63

    def _update_history(self, current_states, history_hashes, history_count, active_mask):
        """Append the current state's hash to each active env's ring → (hashes, count)."""
        write_pos = history_count.clamp(max=self.max_history_size - 1)
        new_hash = self._compute_hash(current_states)
        new_history = history_hashes.scatter(
            1, write_pos.unsqueeze(1),
            torch.where(active_mask.unsqueeze(1), new_hash.unsqueeze(1),
                        history_hashes.gather(1, write_pos.unsqueeze(1))),
        )
        new_count = torch.where(
            active_mask, (history_count + 1).clamp(max=self.max_history_size), history_count)
        return new_history, new_count
