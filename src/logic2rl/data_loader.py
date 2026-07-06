"""The data contract for the logic→RL converter (pillar: base).

Two minimal contracts a dataset author implements so ``base.build_env`` can turn
a logic problem into an RL env. Both are intentionally small — extend by adding
fields (see ``kge/data_loader.py``'s ``KGEMaterializedData``), never by requiring
KGE-specific ones here.

  * ``DataLoader``       — a ``@runtime_checkable`` Protocol: the symbolic
    vocabulary + rules, plus ``materialize(im, device)``. ``build_env`` asserts
    ``isinstance(data_loader, DataLoader)``.
  * ``MaterializedData`` — the tensor bundle ``materialize`` returns: KB tensors
    + one :class:`MaterializedSplit` per split. This is all the converter reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import torch


@dataclass
class MaterializedSplit:
    """Generic tensor bundle for one materialized dataset split.

    The shape of ``queries`` is consumer-dependent:

    - SL: ``[N, 3]`` — single ``(r, h, t)`` triples.
    - RL (DpRL): ``[N, L, max_arity + 1]`` — proof-state padded, with
      ``L`` atom slots per query and ``max_arity + 1`` columns
      (predicate id + argument ids).

    ``labels`` and ``depths`` are always ``[N]``. ``depths`` is ``-1`` for
    queries without a depth annotation.
    """

    queries: torch.LongTensor
    labels: torch.LongTensor
    depths: torch.LongTensor

    def __len__(self) -> int:
        return int(self.queries.shape[0])

    def to(self, device) -> "MaterializedSplit":
        """Move all split tensors to ``device`` (returns a new split)."""
        m = lambda t: t.to(device) if t is not None else t
        return MaterializedSplit(queries=m(self.queries), labels=m(self.labels), depths=m(self.depths))


@dataclass(frozen=True)
class MaterializedData:
    """Minimal tensorized logic problem consumed by ``base.build_env``.

    A task that adds metadata (proof depths, a corruption pool, …) subclasses
    this — see ``KGEMaterializedData``. Per-split tensors live in
    :class:`MaterializedSplit` (all three required; pass ``labels`` all-ones for
    positives and ``depths`` all ``-1`` when unknown).
    """
    facts_idx: torch.LongTensor          # KB facts                          [F, atom_width]
    rules_idx: torch.LongTensor          # rules: head at slot 0, body after [R, 1+body_w, atom_width]
    train: MaterializedSplit
    valid: MaterializedSplit
    test: MaterializedSplit

    def to(self, device) -> "MaterializedData":
        """Move all KB + per-split tensors to ``device`` (returns a new bundle).
        Subclasses override to also move their extra tensors."""
        import dataclasses
        return dataclasses.replace(
            self,
            facts_idx=self.facts_idx.to(device), rules_idx=self.rules_idx.to(device),
            train=self.train.to(device), valid=self.valid.to(device), test=self.test.to(device),
        )


@runtime_checkable
class DataLoader(Protocol):
    """A logic problem the converter can index, materialize, and ground.

    ``build_env`` reads the vocabulary + rules to build the IndexManager, then
    calls ``materialize`` to get the KB/query tensors. ``entity2id`` /
    ``relation2id`` are optional id-alignment hooks (read via ``getattr``), not
    part of the required surface.
    """
    constants: Sequence[str]
    predicates: Sequence[str]
    rules: object
    rules_str: object
    max_arity: int
    padding_idx: int

    def materialize(self, im, device) -> MaterializedData:
        ...
