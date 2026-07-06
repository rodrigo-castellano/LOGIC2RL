"""Generic logic-dataset loader (pillar: base).

``LogicDataset`` runs the minimal canonical KG-dataset pipeline — resolve
paths, parse facts / rules / queries, build the alphabetical 1-based
vocabulary, and materialize tensors — over plain ``(predicate, *args)``
tuples. It is deliberately lean and KGE-agnostic: it imports only stdlib +
torch (never ``kge`` / ``algorithm``), carries no policy knobs, and knows
nothing about domains, query filtering, proof depths, or corruption pools.
A consumer subclasses it and overrides the load/materialize hooks to add
those (see ``kge.data_loader.KGEDataHandler``).

On-disk parsing is reduced to exactly three private primitives —
:meth:`_parse_atom`, :meth:`_parse_rule`, :meth:`_read_triples` — that
every loader routes through. Tensor containers (``MaterializedSplit`` /
``MaterializedData``) live in :mod:`base.data_loader`.

Id assignment is 1-based: id ``0`` is the padding sentinel.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from os.path import join
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch

from logic2rl.data_loader import MaterializedData, MaterializedSplit

# ============================================================================
# Internal parsed-rule record
# ============================================================================


class RuleSpec(NamedTuple):
    """Parsed rule with optional metadata (head/body atoms, name, weight)."""

    head: Tuple[str, ...]
    body: List[Tuple[str, ...]]
    name: Optional[str] = None
    weight: float = 1.0


# ============================================================================
# Generic logic dataset
# ============================================================================


@dataclass
class LogicDataset:
    """Lean generic KG dataset loader (subclass to add consumer-specific extras)."""

    # vocabulary (alphabetical, 1-based unless padding_idx=None)
    entity2id: Dict[str, int] = field(default_factory=dict)
    relation2id: Dict[str, int] = field(default_factory=dict)
    padding_idx: int = 0
    constants: List[str] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)
    # typed string-form data
    facts_str: List[Tuple[str, ...]] = field(default_factory=list)
    rules_str: List[Tuple[Tuple[str, ...], List[Tuple[str, ...]]]] = field(default_factory=list)
    rule_names: List[Optional[str]] = field(default_factory=list)
    rule_weights: List[float] = field(default_factory=list)
    train_queries_str: List[Tuple[str, ...]] = field(default_factory=list)
    valid_queries_str: List[Tuple[str, ...]] = field(default_factory=list)
    test_queries_str: List[Tuple[str, ...]] = field(default_factory=list)
    train_labels: List[int] = field(default_factory=list)
    valid_labels: List[int] = field(default_factory=list)
    test_labels: List[int] = field(default_factory=list)
    # consumer-facing aliases for the typed lists (callers may read
    # .facts / .rules / .{split}_queries directly)
    facts: List[Tuple[str, ...]] = field(default_factory=list)
    rules: List[Tuple[Tuple[str, ...], List[Tuple[str, ...]]]] = field(default_factory=list)
    train_queries: List[Tuple[str, ...]] = field(default_factory=list)
    valid_queries: List[Tuple[str, ...]] = field(default_factory=list)
    test_queries: List[Tuple[str, ...]] = field(default_factory=list)

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        base_path: str = "data",
        train_file: str = "train.txt",
        valid_file: str = "valid.txt",
        test_file: str = "test.txt",
        rules_file: Optional[str] = "rules.txt",
        facts_file: Optional[str] = "facts.txt",
        sort: bool = True,
        seed: int = 0,
    ) -> None:
        # Empty containers first, so a subclass can call ``super().__init__()``
        # for lazy init and still see well-defined fields before loading.
        self.entity2id = {}
        self.relation2id = {}
        self.padding_idx = 0
        self.constants = []
        self.predicates = []
        self.facts_str = []
        self.rules_str = []
        self.rule_names = []
        self.rule_weights = []
        self.train_queries_str = []
        self.valid_queries_str = []
        self.test_queries_str = []
        self.train_labels = []
        self.valid_labels = []
        self.test_labels = []
        self.facts = []
        self.rules = []
        self.train_queries = []
        self.valid_queries = []
        self.test_queries = []
        self._sort = sort
        self._seed = seed
        if dataset_name is not None:
            self.load_dataset(
                dataset_name, base_path,
                train_file=train_file, valid_file=valid_file, test_file=test_file,
                rules_file=rules_file, facts_file=facts_file,
            )

    # ---- orchestrator --------------------------------------------------

    def load_dataset(
        self,
        dataset_name: str,
        base_path: str = "data",
        *,
        train_file: str = "train.txt",
        valid_file: str = "valid.txt",
        test_file: str = "test.txt",
        rules_file: Optional[str] = "rules.txt",
        facts_file: Optional[str] = "facts.txt",
    ) -> None:
        """Standard pipeline: facts → rules → queries → vocabulary.

        Subclasses override the individual hooks (or this orchestrator) to
        add their own steps (domains, query filtering, depth sidecars, …).
        """
        paths = self._resolve_paths(
            dataset_name, base_path, train_file, valid_file, test_file,
            rules_file, facts_file,
        )
        self._paths = paths
        self.load_facts(paths["facts"])
        self.load_rules(paths["rules"])
        self.load_queries(paths)
        self.build_vocabulary(paths)

    @staticmethod
    def _resolve_paths(
        dataset_name: str, base_path: str, train_file: str, valid_file: str,
        test_file: str, rules_file: Optional[str], facts_file: Optional[str],
    ) -> Dict[str, Optional[str]]:
        base = join(base_path, dataset_name)
        return {
            "train": join(base, train_file),
            "valid": join(base, valid_file),
            "test": join(base, test_file),
            "facts": join(base, facts_file) if facts_file else None,
            "rules": join(base, rules_file) if rules_file else None,
        }

    # ---- facts ---------------------------------------------------------

    def load_facts(self, path: Optional[str]) -> None:
        """Parse the fact file → ``facts_str`` (str tuples), sorted if ``sort``."""
        triples = self._read_triples(path) if path else []
        # ``_read_triples`` yields (head, relation, tail); fact tuples are
        # (predicate, head, tail) == (relation, head, tail).
        facts = [(rel, head, tail) for (head, rel, tail) in triples]
        if self._sort:
            facts.sort()
        self.facts_str = list(facts)
        self.facts = list(facts)

    # ---- rules ---------------------------------------------------------

    def load_rules(self, path: Optional[str]) -> None:
        """Parse the rule file → ``rules_str`` + parallel name/weight lists, sorted."""
        specs: List[RuleSpec] = []
        if path and os.path.isfile(path):
            seen_rule = False
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("%"):
                        continue
                    # ``var2domain`` preamble (before the first rule) is metadata.
                    if not seen_rule and stripped.startswith("var2domain"):
                        continue
                    spec = self._parse_rule(stripped)
                    if spec is not None:
                        specs.append(spec)
                        seen_rule = True
        self._store_rule_specs(specs)

    def _store_rule_specs(self, specs: List[RuleSpec]) -> None:
        """Populate rule lists from parsed specs (sort by head if ``sort``)."""
        rules_str = [(s.head, s.body) for s in specs]
        names = [s.name for s in specs]
        weights = [s.weight for s in specs]
        if self._sort and rules_str:
            # Sort by HEAD only; stable sort preserves file order across
            # same-head rules (parity references rely on this).
            order = sorted(range(len(rules_str)), key=lambda i: rules_str[i][0])
            rules_str = [rules_str[i] for i in order]
            names = [names[i] for i in order]
            weights = [weights[i] for i in order]
        self.rules_str = rules_str
        self.rule_names = names
        self.rule_weights = weights
        self.rules = list(rules_str)

    # ---- queries -------------------------------------------------------

    def load_queries(self, paths: Dict[str, Optional[str]]) -> None:
        """Parse per-split query files → atoms + positive labels (NO depth).

        Each line is ``<query>`` (optionally ``<query> <depth>``); only the
        atom is kept here (a trailing integer depth column is dropped).
        Subclasses attach depth.
        """
        for split in ("train", "valid", "test"):
            path = paths[split]
            atoms: List[Tuple[str, ...]] = []
            if path and os.path.isfile(path):
                with open(path) as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped or stripped.startswith("%"):
                            continue
                        # Drop a trailing whitespace-separated integer depth.
                        head, _, tail = stripped.rpartition(" ")
                        query_str = head if (head and tail.lstrip("-").isdigit()) else stripped
                        atom = self._parse_atom(query_str)
                        if atom is not None:
                            atoms.append(atom)
            setattr(self, f"{split}_queries", list(atoms))
            setattr(self, f"{split}_queries_str", list(atoms))
            setattr(self, f"{split}_labels", [1] * len(atoms))

    # ---- vocabulary (encoding, not parsing) ----------------------------

    def build_vocabulary(self, paths: Dict[str, Optional[str]]) -> None:
        """Build entity2id/relation2id (1-based) from facts ∪ valid/test ∪ rules ∪ queries.

        Assigns alphabetical ids from the train + valid + test + facts
        vocabulary, then grows the maps in-place with any unseen names in
        valid/test/facts, and finally extends ``constants`` / ``predicates``
        with rule + query atoms (which don't flow through the triple files).
        """
        train_path, fact_path = paths["train"], paths["facts"]
        valid_path, test_path = paths["valid"], paths["test"]
        offset = 1 if self.padding_idx == 0 else 0

        # 1. Alphabetical ids from train ∪ valid ∪ test ∪ facts.
        all_triples: List[Tuple[str, str, str]] = []
        for p in (train_path, valid_path, test_path, fact_path):
            if p and os.path.isfile(p):
                all_triples.extend(self._read_triples(p))
        ents = sorted({h for h, _r, _t in all_triples} | {t for _h, _r, t in all_triples})
        rels = sorted({r for _h, r, _t in all_triples})
        ent2id = {e: i + offset for i, e in enumerate(ents)}
        rel2id = {r: i + offset for i, r in enumerate(rels)}
        self.entity2id = ent2id
        self.relation2id = rel2id

        # 2. constants/predicates = vocab ∪ rule atoms ∪ query atoms.
        constants = set(ent2id.keys())
        predicates = set(rel2id.keys())
        for (head_pred, *_h), body in self.rules_str:
            predicates.add(head_pred)
            for body_atom in body:
                predicates.add(body_atom[0])
        for atom in (self.train_queries_str + self.valid_queries_str + self.test_queries_str):
            predicates.add(atom[0])
            constants.update(atom[1:])
        self.constants = sorted(constants)
        self.predicates = sorted(predicates)

    # ---- materialize ---------------------------------------------------

    def materialize(
        self,
        *,
        entity_id_fn: Optional[Any] = None,
        relation_id_fn: Optional[Any] = None,
        include_rules: bool = True,
        device: Optional[Any] = None,
    ) -> MaterializedData:
        """Tensorize facts + per-split queries → a :class:`MaterializedData`.

        Pure: builds tensors from the loaded string-form lists and RETURNS a
        bundle (does not mutate ``self``). ``entity_id_fn`` / ``relation_id_fn``
        default to ``self.entity2id`` / ``self.relation2id`` lookups.
        ``include_rules`` controls whether generic rule body/head tensors are
        built (consumers whose rules carry variables pass ``include_rules=False``
        and tensorize rules themselves). Per-split ``depths`` default to all
        ``-1``; a subclass that tracks depths exposes ``{split}_depths`` and
        this reads them.
        """
        device = device or torch.device("cpu")
        ent_fn = entity_id_fn or (lambda e: self.entity2id[e])
        rel_fn = relation_id_fn or (lambda r: self.relation2id[r])

        def _row(atom: Tuple[str, ...]) -> Tuple[int, int, int]:
            # (predicate, head, tail) → (relation_id, head_id, tail_id)
            return (rel_fn(atom[0]), ent_fn(atom[1]), ent_fn(atom[2]))

        if self.facts_str:
            facts_t = torch.tensor([_row(a) for a in self.facts_str], dtype=torch.long, device=device)
        else:
            facts_t = torch.empty((0, 3), dtype=torch.long, device=device)

        # Combined rules tensor: head at slot 0, body atoms after → [R, 1+max_body, 3]. The
        # grounder splits heads/bodies and derives rule lengths from it (Fork 3).
        if self.rules_str and include_rules:
            max_body = max(1, max((len(b) for _, b in self.rules_str), default=0))
            R = len(self.rules_str)
            rules_t = torch.zeros((R, 1 + max_body, 3), dtype=torch.long, device=device)
            for r, (head, body) in enumerate(self.rules_str):
                rules_t[r, 0] = torch.tensor(_row(head), dtype=torch.long)
                for b, atom in enumerate(body):
                    rules_t[r, 1 + b] = torch.tensor(_row(atom), dtype=torch.long)
        else:
            rules_t = torch.empty((0, 2, 3), dtype=torch.long, device=device)

        def _split(name: str) -> MaterializedSplit:
            queries = getattr(self, f"{name}_queries_str")
            labels = getattr(self, f"{name}_labels")
            depths = getattr(self, f"{name}_depths", None)
            if depths is None:
                depths = [-1] * len(queries)
            if queries:
                q_t = torch.tensor([_row(a) for a in queries], dtype=torch.long, device=device)
            else:
                q_t = torch.empty((0, 3), dtype=torch.long, device=device)
            return MaterializedSplit(
                queries=q_t,
                labels=torch.as_tensor(labels, dtype=torch.long, device=device),
                depths=torch.as_tensor(depths, dtype=torch.long, device=device),
            )

        return MaterializedData(
            facts_idx=facts_t, rules_idx=rules_t,
            train=_split("train"), valid=_split("valid"), test=_split("test"),
        )

    # ====================================================================
    # The three private parsers (every loader routes through these)
    # ====================================================================

    @classmethod
    def _parse_atom(cls, atom_str: str) -> Optional[Tuple[str, ...]]:
        """``"p(a, b)"`` → ``("p", "a", "b")``; tolerant of a trailing ``.``.

        The single atom primitive: facts, queries, and rule atoms all route
        through it. Each token is normalized (surrounding quotes / whitespace
        stripped). Returns ``None`` for a non-atom line.
        """
        def norm(token: str) -> str:
            return token.strip().strip("'\"").strip()

        raw = atom_str.strip()
        if raw.endswith("."):
            raw = raw[:-1].rstrip()
        if "(" not in raw or not raw.endswith(")"):
            return None
        predicate, remainder = raw.split("(", 1)
        args = [norm(a) for a in remainder[:-1].split(",") if a.strip()]
        return (norm(predicate), *args)

    @classmethod
    def _parse_rule(cls, line: str) -> Optional[RuleSpec]:
        """Parse one rule line → :class:`RuleSpec`.

        Handles ``head :- body``, ``body -> head``, and an optional
        ``rN:weight:`` prefix. The paren-aware body split is nested here.
        Atoms route through :meth:`_parse_atom`. Returns ``None`` for a
        malformed line.
        """
        def split_body(body_str: str) -> List[str]:
            # Split on top-level commas (not inside parens).
            atoms: List[str] = []
            depth = 0
            current: List[str] = []
            for ch in body_str:
                if ch == "(":
                    depth += 1
                    current.append(ch)
                elif ch == ")":
                    depth -= 1
                    current.append(ch)
                elif ch == "," and depth == 0:
                    atom = "".join(current).strip()
                    if atom:
                        atoms.append(atom)
                    current = []
                else:
                    current.append(ch)
            tail = "".join(current).strip()
            if tail:
                atoms.append(tail)
            return atoms

        raw = line.strip()
        if raw.endswith("."):
            raw = raw[:-1]
        name: Optional[str] = None
        weight: float = 1.0
        if raw.startswith("r"):
            first_colon = raw.find(":")
            first_paren = raw.find("(")
            if first_colon != -1 and (first_paren == -1 or first_colon < first_paren):
                parts = raw.split(":", 2)
                if len(parts) >= 3 and parts[0][1:].isdigit():
                    try:
                        weight = float(parts[1])
                        name = parts[0]
                        raw = parts[2].strip()
                    except ValueError:
                        pass
        if ":-" in raw:
            head_str, body_str = raw.split(":-", 1)
        elif "->" in raw:
            body_str, head_str = raw.split("->", 1)
        else:
            head = cls._parse_atom(raw)
            return RuleSpec(head=head, body=[], name=name, weight=weight) if head is not None else None
        head = cls._parse_atom(head_str)
        if head is None:
            return None
        body: List[Tuple[str, ...]] = []
        for atom_str in split_body(body_str):
            atom = cls._parse_atom(atom_str)
            if atom is None:
                return None
            body.append(atom)
        return RuleSpec(head=head, body=body, name=name, weight=weight)

    @classmethod
    def _read_triples(cls, path: str) -> List[Tuple[str, str, str]]:
        """Read a triple file → list of (head, relation, tail) string tuples.

        Detects Prolog vs TSV/CSV from the first non-empty line (inlined).
        The Prolog branch routes through :meth:`_parse_atom` (reordering
        ``(pred, h, t)`` → ``(h, pred, t)``); TSV/CSV split on the delimiter.
        Skips blank / ``#`` / ``%`` comment lines and non-binary atoms.
        """
        if not path or not os.path.isfile(path):
            return []
        # Format detection (first non-empty line).
        fmt = "unknown"
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                sample = line.strip()
                if not sample:
                    continue
                if "(" in sample and ")" in sample:
                    fmt = "prolog"
                elif sample.count("\t") >= 2:
                    fmt = "tsv"
                elif sample.count(",") >= 2:
                    fmt = "csv"
                break

        out: List[Tuple[str, str, str]] = []
        if fmt == "prolog":
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw or raw.startswith("#") or raw.startswith("%"):
                        continue
                    # Truncate at the closing paren so a trailing token (e.g. a
                    # depth column: ``aunt(1,2) 2``) doesn't defeat the atom
                    # parser, then route through _parse_atom.
                    close = raw.find(")")
                    if close == -1:
                        continue
                    atom = cls._parse_atom(raw[: close + 1])
                    if atom is None or len(atom) != 3:
                        continue
                    pred, h, t = atom
                    out.append((h, pred, t))  # (head, relation, tail)
            return out
        delimiter = "\t" if fmt == "tsv" else ","
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle, delimiter=delimiter):
                cleaned = [c.strip().strip("'\"").strip() for c in row if c.strip()]
                if len(cleaned) < 3:
                    continue
                out.append((cleaned[0], cleaned[1], cleaned[2]))
        return out
