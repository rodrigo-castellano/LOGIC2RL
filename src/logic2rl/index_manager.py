"""Index Manager — vocabulary index spaces and string→index conversions.

Maps constants, predicates, and template variables to integer indices.
Layout: constants [1..n], template vars allocated lazily above them, padding 0.
The runtime-variable id-space (start/end) and the embedder var-table size are
owned by the engine (``base.unification.SLD``), not here.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

# -----------------------------
# Type aliases (indices only)
# -----------------------------
# Atom layout: [predicate_id, arg1_id, …, arg_{max_arity}_id]  (width W = max_arity + 1)
Tensor = torch.Tensor
LongTensor = torch.LongTensor


class IndexManager:
    """
    Vocabulary index spaces and string→index conversions.

    Index spaces:
    - Constants: [1 .. n]
    - Template variables (from rules only): [n+1 .. n+Vt]
    - Padding: 0

    The runtime-variable id-space (reserved pool, start/end) and the embedder's
    branching-derived var-table size live on the engine (``SLD``), not here.

    Strings are *only* used via debug helpers.
    """

    # -----------------------------
    # Construction
    # -----------------------------
    def __init__(
        self,
        constants: Iterable[str],
        predicates: Iterable[str],
        max_arity: int = 2,
        padding_atoms: int = 10,
        device: Optional[torch.device] = None,
        rules: Optional[List] = None,
        rules_str: Optional[List] = None,
        padding_idx: int = 0,
        extra_special_predicates: Sequence[str] = (),
    ) -> None:
        """
        Initialize IndexManager with vocabulary.

        Args:
            constants: Iterable of constant strings
            predicates: Iterable of predicate strings
            max_arity: Maximum arity of predicates
            padding_atoms: Maximum atoms per state (for padding)
            device: Target device for tensors
            rules: Optional list of Rule objects
            rules_str: Optional list of (head, body) string tuples
            extra_special_predicates: special predicate names to allocate beyond the
                always-present ``True``/``False`` (e.g. ``Endf`` for the end-proof action,
                or any user-defined marker). Appended last, so they never shift the regular
                or True/False ids. ``build_env`` receives these via its ``special_predicates`` parameter.

        The id-assignment order is decided by ``_order_vocab`` (default: sorted).
        ``KGEIndexManager`` overrides it to align with a pretrained KGE checkpoint.
        """
        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.max_arity: int = max_arity
        self.padding_idx: int = padding_idx
        self.padding_atoms: int = padding_atoms
        self.extra_special_predicates: Tuple[str, ...] = tuple(extra_special_predicates)

        # Build the vocabulary index spaces (str<->idx maps, sizes, special-pred indices).
        self.build_idx(constants, predicates)

        self.rules = rules if rules is not None else []
        self.rules_str: List = rules_str if rules_str is not None else []

    # -----------------------------
    # Index-space construction
    # -----------------------------
    def _order_vocab(self, constants: Iterable[str], predicates: Iterable[str]):
        """Decide the id-assignment order of the vocab. Default: sorted (reproducible).
        ``KGEIndexManager`` overrides this to align with a pretrained checkpoint's id space."""
        return sorted(set(constants)), sorted(set(predicates))

    def build_idx(self, constants: Iterable[str], predicates: Iterable[str]) -> None:
        """Build the constant/predicate index spaces and the special-pred indices.

        Layout: constants ``[1..constant_no]`` (with the +1 padding shift when
        ``padding_idx == 0``), template vars allocated lazily by ``_ensure_template_var``,
        padding at 0. Sets ``constant_str2idx``/``predicate_str2idx``/``idx2*`` maps,
        ``constant_no``/``predicate_no``, and ``true/false/endf_pred_idx``. The vocab order
        comes from ``_order_vocab`` (sorted by default).
        """
        const_list, pred_list = self._order_vocab(constants, predicates)

        # IMPORTANT: Match str_index_manager.py behavior - add special predicates AFTER regular ones
        # This ensures canonical ordering keys match between str and batched environments
        # First assign indices to regular predicates
        regular_pred_list = pred_list
        special_pred_list = []

        # Special predicates are appended AFTER the regular ones (canonical ordering): the
        # always-present proof markers True/False, then any config-supplied extras (Endf for
        # the end-proof action, or any user-defined marker). Appended last, so adding/dropping
        # an extra never shifts the regular or True/False ids.
        for sp in ('True', 'False', *self.extra_special_predicates):
            if sp not in regular_pred_list and sp not in special_pred_list:
                special_pred_list.append(sp)

        # Combine: regular predicates first (sorted), then special predicates
        pred_list = regular_pred_list + special_pred_list

        # String <-> index maps. ``padding_idx`` drives the offset:
        # ``0`` reserves slot 0 (1-based real ids); any other value (or
        # ``None`` semantically — defaulted to 0 here) collapses to a
        # dense 0-based layout. The ``idx2*`` lists put a sentinel string
        # at the reserved slot so length matches max-id+1.
        offset = 1 if self.padding_idx == 0 else 0
        self.constant_str2idx: Dict[str, int] = {s: i + offset for i, s in enumerate(const_list)}
        self.predicate_str2idx: Dict[str, int] = {s: i + offset for i, s in enumerate(pred_list)}
        if offset == 1:
            self.idx2constant: List[str] = ["<PAD>"] + const_list
            self.idx2predicate: List[str] = ["<PAD>"] + pred_list
        else:
            self.idx2constant = list(const_list)
            self.idx2predicate = list(pred_list)

        # Template vars appear only in rules; we'll allocate lazily when rules are materialized
        self.template_var_str2idx: Dict[str, int] = {}
        self.idx2template_var: List[str] = ["<PAD>"]

        # Unified term map used in one-shot conversions (strings -> indices)
        self.unified_term_map: Dict[str, int] = dict(self.constant_str2idx)  # start with constants

        # Sizes
        self.constant_no: int = len(self.constant_str2idx)
        self.predicate_no: int = len(self.predicate_str2idx)
        self.template_variable_no: int = 0

        # All special-predicate ids by name (True/False + the configured extras). Convenience
        # attrs expose the well-known ones the engine/components consume (None when absent).
        self.special_pred_ids: Dict[str, int] = {
            sp: self.predicate_str2idx[sp]
            for sp in ('True', 'False', *self.extra_special_predicates)
            if sp in self.predicate_str2idx
        }
        self.true_pred_idx: Optional[int] = self.special_pred_ids.get('True')
        self.false_pred_idx: Optional[int] = self.special_pred_ids.get('False')
        self.endf_pred_idx: Optional[int] = self.special_pred_ids.get('Endf')

    # -----------------------------
    # Vocabulary growth for rule variables
    # -----------------------------
    def _ensure_template_var(self, var_name: str) -> int:
        """Ensure template variable exists and return its index.
        
        Note: Template variables are assigned indices in the range 
        [constant_no + 1, constant_no + template_variable_no]. However,
        we do NOT shift the runtime_var_start_index when adding template
        variables - this matches SB3 behavior where variable_start_index
        is always constant_no + 1.
        """
        idx = self.template_var_str2idx.get(var_name)
        if idx is not None:
            return idx
        # allocate next template var
        idx = self.constant_no + self.template_variable_no + 1
        self.template_variable_no += 1
        self.template_var_str2idx[var_name] = idx
        self.idx2template_var.append(var_name)
        # update unified map for one-shot conversions
        self.unified_term_map[var_name] = idx
        # Template vars index within [constant_no+1, constant_no+template_variable_no];
        # the engine owns the runtime-var id-space (start/end), so nothing here shifts it.
        return idx

    # -----------------------------
    # Materializers (strings -> indices)
    # -----------------------------
    def term_to_index(self, token: str) -> int:
        """Return index for a constant or a template variable (rules only)."""
        if token in self.unified_term_map:
            return self.unified_term_map[token]
        # treat as template variable if unseen (likely from rules)
        return self._ensure_template_var(token)

    @property
    def predicate_vocab(self) -> Dict[int, str]:
        """Predicate index → name mapping (for metrics logging)."""
        return {i: name for i, name in enumerate(self.idx2predicate)}

    def states_to_tensor(
        self,
        states: Iterable[Iterable[Tuple[str, ...]]],
        max_atoms: Optional[int] = None,
    ) -> LongTensor:
        """The one string→id structure converter: a batch of symbolic states →
        padded ``[N, M, W]`` long ids (W = max_arity + 1: predicate id + max_arity term ids).

        ``states`` is an iterable of states; each state an iterable of ``(predicate, *args)``
        string atoms — a predicate followed by 0..``max_arity`` term tokens (so binary KGE keeps
        passing ``(predicate, arg1, arg2)``, ternary tasks pass ``(predicate, a, b, c)``). Terms
        route through :meth:`term_to_index` (constants are pre-indexed; rule variables are
        allocated lazily, in first-seen order). Atoms shorter than ``max_arity`` pad their
        trailing arg columns with ``padding_idx``. ``M = max_atoms`` (raises if any state is
        longer) or the longest state when ``None``; unused atom slots are padding (``0``).

        Special cases: a single atom → ``states_to_tensor([[atom]])[0, 0]``; a single
        state → ``states_to_tensor([atoms])[0]``. Per-state allocation is column-major
        (predicates, then all arg1s, then all arg2s, …) so the lazy-var first-seen order is
        stable across the conversion (and identical to the old binary path at max_arity=2).
        """
        W = self.max_arity + 1
        states = [list(s) for s in states]
        lens = [len(s) for s in states]
        longest = max(lens) if lens else 0
        M = int(max_atoms) if max_atoms is not None else longest
        if max_atoms is not None and longest > M:
            raise ValueError(f"state length {longest} > max_atoms={M}")
        out = torch.zeros((len(states), M, W), dtype=torch.long)
        for i, atoms in enumerate(states):
            k = len(atoms)
            if k == 0:
                continue
            for atom in atoms:
                if atom[0] not in self.predicate_str2idx:
                    raise ValueError(f"unknown predicate {atom[0]!r}; declared predicates: {sorted(self.predicate_str2idx)}")
                if len(atom) - 1 > self.max_arity:
                    raise ValueError(f"atom {atom!r} arity {len(atom) - 1} > max_arity={self.max_arity}")
            out[i, :k, 0] = torch.as_tensor([self.predicate_str2idx[a[0]] for a in atoms], dtype=torch.long)
            # Arg columns 1..max_arity, column-major (all arg-j across atoms before arg-(j+1)) so the
            # lazy template-var first-seen order is stable; atoms with arity < max_arity pad the rest.
            for j in range(1, W):
                out[i, :k, j] = torch.as_tensor(
                    [self.term_to_index(a[j]) if j < len(a) else self.padding_idx for a in atoms],
                    dtype=torch.long)
        return out

