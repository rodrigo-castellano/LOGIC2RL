"""
Learnable Embedding Layers for Logical Reasoning (pillar: base).

Provides embedding tables (constants, predicates) and a generic embedder that
composes them with pluggable atom- and state-level encoders. The composer is
domain-agnostic: the atom encoder is supplied by an injected ``atom_factory``
(default: the generic ``base.nn.atom_embedders`` factory) and every atom encoder
declares its output width via ``.out_dim``, so the composer sizes the downstream
state encoder without knowing any KGE model. The KGE application wires its scoring
atom encoders in via ``kge.nn.embeddings.EmbedderLearnable`` (a thin subclass that
injects the KGE atom factory).

Tensor shapes
-------------
predicate_emb:  [B, G, A, 1, embed_dim]
constant_embs:  [B, G, A, 2, embed_dim]
atom_output:    [B, G, A, embed_dim]
state_output:   [B, G, embed_dim]
"""
from __future__ import annotations

import inspect
import logging
import math  # noqa: F401 – used by atom/state sub-modules via shared scope
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401
from torch import Tensor

logger = logging.getLogger(__name__)


# Shared low-level blocks are generic and live in base/nn.
from logic2rl.nn._blocks import _append_custom_loss

# Atom- and state-embedder factories are both generic and live in base/nn. The KGE
# application injects its own scoring-aware atom factory via ``atom_factory``.
from logic2rl.nn.atom_embedders import Emb_Atom_Factory
from logic2rl.nn.state_embedders import Emb_State_Factory

# ---------------------------------------------------------------------------
# Embedding tables
# ---------------------------------------------------------------------------

class ConstantEmbeddings(nn.Module):
    """Module to handle constant embeddings per domain."""

    def __init__(self, num_constants: int, embedding_dim: int, regularization: float = 0.0, device: str = "cpu"):
        super().__init__()
        # num_constants is the count, but indices go from 0 to num_constants
        # so we need num_constants+1 entries
        self.embedder = nn.Embedding(num_constants + 1, embedding_dim, padding_idx=0)
        self.regularization = regularization
        self.device = device

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedder(indices)
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
        return embeddings

    def add_loss(self, loss: torch.Tensor) -> None:
        _append_custom_loss(self, loss)


class PredicateEmbeddings(nn.Module):
    """Module to handle predicate embeddings."""

    def __init__(self, num_predicates: int, embedding_dim: int, regularization: float = 0.0, device: str = "cpu"):
        super().__init__()
        # num_predicates is the count, but indices go from 0 to num_predicates
        # so we need num_predicates+1 entries
        self.embedder = nn.Embedding(num_predicates + 1, embedding_dim, padding_idx=0)
        self.regularization = regularization
        self.device = device

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedder(indices)
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
        return embeddings

    def add_loss(self, loss: torch.Tensor) -> None:
        _append_custom_loss(self, loss)


# ---------------------------------------------------------------------------
# Main embedder
# ---------------------------------------------------------------------------

class EmbedderLearnable(nn.Module):
    def __init__(
        self,
        n_constants: int = 0,
        n_predicates: int = 0,
        n_vars: int = 0,
        max_arity: int = 2,
        padding_atoms: int = 10,
        atom_embedder: str = "sum",
        state_embedder: str = "sum",
        constant_embedding_size: int = 64,
        predicate_embedding_size: int = 64,
        atom_embedding_size: int = 64,
        kge_regularization: float = 0.0,
        kge_dropout_rate: float = 0.0,
        atom_factory: Callable[..., nn.Module] = Emb_Atom_Factory,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()

        # Store device for later use
        self.device = device
        self.embedding_dim = atom_embedding_size
        self.atom_embedding_size = atom_embedding_size
        self.predicate_embedding_size = int(predicate_embedding_size)
        self.n_constants = int(n_constants)
        self.n_vars = int(n_vars)

        # Build vocabulary size directly from counts; ConstantEmbeddings adds
        # its own padding slot, so we do not offset here. Runtime variables share
        # the constant table: each var id (> constant_no) indexes its own row.
        total_constant_vocab = n_constants + n_vars
        self.max_constant_vocab_index = int(total_constant_vocab)

        # Initialize embedder
        self.constant_embedder = ConstantEmbeddings(
            num_constants=total_constant_vocab,
            embedding_dim=constant_embedding_size,
            regularization=kge_regularization,
            device=device,
        )
        self.constant_embedding_size = int(constant_embedding_size)

        self.predicate_embedder = PredicateEmbeddings(
            num_predicates=n_predicates,
            embedding_dim=predicate_embedding_size,
            regularization=kge_regularization,
            device=device,
        )

        # The atom encoder (KGE scorer or generic set encoder) is built by the
        # injected factory, which owns any model-specific size validation and
        # reports the atom feature width via ``.out_dim``. The composer stays
        # agnostic to which atom model was chosen.
        self.atom_embedder = atom_factory(
            name=atom_embedder,
            atom_embedding_size=atom_embedding_size,
            predicate_embedding_size=predicate_embedding_size,
            constant_embedding_size=constant_embedding_size,
            max_arity=max_arity,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate,
            device=device,
        )
        state_input_dim = self.atom_embedder.out_dim

        self.state_embedder = Emb_State_Factory(
            name=state_embedder,
            embedding_dim=state_input_dim,
            padding_atoms=padding_atoms,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate,
            device=device,
            **kwargs,
        )
        self._state_accepts_sub_indices = "sub_indices" in inspect.signature(self.state_embedder.forward).parameters

        # Expose state output dimensionality used by policy modules.
        if state_embedder.casefold() == "concat":
            self.embed_dim = state_input_dim * padding_atoms
        else:
            self.embed_dim = state_input_dim
        self.embedding_dim = self.embed_dim

        # Move entire module to device
        self.to(device)

    def get_embeddings_batch(self, sub_indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for a batch of sub-indices.

        Args:
            sub_indices: [B, G, A, 3] where B=n_envs, G=goals (search width), A=atoms, 3=(pred, arg1, arg2)

        Returns:
            [B, G, embed_dim]
        """
        predicate_indices = sub_indices[..., 0].unsqueeze(-1)
        constant_embeddings = self._encode_argument_embeddings(sub_indices)
        predicate_embeddings = self.predicate_embedder(predicate_indices)

        atom_embeddings = self.atom_embedder(predicate_embeddings, constant_embeddings)
        if self._state_accepts_sub_indices:
            state_embeddings = self.state_embedder(atom_embeddings, sub_indices=sub_indices)
        else:
            state_embeddings = self.state_embedder(atom_embeddings)

        return state_embeddings

    def _encode_argument_embeddings(self, sub_indices: torch.Tensor) -> torch.Tensor:
        """Embed argument ids. Runtime variables (id > constant_no) share the constant
        table — each var id indexes its own row (``unique_id`` encoding)."""
        args = sub_indices[..., 1:]
        lower_ok = (args >= 0).all()
        upper_ok = (args <= self.max_constant_vocab_index).all()
        torch._assert_async(lower_ok, "Negative constant/variable ID in embedding input")
        torch._assert_async(upper_ok, "Embedding input ID exceeded constant+variable vocabulary")
        return self.constant_embedder(args)

    def forward(self, sub_indices: torch.Tensor) -> torch.Tensor:
        return self.get_embeddings_batch(sub_indices)
