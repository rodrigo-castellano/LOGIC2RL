"""Generic atom-level set encoders: predicate + constants → atom vector (pillar: base).

Each class consumes predicate_emb ``[B, G, A, 1, E]`` and constant_embs
``[B, G, A, 2, E]`` and returns atom embeddings ``[B, G, A, E]``. These are the
atom-level twins of the state-level set encoders in ``state_embedders`` — pure,
domain-agnostic composition/pooling over the ``{predicate, args}`` token set, built
on the CUDA-graph-safe blocks in ``_blocks``. KGE *scoring* atom embedders
(TransE/ComplEx/RotatE/DistMult) are the KGE application's concern and live in
``kge.nn.atom_embedders``, whose factory delegates the generic names below to
``Emb_Atom_Factory`` here.

Imports nothing from ``kge`` or ``algorithm``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from logic2rl.nn._blocks import (
    _append_custom_loss,
    _ManualGRUCell,
    _ManualMultiheadAttention,
)

# ---------------------------------------------------------------------------
# Simple composition atom embedders
# ---------------------------------------------------------------------------

class SumAtom(nn.Module):
    """For atom or state embeddings, simply sum the embeddings of the constants&predicates or atoms."""
    def __init__(self, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"):
        super(SumAtom, self).__init__()
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        self.to(device)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:

        predicate_emb = predicate_emb.squeeze(-2)  # Remove unnecessary dimension if present
        if self.dropout_rate > 0:
            predicate_emb = self.dropout(predicate_emb)
            constant_embs = self.dropout(constant_embs)  # Apply dropout to all constants

        embeddings = predicate_emb - constant_embs.sum(dim=-2)

        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))

        return embeddings

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


class ConcatAtoms(nn.Module):
    """Concat the predicate and constant embeddings."""
    def __init__(self,
                atom_embedding_dim: int,
                max_arity: int,
                dropout_rate: float=0.0,
                regularization: float=0.0,
                device="cpu"):
        super(ConcatAtoms, self).__init__()
        self.max_arity = max_arity
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        if regularization > 0:
            self.regularization = regularization
        self.device = device

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        predicate_emb = predicate_emb.squeeze(-2)  # Remove unnecessary dimension if present
        if self.dropout_rate > 0:
            predicate_emb = self.dropout(predicate_emb)
            constant_embs = self.dropout(constant_embs)

        # Determine the number of constants in the constant_embs
        num_constants = constant_embs.size(-2)
        # Pad the embeddings with zeros to reach 10 constants
        if num_constants < self.max_arity:
            padding_tensor = torch.zeros(*constant_embs.shape[:-2], self.max_arity - num_constants, constant_embs.size(-1), device=self.device)
            constant_embs = torch.cat([constant_embs, padding_tensor], dim=-2)
        # Concatenate constant embeddings along the last dimension
        constant_embs = constant_embs.view(*constant_embs.shape[:-2], -1)
        embeddings = torch.cat([predicate_emb, constant_embs], dim=-1)

        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))

        return embeddings

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


# ---------------------------------------------------------------------------
# Attention-based atom embedders
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Attention-based layer for computing atom embeddings using dot-product attention.

    Expected input shapes:
      - predicate_emb: [B, n_states, n_atoms, 1, embed_dim]
      - constant_embs: [B, n_states, n_atoms, 2, embed_dim]

    Returns:
      - output: [B, n_states, n_atoms, embed_dim]
    """
    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu"
    ):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.regularization = regularization
        self.device = device

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicate_emb: Tensor of shape [B, n_states, n_atoms, 1, embed_dim]
            constant_embs: Tensor of shape [B, n_states, n_atoms, 2, embed_dim]
        Returns:
            output: Tensor of shape [B, n_states, n_atoms, embed_dim]
        """
        # Apply dropout
        predicate_emb = self.dropout(predicate_emb)
        constant_embs = self.dropout(constant_embs)

        # Remove the singleton arity dimension from predicate_emb
        # New shape: [B, n_states, n_atoms, embed_dim]
        predicate_emb = predicate_emb.squeeze(3)

        # Compute dot-product attention scores.
        # Expand predicate_emb to [B, n_states, n_atoms, 1, embed_dim] (if not already)
        # then multiply element-wise with constant_embs and sum over embed_dim.
        scores = (predicate_emb.unsqueeze(3) * constant_embs).sum(dim=-1)  # [B, n_states, n_atoms, 2]

        # Compute attention weights with softmax along the arity dimension (dim=-1).
        attn_weights = torch.softmax(scores, dim=-1)  # [B, n_states, n_atoms, 2]

        # Weighted sum of the constant embeddings using the attention weights.
        # Multiply weights (expanded to have embed_dim) and sum over the arity dimension.
        output = (attn_weights.unsqueeze(-1) * constant_embs).sum(dim=3)  # [B, n_states, n_atoms, embed_dim]

        if self.regularization > 0:
            self.add_loss(self.regularization * output.norm(p=2))

        return output

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


class MultiHeadAttention(nn.Module):
    """Multi-head attention for computing atom embeddings with scaled dot-product attention.

    Expected input shapes:
      - predicate_emb: [B, n_states, n_atoms, 1, embed_dim]
      - constant_embs: [B, n_states, n_atoms, 2, embed_dim]

    Returns:
      - output: [B, n_states, n_atoms, embed_dim]
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu"
    ):
        super(MultiHeadAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.regularization = regularization
        self.device = device

        # Linear projections for query, key and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final projection after concatenating heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicate_emb: Tensor of shape [B, n_states, n_atoms, 1, embed_dim]
            constant_embs: Tensor of shape [B, n_states, n_atoms, 2, embed_dim]
        Returns:
            output: Tensor of shape [B, n_states, n_atoms, embed_dim]
        """
        # Apply dropout
        predicate_emb = self.dropout(predicate_emb)
        constant_embs = self.dropout(constant_embs)

        # Remove the singleton dimension from predicate_emb:
        # Now shape: [B, n_states, n_atoms, embed_dim]
        predicate_emb = predicate_emb.squeeze(3)

        # Linear projections
        # For predicate: [B, n_states, n_atoms, embed_dim]
        # For constants: [B, n_states, n_atoms, 2, embed_dim]
        Q = self.q_proj(predicate_emb)  # [B, n_states, n_atoms, embed_dim]
        K = self.k_proj(constant_embs)    # [B, n_states, n_atoms, 2, embed_dim]
        V = self.v_proj(constant_embs)    # [B, n_states, n_atoms, 2, embed_dim]

        # Reshape for multi-head attention:
        # Q: reshape to [B, n_states, n_atoms, num_heads, head_dim] then add a singleton constant dim
        Q = Q.view(*Q.shape[:-1], self.num_heads, self.head_dim).unsqueeze(3)  # [B, n_states, n_atoms, 1, num_heads, head_dim]

        # K and V: reshape to [B, n_states, n_atoms, 2, num_heads, head_dim]
        K = K.view(*K.shape[:-1], self.num_heads, self.head_dim)
        V = V.view(*V.shape[:-1], self.num_heads, self.head_dim)

        # Transpose to bring num_heads before the constant dimension:
        # Q becomes [B, n_states, n_atoms, num_heads, 1, head_dim]
        Q = Q.transpose(3, 4)
        # K and V become [B, n_states, n_atoms, num_heads, 2, head_dim]
        K = K.transpose(3, 4)
        V = V.transpose(3, 4)

        # Scaled dot-product attention:
        # Compute scores: Q @ K^T along last two dimensions.
        # K.transpose(-2, -1) changes K to [B, n_states, n_atoms, num_heads, head_dim, 2]
        # Resulting scores shape: [B, n_states, n_atoms, num_heads, 1, 2]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores.squeeze(-2)  # Now [B, n_states, n_atoms, num_heads, 2]
        scores = scores / math.sqrt(self.head_dim)

        # Compute attention weights with softmax over the constant dimension (last dim)
        attn_weights = F.softmax(scores, dim=-1)  # [B, n_states, n_atoms, num_heads, 2]

        # Expand weights for weighted sum: [B, n_states, n_atoms, num_heads, 2, 1]
        attn_weights = attn_weights.unsqueeze(-1)
        # Weighted sum of V: multiply and sum over constant dimension -> [B, n_states, n_atoms, num_heads, head_dim]
        weighted_sum = (attn_weights * V).sum(dim=-2)

        # Concatenate heads: reshape from [B, n_states, n_atoms, num_heads, head_dim] to [B, n_states, n_atoms, embed_dim]
        concat = weighted_sum.view(*weighted_sum.shape[:-2], self.embed_dim)

        # Final linear projection
        output = self.out_proj(concat)
        if self.regularization > 0:
            self.add_loss(self.regularization * output.norm(p=2))

        return output

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


class Transformer(nn.Module):
    """Transformer-based layer for computing atom embeddings.

    Expected input shapes:
      - predicate_emb: [B, n_states, n_atoms, 1, embed_dim]
      - constant_embs: [B, n_states, n_atoms, 2, embed_dim]

    Returns:
      - output: [B, n_states, n_atoms, embed_dim]
    """
    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu",
        num_heads: int = 1
    ):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.regularization = regularization
        self.device = device

        # Fall back to valid num_heads
        if embed_dim % num_heads != 0:
            for h in (num_heads, 4, 2, 1):
                if embed_dim % h == 0:
                    num_heads = h
                    break
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Manual cross-attention (CUDA graph safe)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicate_emb: Tensor of shape [B, n_states, n_atoms, 1, embed_dim]
            constant_embs: Tensor of shape [B, n_states, n_atoms, 2, embed_dim]
        Returns:
            output: Tensor of shape [B, n_states, n_atoms, embed_dim]
        """
        predicate_emb = self.dropout(predicate_emb)
        constant_embs = self.dropout(constant_embs)

        # [B, n_states, n_atoms, embed_dim]
        predicate_emb = predicate_emb.squeeze(3)
        B, n_states, n_atoms, embed_dim = predicate_emb.shape
        flat_batch = B * n_states * n_atoms
        H, D = self.num_heads, self.head_dim

        # query from predicate: [flat_batch, 1, embed_dim]
        query = predicate_emb.reshape(flat_batch, 1, embed_dim)
        # key/value from constants: [flat_batch, 2, embed_dim]
        kv = constant_embs.reshape(flat_batch, 2, embed_dim)

        q = self.q_proj(query).reshape(flat_batch, 1, H, D).transpose(1, 2)  # [N, H, 1, D]
        k = self.k_proj(kv).reshape(flat_batch, 2, H, D).transpose(1, 2)     # [N, H, 2, D]
        v = self.v_proj(kv).reshape(flat_batch, 2, H, D).transpose(1, 2)     # [N, H, 2, D]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [N, H, 1, 2]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(flat_batch, embed_dim)  # [N, E]
        output = self.out_proj(out).reshape(B, n_states, n_atoms, embed_dim)

        if self.regularization > 0:
            self.add_loss(self.regularization * output.norm(p=2))

        return output

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


# ---------------------------------------------------------------------------
# RNN atom embedder
# ---------------------------------------------------------------------------

class RNN(nn.Module):
    """RNN-based layer for computing atom embeddings with specific input shapes.

    Uses _ManualGRUCell for torch.compile(fullgraph=True) compatibility.

    Expected input shapes:
      - predicate_emb: [B, n_states, n_atoms, 1, embed_dim]
      - constant_embs: [B, n_states, n_atoms, 2, embed_dim]
    Returns:
      - output: [B, n_states, n_atoms, embed_dim]
    """
    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu"
    ):
        super(RNN, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout_rate)
        self.regularization = regularization
        self.device = device

        self.gru_cell = _ManualGRUCell(input_size=embed_dim, hidden_size=embed_dim)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicate_emb: Tensor of shape [B, n_states, n_atoms, 1, embed_dim]
            constant_embs: Tensor of shape [B, n_states, n_atoms, 2, embed_dim]
        Returns:
            output: Tensor of shape [B, n_states, n_atoms, embed_dim]
        """
        predicate_emb = self.dropout(predicate_emb)
        constant_embs = self.dropout(constant_embs)

        # [B, n_states, n_atoms, embed_dim]
        predicate_emb = predicate_emb.squeeze(-2)
        B, n_states, n_atoms, embed_dim = predicate_emb.shape
        N = B * n_states * n_atoms

        # Initial hidden state from predicate embedding
        h = predicate_emb.reshape(N, embed_dim)

        # Process constant embeddings as 2-step sequence
        # constant_embs: [B, G, A, 2, D] -> [2, N, D]
        seq = constant_embs.permute(3, 0, 1, 2, 4).reshape(2, N, embed_dim)

        # Two GRU steps (fixed-length sequence of 2)
        h = self.gru_cell(seq[0], h)
        h = self.gru_cell(seq[1], h)
        hidden = h  # [N, embed_dim]
        output = hidden.squeeze(0)  # shape: [B*n_states*n_atoms, embed_dim]

        # Reshape output back to [B, n_states, n_atoms, embed_dim]
        output = output.reshape(B, n_states, n_atoms, embed_dim)

        # (Optional) Add regularization loss.
        if self.regularization > 0:
            self.add_loss(self.regularization * output.norm(p=2))

        return output

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


# ---------------------------------------------------------------------------
# Advanced atom embedders
# ---------------------------------------------------------------------------

class CrossAttentionAtom(nn.Module):
    """Cross-attention atom embedder: predicate attends to [arg1, arg2].

    Uses _ManualMultiheadAttention (CUDA graph safe).
    Architecture: pred is query, [head, tail] are key/value → residual + LayerNorm.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4,
                 dropout_rate: float = 0.0, regularization: float = 0.0, device="cpu"):
        super().__init__()
        self.attn = _ManualMultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.regularization = regularization
        self.to(device)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        """predicate_emb: [..., 1, E], constant_embs: [..., 2, E] → [..., E]"""
        leading = predicate_emb.shape[:-2]  # e.g. (B, G, A)
        E = predicate_emb.shape[-1]
        # Flatten leading dims for attention: [N, 1, E] query, [N, 2, E] key/value
        q = predicate_emb.reshape(-1, 1, E)       # [N, 1, E]
        kv = constant_embs.reshape(-1, 2, E)       # [N, 2, E]
        attn_out = self.attn(q, kv, kv)            # [N, 1, E]
        # Residual + LayerNorm
        out = self.norm(q + self.dropout(attn_out))  # [N, 1, E]
        embeddings = out.reshape(*leading, E)
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
        return embeddings

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


class MLPAtom(nn.Module):
    """MLP atom embedder: FC(3E→2E) → ReLU → FC(2E→E) → LayerNorm."""
    def __init__(self, embed_dim: int, dropout_rate: float = 0.0,
                 regularization: float = 0.0, device="cpu"):
        super().__init__()
        self.fc1 = nn.Linear(3 * embed_dim, 2 * embed_dim)
        self.fc2 = nn.Linear(2 * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.regularization = regularization
        self.to(device)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        """predicate_emb: [..., 1, E], constant_embs: [..., 2, E] → [..., E]"""
        pred = predicate_emb.squeeze(-2)         # [..., E]
        head = constant_embs[..., 0, :]          # [..., E]
        tail = constant_embs[..., 1, :]          # [..., E]
        x = torch.cat([pred, head, tail], dim=-1)  # [..., 3E]
        x = self.dropout(F.relu(self.fc1(x)))      # [..., 2E]
        embeddings = self.norm(self.fc2(x))         # [..., E]
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
        return embeddings

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


class SetTransformerAtom(nn.Module):
    """Set Transformer atom embedder: SAB over [pred, arg1, arg2] + PMA pooling.

    Treats the 3 components (predicate, head, tail) as a set of 3 tokens
    and applies self-attention + pooling to produce a single embedding.
    Uses _ManualMultiheadAttention (CUDA graph safe).
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, num_sab_layers: int = 1,
                 dropout_rate: float = 0.0, regularization: float = 0.0, device="cpu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.regularization = regularization

        # SAB layers
        self.sab_layers = nn.ModuleList()
        self.sab_norms = nn.ModuleList()
        for _ in range(num_sab_layers):
            self.sab_layers.append(_ManualMultiheadAttention(embed_dim, num_heads))
            self.sab_norms.append(nn.LayerNorm(embed_dim))

        # PMA: pool 3 tokens → 1 output
        self.pma_seed = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pma_attn = _ManualMultiheadAttention(embed_dim, num_heads)
        self.pma_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.to(device)

    def forward(self, predicate_emb: torch.Tensor, constant_embs: torch.Tensor) -> torch.Tensor:
        """predicate_emb: [..., 1, E], constant_embs: [..., 2, E] → [..., E]"""
        leading = predicate_emb.shape[:-2]  # e.g. (B, G, A)
        E = predicate_emb.shape[-1]
        # Build 3-token set: [pred, head, tail] → [N, 3, E]
        pred = predicate_emb.reshape(-1, 1, E)      # [N, 1, E]
        args = constant_embs.reshape(-1, 2, E)       # [N, 2, E]
        x = torch.cat([pred, args], dim=1)            # [N, 3, E]

        # SAB layers (residual self-attention)
        for sab, norm in zip(self.sab_layers, self.sab_norms):
            residual = x
            x = norm(residual + self.dropout(sab(x, x, x)))

        # PMA: pool into single vector
        N = x.shape[0]
        seed = self.pma_seed.expand(N, 1, E)
        pooled = self.pma_norm(seed + self.dropout(self.pma_attn(seed, x, x)))

        embeddings = pooled.squeeze(1).reshape(*leading, E)
        if self.regularization > 0:
            self.add_loss(self.regularization * embeddings.norm(p=2))
        return embeddings

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def Emb_Atom_Factory(name: str = 'sum',
            atom_embedding_size: int = -1,
            predicate_embedding_size: int | None = None,
            constant_embedding_size: int | None = None,
            max_arity: int = 2,
            regularization: float = 0.0,
            dropout_rate: float = 0.0,
            device="cpu") -> nn.Module:
    """Build a generic atom set-encoder. Returns a module with ``.out_dim`` (the atom
    feature width fed to the downstream state encoder). Raises on unknown names — the
    KGE factory in ``kge.nn.atom_embedders`` handles the scoring names and delegates
    the rest here."""
    lname = name.casefold()
    E = atom_embedding_size
    if predicate_embedding_size is None:
        predicate_embedding_size = E
    if constant_embedding_size is None:
        constant_embedding_size = E

    if lname == 'sum':
        module = SumAtom(dropout_rate=dropout_rate, regularization=regularization, device=device)
        out_dim = E
    elif lname == 'concat':
        module = ConcatAtoms(E, max_arity, dropout_rate=dropout_rate, regularization=regularization, device=device)
        out_dim = predicate_embedding_size + max_arity * constant_embedding_size
    elif lname == 'transformer':
        module = Transformer(embed_dim=E, dropout_rate=dropout_rate, regularization=regularization, device=device)
        out_dim = E
    elif lname == 'rnn':
        module = RNN(embed_dim=E, dropout_rate=dropout_rate, regularization=regularization, device=device)
        out_dim = E
    elif lname == 'attention':
        module = MultiHeadAttention(embed_dim=E, dropout_rate=dropout_rate, regularization=regularization, device=device)
        out_dim = E
    elif lname == 'cross_attention':
        module = CrossAttentionAtom(embed_dim=E, dropout_rate=dropout_rate, regularization=regularization, device=device)
        out_dim = E
    elif lname == 'mlp':
        module = MLPAtom(embed_dim=E, dropout_rate=dropout_rate, regularization=regularization, device=device)
        out_dim = E
    elif lname == 'set_transformer_atom':
        module = SetTransformerAtom(embed_dim=E, dropout_rate=dropout_rate, regularization=regularization, device=device)
        out_dim = E
    else:
        raise ValueError(f"Unknown generic atom embedder: {name}")

    module.out_dim = out_dim
    return module
