"""State-level embedders: atom embeddings → state vector.

Each class consumes atom_embeddings [B, n_states, n_atoms, embed_dim]
and returns state embeddings [B, n_states, embed_dim].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ._blocks import (
        _ISAB,
        _MAB,
        _append_custom_loss,
        _ManualGRUCell,
        _ManualMultiheadAttention,
        _mask_from_atoms,
        _safe_mean_pool,
        _TransformerBlock,
    )
except ImportError:
    from _blocks import (
        _ISAB,
        _MAB,
        _append_custom_loss,
        _ManualGRUCell,
        _ManualMultiheadAttention,
        _mask_from_atoms,
        _safe_mean_pool,
        _TransformerBlock,
    )


# ---------------------------------------------------------------------------
# Simple pooling state embedders
# ---------------------------------------------------------------------------

class SumState(nn.Module):
    def __init__(self, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"):
        super(SumState, self).__init__()
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        self.to(device)

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        if self.dropout_rate > 0:
            atom_embeddings = self.dropout(atom_embeddings)
        state_embeddings = atom_embeddings.sum(dim=-2)
        if self.regularization > 0:
            self.add_loss(self.regularization * state_embeddings.norm(p=2))
        return state_embeddings

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


class MeanState(nn.Module):
    def __init__(self, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"):
        super(MeanState, self).__init__()
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        if self.dropout_rate > 0:
            atom_embeddings = self.dropout(atom_embeddings)
        valid = (atom_embeddings.abs().sum(dim=-1, keepdim=True) > 0).float()
        state_embeddings = (atom_embeddings * valid).sum(dim=-2) / valid.sum(dim=-2).clamp(min=1.0)
        if self.regularization > 0:
            self.add_loss(self.regularization * state_embeddings.norm(p=2))
        return state_embeddings

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


class MaxState(nn.Module):
    """Max pooling over atoms to get state embedding.

    Takes the element-wise maximum across all atoms, preserving the most
    prominent features from each atom position.
    """
    def __init__(self, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"):
        super(MaxState, self).__init__()
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        self.to(device)

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        if self.dropout_rate > 0:
            atom_embeddings = self.dropout(atom_embeddings)
        valid = atom_embeddings.abs().sum(dim=-1, keepdim=True) > 0
        masked_atoms = atom_embeddings.masked_fill(~valid, torch.finfo(atom_embeddings.dtype).min)
        state_embeddings = masked_atoms.max(dim=-2)[0]
        all_padded = ~valid.squeeze(-1).any(dim=-1)
        state_embeddings = torch.where(
            all_padded.unsqueeze(-1), torch.zeros_like(state_embeddings), state_embeddings
        )
        if self.regularization > 0:
            self.add_loss(self.regularization * state_embeddings.norm(p=2))
        return state_embeddings

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


class ConcatStates(nn.Module):
    """Concat the atom embeddings."""
    def __init__(self,
                padding_atoms: int,
                dropout_rate: float=0.0,
                regularization: float=0.0,
                device="cpu"):
        super(ConcatStates, self).__init__()
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        if regularization > 0:
            self.regularization = regularization
        self.device = device

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        if self.dropout_rate > 0:
            atom_embeddings = self.dropout(atom_embeddings)
        if self.regularization > 0:
            self.add_loss(self.regularization * atom_embeddings.norm(p=2))

        # Concatenate constant embeddings along the last dimension
        atom_embeddings = atom_embeddings.view(*atom_embeddings.shape[:-2], -1)
        return atom_embeddings

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


# ---------------------------------------------------------------------------
# Attention-based state embedders
# ---------------------------------------------------------------------------

class SelfAttentionState(nn.Module):
    """Self-attention pooling over atoms to get state embedding.

    Uses a learned query vector to attend over atom embeddings,
    producing a weighted combination that preserves structural information.
    """
    def __init__(self, embed_dim: int, dropout_rate: float=0.0, regularization: float=0.0, device="cpu"):
        super(SelfAttentionState, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        self.device = device

        # Learned query vector for attention
        self.query = nn.Parameter(torch.randn(embed_dim))
        # Scale factor for dot-product attention
        self.scale = math.sqrt(embed_dim)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_embeddings: [B, n_states, n_atoms, embed_dim]
        Returns:
            state_embeddings: [B, n_states, embed_dim]
        """
        if self.dropout_rate > 0:
            atom_embeddings = self.dropout(atom_embeddings)

        # Compute attention scores: [B, G, A]
        scores = torch.einsum('bsae,e->bsa', atom_embeddings, self.query) / self.scale
        valid = atom_embeddings.abs().sum(dim=-1) > 0
        B, n_states, n_atoms = scores.shape
        scores_f = scores.reshape(B * n_states, n_atoms)
        valid_f = valid.reshape(B * n_states, n_atoms)
        all_padded = ~valid_f.any(dim=1, keepdim=True)  # [BS, 1]
        first_col = (torch.arange(n_atoms, device=valid_f.device).unsqueeze(0) == 0)  # [1, A]
        valid_f = valid_f | (all_padded & first_col)
        scores_f = scores_f.masked_fill(~valid_f, float("-inf"))
        attn_weights = F.softmax(scores_f, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        state_embeddings = torch.bmm(
            attn_weights.unsqueeze(1),
            atom_embeddings.reshape(B * n_states, n_atoms, self.embed_dim),
        ).squeeze(1).reshape(B, n_states, self.embed_dim)

        if self.regularization > 0:
            self.add_loss(self.regularization * state_embeddings.norm(p=2))

        return state_embeddings

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


class AttentionState(nn.Module):
    """
    Computes state embeddings using multi-head self-attention over atom embeddings.
    Assumes atoms within a state form a set/sequence to attend over.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout_rate: float = 0.0,
                 regularization: float = 0.0,
                 device="cpu"):
        """
        Args:
            embed_dim: The embedding dimension of atoms and the final state.
            num_heads: Number of attention heads. Must divide embed_dim.
            dropout_rate: Dropout probability for attention and final output.
            regularization: Coefficient for L2 regularization loss on the output.
            device: Device for computation.
        """
        super(AttentionState, self).__init__()

        # Fall back to valid num_heads if embed_dim not divisible
        if embed_dim % num_heads != 0:
            for h in (num_heads, 4, 2, 1):
                if embed_dim % h == 0:
                    num_heads = h
                    break

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        self.device = device

        # Manual multi-head attention (CUDA graph safe, avoids nn.MultiheadAttention)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Layer Normalization for stability
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        # FeedForward network after attention pooling
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        if dropout_rate > 0:
            self.output_dropout = nn.Dropout(p=dropout_rate)


    def add_loss(self, loss_value):
        _append_custom_loss(self, loss_value)


    def forward(self, atom_embeddings: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            atom_embeddings: Tensor of shape [B, n_states, n_atoms, embed_dim].
                             Represents the embeddings of atoms within each state.
            padding_mask: Optional tensor of shape [B, n_states, n_atoms] where True indicates
                          a padded atom that should be ignored by attention.
                          (Requires adjustment below if used).

        Returns:
            output: Tensor of shape [B, n_states, embed_dim] representing state embeddings.
        """
        B, n_states, n_atoms, embed_dim = atom_embeddings.shape

        # Reshape for MultiheadAttention: Combine B and n_states into the batch dim
        # Input shape: [B * n_states, n_atoms, embed_dim]
        flat_atoms = atom_embeddings.reshape(B * n_states, n_atoms, embed_dim)

        # --- Handle Padding Mask (if provided) ---
        # MultiheadAttention expects key_padding_mask of shape [Batch, Seq_len]
        # where True indicates positions to be *ignored*.
        key_padding_mask = None
        if padding_mask is not None:
            if padding_mask.shape != (B, n_states, n_atoms):
                 raise ValueError("padding_mask shape must be [B, n_states, n_atoms]")
            key_padding_mask = padding_mask.reshape(B * n_states, n_atoms)

        # Apply LayerNorm before attention
        normed_atoms = self.layer_norm1(flat_atoms)

        # Manual self-attention (CUDA graph safe)
        N, L, E = normed_atoms.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(normed_atoms).reshape(N, L, H, D).transpose(1, 2)  # [N, H, L, D]
        k = self.k_proj(normed_atoms).reshape(N, L, H, D).transpose(1, 2)
        v = self.v_proj(normed_atoms).reshape(N, L, H, D).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [N, H, L, L]
        if key_padding_mask is not None:
            mask = key_padding_mask.bool().unsqueeze(1).unsqueeze(2)  # [N, 1, 1, L]
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(N, L, E)
        attn_output = self.out_proj(attn_output)

        # Add residual connection (skip connection)
        res_atoms = flat_atoms + attn_output # Or normed_atoms + attn_output

        # --- Pooling Strategy ---
        # Average pooling across the atoms dimension (n_atoms)
        # Apply mask *before* pooling if available to ignore padding tokens in mean calculation
        if key_padding_mask is not None:
             # Expand mask for broadcasting: [B*n_states, n_atoms, 1]
             mask_expanded = (~key_padding_mask).unsqueeze(-1).float()
             # Element-wise multiply features by mask (zeros out padded positions)
             masked_res_atoms = res_atoms * mask_expanded
             # Summing masked features and dividing by the count of non-padded items
             pooled_state = masked_res_atoms.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
        else:
             # Simple mean pooling if no mask provided
             pooled_state = res_atoms.mean(dim=1) # Shape: [B * n_states, embed_dim]

        # Apply LayerNorm and FeedForward network
        normed_pooled = self.layer_norm2(pooled_state)
        ffn_output = self.ffn(normed_pooled)

        # Add second residual connection
        final_pooled_state = pooled_state + ffn_output # Or normed_pooled + ffn_output

        # Reshape back to original batch/state structure
        # Shape: [B, n_states, embed_dim]
        state_emb = final_pooled_state.reshape(B, n_states, embed_dim)

        # Apply final dropout
        if self.dropout_rate > 0:
            state_emb = self.output_dropout(state_emb)

        # Apply regularization loss (using the custom method)
        if self.regularization > 0:
            # Calculate L2 norm across the embedding dimension for each state
            reg_loss = self.regularization * torch.linalg.vector_norm(state_emb, ord=2, dim=-1).mean()
            self.add_loss(reg_loss)

        return state_emb


# ---------------------------------------------------------------------------
# Sequential state embedders
# ---------------------------------------------------------------------------

class RNNState(nn.Module):
    def __init__(self, embed_dim: int, dropout_rate: float = 0.0, regularization: float = 0.0, device="cpu"):
        """
        Args:
            embed_dim: The embedding dimension.
            dropout_rate: Dropout probability.
            regularization: Coefficient for L2 regularization loss.
            device: Device for computation.
        """
        super(RNNState, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.gru_cell = _ManualGRUCell(input_size=embed_dim, hidden_size=embed_dim)
        self.device = device

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_embeddings: Tensor of shape [B, n_states, n_atoms, embed_dim]
        Returns:
            output: Tensor of shape [B, n_states, embed_dim]
        """

        if self.dropout_rate > 0:
            atom_embeddings = self.dropout(atom_embeddings)

        B, n_states, n_atoms, embed_dim = atom_embeddings.shape
        # Flatten batch and state dimensions: [B*n_states, n_atoms, embed_dim]
        flat_atoms = atom_embeddings.reshape(B * n_states, n_atoms, embed_dim)

        # Mask padded atoms to zero so GRU cell ignores them.
        valid_mask = flat_atoms.abs().sum(dim=-1) > 0  # [B*n_states, n_atoms]
        flat_atoms = flat_atoms * valid_mask.unsqueeze(-1).float()

        # Manual GRU loop over the atom sequence dimension.
        N = flat_atoms.size(0)
        h = torch.zeros(N, embed_dim, device=flat_atoms.device, dtype=flat_atoms.dtype)
        for t in range(n_atoms):
            x_t = flat_atoms[:, t, :]         # [N, D]
            h_new = self.gru_cell(x_t, h)     # [N, D]
            # Only update hidden state for valid (non-padded) positions.
            gate = valid_mask[:, t].unsqueeze(-1).float()  # [N, 1]
            h = gate * h_new + (1 - gate) * h

        # h is now the last valid hidden state per sequence.
        state_emb = h.reshape(B, n_states, embed_dim)

        if self.regularization > 0:
            self.add_loss(self.regularization * state_emb.norm(p=2))

        return state_emb

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


# ---------------------------------------------------------------------------
# Transformer-based state embedders
# ---------------------------------------------------------------------------

class TransformerState(nn.Module):
    """SOTA-style Transformer encoder over atom sets.

    Uses PreNorm encoder blocks (GELU FFN) and a learned CLS token for
    permutation-invariant state pooling. Padding atoms are masked.
    Uses manual attention (CUDA graph safe).
    """

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu",
        num_heads: int = 4,
        num_layers: int = 2,
        ff_mult: int = 4,
        use_cls_token: bool = True,
    ):
        super(TransformerState, self).__init__()
        self.embed_dim = embed_dim
        self.regularization = regularization
        self.use_cls_token = use_cls_token
        self.device = device

        # Choose a valid number of heads for arbitrary embed_dim.
        if embed_dim % num_heads != 0:
            for h in (8, 4, 2, 1):
                if embed_dim % h == 0:
                    num_heads = h
                    break

        ff_dim = max(embed_dim, ff_mult * embed_dim)
        self.blocks = nn.ModuleList([
            _TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) if use_cls_token else None
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        # atom_embeddings: [B, G, A, D]
        x = self.dropout(atom_embeddings)
        B, G, A, D = x.shape
        x = x.reshape(B * G, A, D)  # [BS, A, D]

        # True where atom is padded
        pad_mask = x.abs().sum(dim=-1) == 0  # [BS, A]
        # Avoid all-masked rows without Python data-dependent branching.
        all_padded = pad_mask.all(dim=1, keepdim=True)  # [BS, 1]
        first_col = (torch.arange(A, device=pad_mask.device).unsqueeze(0) == 0)  # [1, A]
        pad_mask = pad_mask & ~(all_padded & first_col)

        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)  # [BS, 1, D]
            x = torch.cat([cls, x], dim=1)  # [BS, 1+A, D]
            cls_mask = torch.zeros((pad_mask.size(0), 1), dtype=torch.bool, device=pad_mask.device)
            key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)
        else:
            key_padding_mask = pad_mask

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        if self.use_cls_token:
            state_emb = x[:, 0, :]  # [BS, D]
        else:
            valid = (~pad_mask).float().unsqueeze(-1)
            atoms = x[:, :A, :]
            state_emb = (atoms * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        state_emb = self.out_norm(state_emb).reshape(B, G, D)
        if self.regularization > 0:
            self.add_loss(self.regularization * state_emb.norm(p=2))
        return state_emb

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


class SetTransformerState(nn.Module):
    """Set Transformer: SAB (self-attention between atoms) + PMA (pooling).

    Architecture: atoms → SAB×L → PMA → state_embedding
    - SAB: Self-Attention Block (atoms attend to each other via self-attention)
    - PMA: Pooling by Multi-head Attention (1 learned seed → output)

    Uses manual attention (matmul+softmax) for CUDA graph compatibility.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, num_sab_layers: int = 2,
                 dropout_rate: float = 0.0, regularization: float = 0.0, device="cpu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.regularization = regularization

        # SAB layers: self-attention between atoms
        self.sab_layers = nn.ModuleList()
        self.sab_norms = nn.ModuleList()
        for _ in range(num_sab_layers):
            self.sab_layers.append(_ManualMultiheadAttention(embed_dim, num_heads))
            self.sab_norms.append(nn.LayerNorm(embed_dim))

        # PMA: learned seed vector pools atom set into single vector
        self.pma_seed = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pma_attn = _ManualMultiheadAttention(embed_dim, num_heads)
        self.pma_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        """atom_embeddings: [B, G, A, E] → state_embeddings: [B, G, E]"""
        B, G, A, E = atom_embeddings.shape
        x = atom_embeddings.reshape(B * G, A, E)  # [BS, A, E]
        pad_mask = x.abs().sum(dim=-1) == 0  # [BS, A]
        all_padded = pad_mask.all(dim=1, keepdim=True)  # [BS, 1]
        first_col = (torch.arange(A, device=pad_mask.device).unsqueeze(0) == 0)  # [1, A]
        pad_mask = pad_mask & ~(all_padded & first_col)

        # SAB layers (residual self-attention)
        for sab, norm in zip(self.sab_layers, self.sab_norms):
            residual = x
            x = norm(residual + self.dropout(sab(x, x, x, key_padding_mask=pad_mask)))

        # PMA: pool into single vector
        seed = self.pma_seed.expand(B * G, 1, E)
        pooled = self.pma_norm(
            seed + self.dropout(self.pma_attn(seed, x, x, key_padding_mask=pad_mask))
        )

        state_emb = pooled.squeeze(1).reshape(B, G, E)  # [B, G, E]

        if self.regularization > 0:
            self.add_loss(self.regularization * state_emb.norm(p=2))

        return state_emb

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


# ---------------------------------------------------------------------------
# Literature-inspired state encoders (from embeddings_literature.py)
# ---------------------------------------------------------------------------

class DeepSetsState(nn.Module):
    """DeepSets-style permutation-invariant state encoder."""

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        hidden_dim = max(embed_dim, 2 * embed_dim)
        self.phi = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.regularization = regularization
        self.device = device

    def forward(self, atom_embeddings: torch.Tensor, sub_indices: torch.Tensor | None = None) -> torch.Tensor:
        del sub_indices
        atom_embeddings = self.dropout(atom_embeddings)
        mask = (atom_embeddings.abs().sum(dim=-1, keepdim=True) > 0).float()
        phi_out = self.phi(atom_embeddings) * mask
        pooled = phi_out.sum(dim=-2)
        state_embeddings = self.rho(pooled)
        if self.regularization > 0:
            self.add_loss(self.regularization * state_embeddings.norm(p=2))
        return state_embeddings

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


class PhiSumState(nn.Module):
    """Per-atom MLP (φ) followed by sum aggregation, no post-sum ρ.

    Like DeepSets but without the ρ network — the downstream body serves as ρ.
    Preserves the magnitude properties of sum while allowing learned per-atom features.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        hidden_dim = 2 * embed_dim
        self.phi = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.regularization = regularization
        self.device = device

    def forward(self, atom_embeddings: torch.Tensor, sub_indices: torch.Tensor | None = None) -> torch.Tensor:
        del sub_indices
        atom_embeddings = self.dropout(atom_embeddings)
        mask = (atom_embeddings.abs().sum(dim=-1, keepdim=True) > 0).float()
        phi_out = self.phi(atom_embeddings) * mask
        state_embeddings = phi_out.sum(dim=-2)
        if self.regularization > 0:
            self.add_loss(self.regularization * state_embeddings.norm(p=2))
        return state_embeddings

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


class ISABSetTransformerState(nn.Module):
    """Set Transformer with ISAB blocks + PMA pooling."""

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu",
        num_heads: int = 4,
        num_layers: int = 2,
        num_inducing: int = 16,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.blocks = nn.ModuleList(
            [
                _ISAB(
                    dim=embed_dim,
                    num_heads=num_heads,
                    num_inducing=num_inducing,
                    dropout=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        self.seed = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pma = _MAB(dim=embed_dim, num_heads=num_heads, dropout=dropout_rate)
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.regularization = regularization
        self.device = device

    def forward(self, atom_embeddings: torch.Tensor, sub_indices: torch.Tensor | None = None) -> torch.Tensor:
        del sub_indices
        atom_embeddings = self.dropout(atom_embeddings)
        bsz, n_states, n_atoms, dim = atom_embeddings.shape
        x = atom_embeddings.reshape(bsz * n_states, n_atoms, dim)
        pad_mask = _mask_from_atoms(x)
        # Ensure fully-padded sequences have at least one unmasked position
        # to avoid NaN in attention (branch-free for torch.compile fullgraph).
        all_padded = pad_mask.all(dim=1)  # [B*n_states]
        pad_mask = pad_mask.clone()
        pad_mask[:, 0] = pad_mask[:, 0] & ~all_padded

        for block in self.blocks:
            x = block(x, padding_mask=pad_mask)
        query = self.seed.expand(x.size(0), -1, -1)
        pooled = self.pma(query, x, kv_padding_mask=pad_mask).squeeze(1)
        out = self.out(pooled).reshape(bsz, n_states, dim)
        if self.regularization > 0:
            self.add_loss(self.regularization * out.norm(p=2))
        return out

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)


# ---------------------------------------------------------------------------
# Relation-aware graph encoders
# ---------------------------------------------------------------------------

class _RelGraphStateBase(nn.Module):
    """Base class for relation-aware atom-graph encoders."""

    def __init__(
        self,
        embed_dim: int,
        n_relations: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_relations = n_relations
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.regularization = regularization
        self.device = device

    def add_loss(self, loss: torch.Tensor):
        _append_custom_loss(self, loss)

    def _build_rel_adj(self, subidx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Build relation-specific adjacency [B, R, N, N] from symbolic indices."""
        # subidx: [B, N, A+1], valid: [B, N]
        bsz, n_atoms = valid.shape
        pred = subidx[..., 0]
        arg1 = subidx[..., 1]
        arg2 = subidx[..., 2] if subidx.size(-1) > 2 else subidx[..., 1]
        valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)
        eye = torch.eye(n_atoms, device=subidx.device, dtype=torch.bool).unsqueeze(0).expand(bsz, -1, -1)

        same_pred = (pred.unsqueeze(2) == pred.unsqueeze(1)) & valid_pair
        share_arg1 = (arg1.unsqueeze(2) == arg1.unsqueeze(1)) & valid_pair
        share_arg2 = (arg2.unsqueeze(2) == arg2.unsqueeze(1)) & valid_pair
        share_cross = (
            (arg1.unsqueeze(2) == arg2.unsqueeze(1))
            | (arg2.unsqueeze(2) == arg1.unsqueeze(1))
        ) & valid_pair

        rels = [
            eye & valid_pair,       # 0: self
            same_pred & ~eye,       # 1: same predicate
            share_arg1 & ~eye,      # 2: share first arg
            share_arg2 & ~eye,      # 3: share second arg
            share_cross & ~eye,     # 4: cross share
        ]
        adj = torch.stack(rels, dim=1).float()
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return adj / deg


class RGCNStateEncoder(_RelGraphStateBase):
    """R-GCN-style state encoder over atom interaction graph."""

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu",
        n_relations: int = 5,
        n_layers: int = 2,
    ):
        super().__init__(
            embed_dim=embed_dim,
            n_relations=n_relations,
            dropout_rate=dropout_rate,
            regularization=regularization,
            device=device,
        )
        self.n_layers = n_layers
        self.rel_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(n_relations, embed_dim, embed_dim) * 0.02) for _ in range(n_layers)]
        )
        self.self_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, atom_embeddings: torch.Tensor, sub_indices: torch.Tensor | None = None) -> torch.Tensor:
        if sub_indices is None:
            raise ValueError("RGCNStateEncoder requires sub_indices.")
        x = self.dropout(atom_embeddings)
        bsz, n_states, n_atoms, emb_dim = x.shape
        x = x.reshape(bsz * n_states, n_atoms, emb_dim)
        sub = sub_indices.reshape(bsz * n_states, n_atoms, sub_indices.size(-1))
        valid = ~_mask_from_atoms(x)
        adj = self._build_rel_adj(sub, valid)  # [BS, R, N, N]

        for layer in range(self.n_layers):
            rel_w = self.rel_weights[layer]      # [R, D, D]
            self_w = self.self_weights[layer]    # [D, D]
            rel_msgs = []
            for rel in range(self.n_relations):
                neigh = torch.matmul(adj[:, rel], x)                 # [BS, N, D]
                rel_msgs.append(torch.matmul(neigh, rel_w[rel]))     # [BS, N, D]
            msg = torch.stack(rel_msgs, dim=0).sum(dim=0)            # [BS, N, D]
            self_msg = torch.matmul(x, self_w)
            x = self.norms[layer](F.relu(msg + self_msg) + x)

        pooled = _safe_mean_pool(x, valid_mask=valid)
        out = self.out(pooled).reshape(bsz, n_states, emb_dim)
        if self.regularization > 0:
            self.add_loss(self.regularization * out.norm(p=2))
        return out


class CompGCNStateEncoder(_RelGraphStateBase):
    """CompGCN-style state encoder with relation composition."""

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = 0.0,
        regularization: float = 0.0,
        device: str = "cpu",
        n_relations: int = 5,
        n_layers: int = 2,
        composition: str = "mult",
    ):
        super().__init__(
            embed_dim=embed_dim,
            n_relations=n_relations,
            dropout_rate=dropout_rate,
            regularization=regularization,
            device=device,
        )
        if composition not in {"mult", "sub"}:
            raise ValueError(f"Unsupported composition: {composition}")
        self.composition = composition
        self.n_layers = n_layers
        self.rel_emb = nn.Parameter(torch.randn(n_relations, embed_dim) * 0.02)
        self.msg_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02) for _ in range(n_layers)]
        )
        self.self_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _compose(self, x: torch.Tensor, rel_vec: torch.Tensor) -> torch.Tensor:
        if self.composition == "mult":
            return x * rel_vec
        return x - rel_vec

    def forward(self, atom_embeddings: torch.Tensor, sub_indices: torch.Tensor | None = None) -> torch.Tensor:
        if sub_indices is None:
            raise ValueError("CompGCNStateEncoder requires sub_indices.")
        x = self.dropout(atom_embeddings)
        bsz, n_states, n_atoms, emb_dim = x.shape
        x = x.reshape(bsz * n_states, n_atoms, emb_dim)
        sub = sub_indices.reshape(bsz * n_states, n_atoms, sub_indices.size(-1))
        valid = ~_mask_from_atoms(x)
        adj = self._build_rel_adj(sub, valid)

        for layer in range(self.n_layers):
            msg_all = 0.0
            for rel in range(self.n_relations):
                rel_vec = self.rel_emb[rel].view(1, 1, emb_dim)
                comp = self._compose(x, rel_vec)
                neigh = torch.matmul(adj[:, rel], comp)
                msg_all = msg_all + torch.matmul(neigh, self.msg_weights[layer])
            self_msg = torch.matmul(x, self.self_weights[layer])
            x = self.norms[layer](F.relu(msg_all + self_msg) + x)

        pooled = _safe_mean_pool(x, valid_mask=valid)
        out = self.out(pooled).reshape(bsz, n_states, emb_dim)
        if self.regularization > 0:
            self.add_loss(self.regularization * out.norm(p=2))
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def Emb_State_Factory(name: str='transe',
            embedding_dim: int=-1,
            padding_atoms: int=10,
            regularization: float=0.0,
            dropout_rate: float=0.0,
            device="cpu",
            set_transformer_heads: int=4,
            set_transformer_sab_layers: int=2,
            **kwargs) -> nn.Module:

    if name.casefold() == 'concat':
        return ConcatStates(padding_atoms, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'sum':
        return SumState(dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'mean':
        return MeanState(dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'max':
        return MaxState(dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'attention':
        return SelfAttentionState(embed_dim=embedding_dim, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'rnn':
        return RNNState(embed_dim=embedding_dim, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'transformer':
        return TransformerState(embed_dim=embedding_dim, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'set_transformer':
        return SetTransformerState(
            embed_dim=embedding_dim,
            num_heads=set_transformer_heads,
            num_sab_layers=set_transformer_sab_layers,
            dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'deepsets':
        return DeepSetsState(embed_dim=embedding_dim, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() == 'phi_sum':
        return PhiSumState(embed_dim=embedding_dim, dropout_rate=dropout_rate, regularization=regularization, device=device)
    elif name.casefold() in {'isab_set_transformer', 'set_transformer_isab', 'isab'}:
        return ISABSetTransformerState(
            embed_dim=embedding_dim,
            dropout_rate=dropout_rate,
            regularization=regularization,
            device=device,
        )
    elif name.casefold() in {'rgcn_state', 'rgcn'}:
        return RGCNStateEncoder(
            embed_dim=embedding_dim,
            dropout_rate=dropout_rate,
            regularization=regularization,
            device=device,
        )
    elif name.casefold() in {'compgcn_state', 'compgcn'}:
        return CompGCNStateEncoder(
            embed_dim=embedding_dim,
            dropout_rate=dropout_rate,
            regularization=regularization,
            device=device,
        )
    else:
        raise ValueError(f"Unknown state embedder: {name}")
