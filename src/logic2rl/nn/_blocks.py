"""Shared NN building blocks used by atom and state embedders.

Provides CUDA-graph-safe attention modules, GRU cells, and utility helpers
that are consumed by both ``atom_embedders`` and ``state_embedders``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _append_custom_loss(module: nn.Module, loss: torch.Tensor) -> None:
    if not hasattr(module, "_custom_losses"):
        module._custom_losses = []
    module._custom_losses.append(loss)


def _resolve_num_heads(embed_dim: int, requested_heads: int) -> int:
    """Pick a valid head count that divides embed_dim."""
    if requested_heads <= 0:
        raise ValueError("requested_heads must be positive")
    heads = min(requested_heads, max(1, embed_dim))
    while heads > 1 and (embed_dim % heads != 0):
        heads -= 1
    return heads


def _mask_from_atoms(atom_embeddings: torch.Tensor) -> torch.Tensor:
    """Return True for padded atoms."""
    return atom_embeddings.abs().sum(dim=-1) == 0


def _safe_mean_pool(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pool over atom axis."""
    weights = valid_mask.unsqueeze(-1).float()
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


# ---------------------------------------------------------------------------
# Manual GRU cell (torch.compile fullgraph safe)
# ---------------------------------------------------------------------------

class _ManualGRUCell(nn.Module):
    """Manual GRU cell using Linear layers (torch.compile fullgraph safe).

    nn.GRU/nn.GRUCell are not supported by Dynamo with fullgraph=True.
    This implements the standard GRU equations with explicit projections.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Gates: reset (r), update (z), and new (n)
        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Single GRU step: x [N, D_in], h [N, D_h] -> h_new [N, D_h]."""
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h))
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h))
        return (1 - z) * n + z * h


# ---------------------------------------------------------------------------
# Manual multi-head attention (CUDA graph safe)
# ---------------------------------------------------------------------------

class _ManualMultiheadAttention(nn.Module):
    """Manual multi-head attention using matmul+softmax (CUDA graph safe).

    Avoids nn.MultiheadAttention which uses SDPA kernels that can fail
    during CUDA graph capture with certain shape configurations.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = _resolve_num_heads(embed_dim, num_heads)
        self.head_dim = embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[N, L, E] query/key/value → [N, L_q, E] output."""
        N, L_q, E = query.shape
        L_kv = key.shape[1]
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(query).reshape(N, L_q, H, D).transpose(1, 2)   # [N, H, L_q, D]
        k = self.k_proj(key).reshape(N, L_kv, H, D).transpose(1, 2)    # [N, H, L_kv, D]
        v = self.v_proj(value).reshape(N, L_kv, H, D).transpose(1, 2)  # [N, H, L_kv, D]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [N, H, L_q, L_kv]
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.bool()
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, L_kv]
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        out = (attn @ v).transpose(1, 2).reshape(N, L_q, E)  # [N, L_q, E]
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer block (PreNorm, CUDA graph safe)
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    """PreNorm Transformer block with manual attention (CUDA graph safe)."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        N, L, E = x.shape
        H, D = self.num_heads, self.head_dim
        xn = self.norm1(x)
        q = self.q_proj(xn).reshape(N, L, H, D).transpose(1, 2)
        k = self.k_proj(xn).reshape(N, L, H, D).transpose(1, 2)
        v = self.v_proj(xn).reshape(N, L, H, D).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            mask = key_padding_mask.bool().unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(N, L, E)
        x = x + self.out_proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Set Transformer blocks (_MAB, _ISAB)
# ---------------------------------------------------------------------------

class _MAB(nn.Module):
    """Multihead attention block used by Set Transformer variants.

    Uses manual matmul+softmax attention instead of nn.MultiheadAttention
    to be compatible with torch.compile(fullgraph=True) + CUDA graphs.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            # Fall back to largest divisor
            for h in (num_heads, 4, 2, 1):
                if dim % h == 0:
                    num_heads = h
                    break
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_o = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, L_q, E = q.shape
        L_kv = kv.shape[1]
        H, D = self.num_heads, self.head_dim

        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)

        q_h = self.q_proj(qn).reshape(N, L_q, H, D).transpose(1, 2)   # [N, H, L_q, D]
        k_h = self.k_proj(kvn).reshape(N, L_kv, H, D).transpose(1, 2) # [N, H, L_kv, D]
        v_h = self.v_proj(kvn).reshape(N, L_kv, H, D).transpose(1, 2) # [N, H, L_kv, D]

        attn = (q_h @ k_h.transpose(-2, -1)) * self.scale  # [N, H, L_q, L_kv]
        if kv_padding_mask is not None:
            mask = kv_padding_mask.bool().unsqueeze(1).unsqueeze(2)  # [N, 1, 1, L_kv]
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn_out = (attn @ v_h).transpose(1, 2).reshape(N, L_q, E)  # [N, L_q, E]
        attn_out = self.out_proj(attn_out)

        x = q + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm_o(x)))
        return x


class _ISAB(nn.Module):
    """Induced Set Attention Block (Set Transformer)."""

    def __init__(self, dim: int, num_heads: int, num_inducing: int, dropout: float):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inducing, dim))
        self.mab_i = _MAB(dim=dim, num_heads=num_heads, dropout=dropout)
        self.mab_o = _MAB(dim=dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz = x.size(0)
        inducing = self.inducing.expand(bsz, -1, -1)
        h = self.mab_i(inducing, x, kv_padding_mask=padding_mask)
        return self.mab_o(x, h, kv_padding_mask=None)
