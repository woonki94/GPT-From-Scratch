import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, q_input_dim, kv_input_dim, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(q_input_dim, d_model)
        self.k_proj = nn.Linear(kv_input_dim, d_model)
        self.v_proj = nn.Linear(kv_input_dim, d_model)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout on attention weights
        self.dropout = nn.Dropout(dropout)

        # Scaling factor (root dim)
        self.scale = math.sqrt(self.d_head)

    def forward(self, q, kv, mask=None):  # <— separate inputs
        B, T_q, _  = q.size()
        _, T_kv, _ = kv.size()

        # Linear projections
        Q = self.q_proj(q)  # [B, T_qkv, d_model]
        K = self.k_proj(kv)  # [B, T_qkv, d_model]
        V = self.v_proj(kv)  # [B, T_qkv, d_model]

        # Split into heads: [B, T, d_model] → [B, n_heads, T, d_head]
        Q = Q.view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)  # [B, n_heads, T_q, d_head]
        K = K.view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)  # [B, n_heads, T_kv, d_head]
        V = V.view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)  # [B, n_heads, T_kv, d_head]

        # Compute scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, n_heads, T_q, T_kv]

        if mask is not None:
            # Accept [L, L] masks and expand to [B, heads, L, L]
            if mask.dim() == 2:  # [L, L]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
                mask = mask.expand(B, self.n_heads, T_q, T_kv)

            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, n_heads, T_q, T_kv]
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        attended = torch.matmul(attn_weights, V)  # [B, n_heads, T_q, d_head]

        # Combine heads
        attended = attended.transpose(1, 2).contiguous()  # [B, T_q, n_heads, d_head]
        attended = attended.view(B, T_q, self.n_heads * self.d_head)  # [B, T_q, d_model]

        # Final projection
        output = self.out_proj(attended)  # [B, T_q, d_model]

        return output, attn_weights

class MultiQueryAttention(nn.Module):
    def __init__(self, q_input_dim, kv_input_dim, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Per-head Q, shared K/V across all heads
        self.q_proj = nn.Linear(q_input_dim, d_model)
        self.k_proj = nn.Linear(kv_input_dim, self.d_head)  # shared across heads
        self.v_proj = nn.Linear(kv_input_dim, self.d_head)  # shared across heads

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, q, kv, mask=None):
        B, T_q, _  = q.size()
        _, T_kv, _ = kv.size()

        # Q: [B, H, T, Dh]
        Q = self.q_proj(q).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)

        # Shared K/V across heads → [B, 1, T, Dh]
        K = self.k_proj(kv).unsqueeze(1)
        V = self.v_proj(kv).unsqueeze(1)

        # Attention scores: [B, H, T, T]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            if mask.dim() == 2:
                mask = mask[None, None, :, :]  # [1, 1, T, T]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attended = torch.matmul(attn_weights, V)  # [B, H, T, Dh]
        attended = attended.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, d_model]

        return self.out_proj(attended), attn_weights


class GroupedQueryAttention(nn.Module):
    def __init__(self, q_input_dim, kv_input_dim, d_model, n_heads, n_kv_groups, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_kv_groups <= n_heads and n_heads % n_kv_groups == 0, "Invalid group config"

        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(q_input_dim, d_model)
        self.k_proj = nn.Linear(kv_input_dim, self.d_head * n_kv_groups)
        self.v_proj = nn.Linear(kv_input_dim, self.d_head * n_kv_groups)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x, mask=None):
        B, T, _ = x.size()

        # Q: [B, H, T, Dh]
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # K/V per group: [B, G, T, Dh]
        K = self.k_proj(x).view(B, T, self.n_kv_groups, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_kv_groups, self.d_head).transpose(1, 2)

        # Map each query head to its group
        head2group = torch.arange(self.n_heads, device=x.device) * self.n_kv_groups // self.n_heads  # [H]

        # Expand group assignments to gather correct K/V for each head
        # K, V: [B, H, T, Dh] after gathering from [B, G, T, Dh]
        K = K.gather(
            dim=1,
            index=head2group[None, :, None, None].expand(B, self.n_heads, T, self.d_head)
        )
        V = V.gather(
            dim=1,
            index=head2group[None, :, None, None].expand(B, self.n_heads, T, self.d_head)
        )

        # Attention scores: [B, H, T, T]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            if mask.dim() == 2:
                mask = mask[None, None, :, :]  # [1, 1, T, T]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)  # [B, H, T, Dh]
        attended = attended.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, d_model]

        return self.out_proj(attended), attn_weights
