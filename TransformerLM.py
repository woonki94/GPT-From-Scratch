import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, q_input_dim, cand_input_dim, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(q_input_dim, d_model)
        self.k_proj = nn.Linear(cand_input_dim, d_model)
        self.v_proj = nn.Linear(cand_input_dim, d_model)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout on attention weights
        self.dropout = nn.Dropout(dropout)

        # Scaling factor (root dim)
        self.scale = math.sqrt(self.d_head)

    def forward(self, query, key_value, mask=None):
        B, T_kv, _ = key_value.size()
        _, T_q, _ = query.size()

        # Linear projections
        Q = self.q_proj(query)  # [B, T_q, d_model]
        K = self.k_proj(key_value)  # [B, T_kv, d_model]
        V = self.v_proj(key_value)  # [B, T_kv, d_model]

        # Split into heads: [B, T, d_model] → [B, n_heads, T, d_head]
        Q = Q.view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)  # [B, n_heads, T_q, d_head]
        K = K.view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)  # [B, n_heads, T_kv, d_head]
        V = V.view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)  # [B, n_heads, T_kv, d_head]

        # Compute scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, n_heads, T_q, T_kv]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

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


class TransformerBLock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, d_model, d_model, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Multi-head self-attention with residual connection and layer norm
        attn_out, _ = self.attn(x, x, mask)
        x = self.attn_norm(x + self.attn_dropout(attn_out))

        # Feed-forward with residual connection and layer norm
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        return x
