import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import math

#from flash_attn.modules.mha import MHA


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

    def forward(self, qkv, mask=None):
        B, T_kv, _ = qkv.size()
        _, T_q, _ = qkv.size()

        # Linear projections
        Q = self.q_proj(qkv)  # [B, T_qkv, d_model]
        K = self.k_proj(qkv)  # [B, T_qkv, d_model]
        V = self.v_proj(qkv)  # [B, T_qkv, d_model]

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

'''
class FlashMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.flash_attn = MHA(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            causal=True,     # very important for decoder-style (GPT) models
            #bias=True        # match your original setup
        )

    def forward(self, x, mask=None):
        # FlashAttention handles causal masking internally
        out = self.flash_attn(x)
        return out, None  # match original API (output, attn_weights)
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        B, L, d_model = x.size()
        position = torch.arange(0, L, dtype=torch.float, device=x.device).unsqueeze(1)

        # Calculate frequency
        div_term = torch.exp(
            -math.log(10000.0) * torch.arange(0, self.d_model, 2, device=x.device).float() / self.d_model)

        # positional encodings
        pe = torch.zeros(L, d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #singleton dimension ->  (1, L, d_model)
        pe = pe.unsqueeze(0)

        # input+pe
        return x + pe

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, d_model, d_model, n_heads, dropout)
        #self.attn = FlashMultiHeadAttention(d_model, n_heads, dropout)
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

    def forward(self, x, mask=None, return_attn=False):
        #both flash, mha-> compatible
        attn_out, attn_weights = self.attn( x, mask)

        x = self.attn_norm(x + self.attn_dropout(attn_out))

        # Feed-forward with residual connection and layer norm
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        if return_attn:
            return x, attn_weights
        else:
            return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers=1, dropout=0.1):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim=d_model * 4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, vocab_size)

    def generateCausalMask(self, L, device):
        # Shape: [L, L]
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).to(torch.bool)
        # Shape needed: [1, 1, L, L] for broadcasting with [B, heads, L, L]
        return mask

    def forward(self, x, return_attn=False):
        B, L = x.size()
        x = self.embeddings(x)                    # [B, L, d_model]
        x = self.position(x)                      # [B, L, d_model]

        # Create causal mask [1, 1, L, L]
        causal_mask = self.generateCausalMask(L, x.device)

        all_attn_weights = [] if return_attn else None

        # Pass through each Transformer block with masking
        for block in self.blocks:
            if return_attn:
                x, attn_weights = block(x, mask=causal_mask, return_attn=True)
                all_attn_weights.append(attn_weights)
            else:
                x = block(x, mask=causal_mask)

        logits = self.classifier(x)               # [B, L, vocab_size]

        if return_attn:
            return logits, all_attn_weights
        else:
            return logits