import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import math
from attentions import MultiHeadAttention, MultiQueryAttention
#from flash_attn.modules.mha import MHA


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

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts

        # Expert networks: [E] × MLP(input_dim → hidden_dim → input_dim)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])

        # Gating layer: maps input to logits over experts
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        x: [B, T, D]
        Returns:
            output: [B, T, D] – MoE-processed tensor
        """
        B, T, D = x.size()

        # Compute gating logits and pick top-1 expert per token
        gate_scores = self.gate(x)                  # [B, T, E]
        top1 = torch.argmax(gate_scores, dim=-1)    # [B, T]

        # Prepare output
        output = torch.zeros_like(x)

        for expert_idx, expert in enumerate(self.experts):
            mask = (top1 == expert_idx).unsqueeze(-1)   # [B, T, 1]
            expert_input = x * mask                    # [B, T, D] masked
            expert_output = expert(expert_input)
            output += expert_output * mask             # accumulate output

        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, d_model, d_model, n_heads, dropout)
        #self.attn = FlashMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        '''
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        '''

        self.ffn = MoELayer(d_model, ffn_dim, num_experts=4, dropout=dropout)
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