"""
BDH (Baby Dragon Hatchling) configuration and core components.
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 192
    n_head: int = 6
    vocab_size: int = 256
    mlp_internal_dim_multiplier: int = 32
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.n_embd // self.n_head


class RoPEEmbedding(nn.Module):
    """Rotary Positional Embedding — used in original BDH."""
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x, seq_len: int):
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return (x * cos) + (self.rotate_half(x) * sin)


class BDHAttention(nn.Module):
    """
    BDH attention: Q=K constraint + raw scores (no softmax) + sparse (ReLU) activations.
    Bidirectional (no causal mask) for vision.
    """
    def __init__(self, config: BDHConfig, n_tokens: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_tokens = n_tokens
        mlp_dim = config.n_embd * config.mlp_internal_dim_multiplier // config.n_head

        # Q=K constraint: single shared encoder projects to sparse latent space
        # Shape: (n_head, n_embd_per_head, mlp_dim)
        self.encoder = nn.Parameter(
            torch.empty(config.n_head, self.head_dim, mlp_dim)
        )
        # Value encoder (separate from Q=K encoder)
        self.encoder_v = nn.Parameter(
            torch.empty(config.n_head, self.head_dim, mlp_dim)
        )
        # Decoder: maps back from latent to embedding space
        self.decoder = nn.Parameter(
            torch.empty(config.n_head, mlp_dim, self.head_dim)
        )
        # Gating mechanism (multiplicative, BDH innovation)
        self.gate = nn.Parameter(
            torch.empty(config.n_head, self.head_dim, self.head_dim)
        )

        self.rope = RoPEEmbedding(self.head_dim, max_seq_len=n_tokens + 64)
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform init (v2 improvement over v1)
        for p in [self.encoder, self.encoder_v, self.decoder, self.gate]:
            nn.init.xavier_uniform_(p.view(p.shape[0], -1).unsqueeze(0)).squeeze(0)
            # simpler: just xavier per head slice
        nn.init.xavier_uniform_(self.encoder.reshape(self.n_head, -1).unsqueeze(0)
                                 ).squeeze(0)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.encoder.view(-1, self.encoder.shape[-1]))
            nn.init.xavier_uniform_(self.encoder_v.view(-1, self.encoder_v.shape[-1]))
            nn.init.xavier_uniform_(self.decoder.view(-1, self.decoder.shape[-1]))
            nn.init.xavier_uniform_(self.gate.view(-1, self.gate.shape[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # Reshape to (B, n_head, T, head_dim)
        x_h = x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to x_h for Q=K constraint
        x_rope = self.rope(x_h, T)

        # Sparse projection: ReLU produces sparse activations (BDH key feature)
        # Q=K: same projection used for both query and key
        # x_rope: (B, nh, T, head_dim) @ encoder: (nh, head_dim, mlp_dim)
        x_latent = torch.einsum('bnti,nio->bnto', x_rope, self.encoder)
        x_latent = F.relu(x_latent)  # SPARSE activations — key BDH feature

        # Value latent (separate encoder)
        x_v_latent = torch.einsum('bnti,nio->bnto', x_h, self.encoder_v)
        x_v_latent = F.relu(x_v_latent)

        # Raw attention scores (no softmax — v2 optimized key insight)
        # attn: (B, nh, T, T) via outer product of sparse latents
        # Q=K means attn[i,j] = dot(sparse(x_i), sparse(x_j))
        N = x_latent.shape[2]
        attn = torch.einsum('bnti,bnsi->bnts', x_latent, x_latent) # / math.sqrt(N) # (can change this)
        # NO softmax — raw scores, bounded by RoPE & Q=K natural normalization
        attn = self.dropout(attn)

        # Aggregate values: (B, nh, T, mlp_dim) weighted by attn
        out_latent = torch.einsum('bnts,bnso->bnto', attn, x_v_latent)

        # Decode back to embedding space
        out = torch.einsum('bnto,noi->bnti', out_latent, self.decoder)

        # Multiplicative gating (BDH innovation — replaces standard residuals)
        gate_out = torch.einsum('bnti,nij->bntj', x_h, self.gate)
        gate_out = torch.sigmoid(gate_out)
        out = out * gate_out

        # Merge heads back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return out


class BDHBlock(nn.Module):
    """Single BDH block: Pre-LN + BDH attention + SIGMOID GATING."""
    def __init__(self, config: BDHConfig, n_tokens: int):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd)
        self.attn = BDHAttention(config, n_tokens)
        self.dropout = nn.Dropout(config.dropout)
        
        self.gate_proj = nn.Linear(config.n_embd, config.n_embd)
        # Initialize gate bias to a small positive value (e.g., 1.0) 
        # to start with a "mostly open" gate during early training.
        nn.init.constant_(self.gate_proj.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm
        x_norm = self.ln(x)
        
        # Calculate the Attention output
        attn_out = self.attn(x_norm)
        attn_out = self.dropout(attn_out)
        
        # --- NEW GATING LOGIC ---
        # Compute the sigmoid gate based on the normalized input
        # This allows the model to selectively 'write' to the state
        gate = torch.sigmoid(self.gate_proj(x_norm))
        
        x = x + (gate * attn_out) 
        # ------------------------
        
        return x