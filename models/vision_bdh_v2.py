"""
Vision-BDH v2 — Optimized architecture with Pre-LN + raw attention (no softmax).
Faithfully reconstructed from Pika 2025 (takzen/vision-bdh).

Key features:
- Pre-LayerNorm for stable gradient flow
- Raw attention scores (NO softmax) — synergistic with Pre-LN
- Q=K constraint with RoPE
- Sparse activations (ReLU)
- Multiplicative gating
- Xavier initialization
- Recurrent depth (single shared block)
- Bidirectional attention (no causal mask) for vision

Modifications from original CIFAR code:
- Parameterized img_size and patch_size (supports 96×96 STL-10)
- Returns intermediate activations for attention visualization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bdh import BDHConfig, BDHBlock


class PatchEmbedding(nn.Module):
    """Converts image to patch tokens."""
    def __init__(self, img_size: int, patch_size: int, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        assert img_size % patch_size == 0, f"img_size {img_size} must be divisible by patch_size {patch_size}"
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Convolution-based patch embedding (efficient)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, n_patches, embed_dim)
        x = self.proj(x)           # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)          # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)     # (B, n_patches, embed_dim)
        return x


class VisionBDHv2(nn.Module):
    """
    Vision-BDH v2 — Optimized.

    Architecture:
        Input → PatchEmbedding → Positional Embedding → BDH Core (recurrent) →
        Global Average Pooling → Classification Head

    The BDH core uses a SINGLE block applied n_layer times (weight sharing = recurrent depth).
    This is the key parameter-efficiency innovation vs ViT's independent layers.
    """

    def __init__(
        self,
        bdh_config: BDHConfig,
        img_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        super().__init__()
        self.config = bdh_config
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.n_layer = bdh_config.n_layer

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=bdh_config.n_embd,
        )

        # Learned positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, bdh_config.n_embd))
        nn.init.normal_(self.pos_embed, std=0.02)

        # SINGLE shared BDH block — reused n_layer times (recurrent depth)
        self.bdh_block = BDHBlock(bdh_config, n_tokens=self.n_patches)

        # Final layer norm before classification
        self.norm = nn.LayerNorm(bdh_config.n_embd)

        # Classification head
        self.head = nn.Linear(bdh_config.n_embd, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.dropout = nn.Dropout(bdh_config.dropout)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VisionBDHv2] img={img_size}×{img_size}, patch={patch_size}×{patch_size}, "
              f"tokens={self.n_patches}, params={total_params/1e6:.2f}M")

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)          # (B, n_patches, n_embd)
        x = x + self.pos_embed           # Add positional info
        x = self.dropout(x)

        # Recurrent BDH core — same block applied n_layer times
        intermediate = []
        for _ in range(self.n_layer):
            x = self.bdh_block(x)
            if return_features:
                intermediate.append(x.detach())

        x = self.norm(x)

        # Global Average Pooling (no CLS token, unlike ViT)
        x = x.mean(dim=1)               # (B, n_embd)

        logits = self.head(x)

        if return_features:
            return logits, intermediate
        return logits

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Extract attention maps at each recurrent depth for visualization.
        Returns list of attention weight tensors of shape (B, n_head, T, T).
        """
        self.eval()
        B, C, H, W = x.shape

        x_tok = self.patch_embed(x)
        x_tok = x_tok + self.pos_embed

        attn_maps = []
        x_cur = x_tok

        # Hook into the attention layer
        attn_weights = []

        def hook_fn(module, input, output):
            # Compute attention weights manually for visualization
            with torch.no_grad():
                inp = input[0]
                inp_h = inp.view(B, -1, module.n_head, module.head_dim).transpose(1, 2)
                # Apply norm first (Pre-LN context — attn receives normed x)
                x_norm = module.attn.rope(inp_h, inp_h.shape[2])
                x_latent = torch.einsum('bnti,nio->bnto', x_norm, module.attn.encoder)
                x_latent = F.relu(x_latent)
                attn = torch.einsum('bnti,bnsi->bnts', x_latent, x_latent)
                attn_weights.append(attn.cpu())

        # Register hook on the block's attention
        handle = self.bdh_block.ln.register_forward_hook(
            lambda m, i, o: attn_weights.append(
                self._compute_attn_weights(o, B)
            )
        )

        with torch.no_grad():
            for _ in range(self.n_layer):
                x_cur = self.bdh_block(x_cur)

        handle.remove()
        return attn_weights

    def _compute_attn_weights(self, x_norm: torch.Tensor, B: int) -> torch.Tensor:
        """Helper: compute attention weights from normalized tokens."""
        attn_layer = self.bdh_block.attn
        T = x_norm.shape[1]
        x_h = x_norm.view(B, T, attn_layer.n_head, attn_layer.head_dim).transpose(1, 2)
        x_rope = attn_layer.rope(x_h, T)
        x_latent = torch.einsum('bnti,nio->bnto', x_rope, attn_layer.encoder)
        x_latent = F.relu(x_latent)
        attn = torch.einsum('bnti,bnsi->bnts', x_latent, x_latent)
        return attn.cpu()


def build_vision_bdh_v2_stl10(num_classes: int = 10) -> VisionBDHv2:
    """
    Build Vision-BDH v2 configured for STL-10 (96×96 images).
    patch_size=8 → 144 tokens (12×12 grid).
    """
    config = BDHConfig(
        n_layer=6,
        n_embd=256,
        n_head=4,
        vocab_size=256,
        mlp_internal_dim_multiplier=64,
    )
    return VisionBDHv2(
        bdh_config=config,
        img_size=96,
        patch_size=8,
        num_classes=num_classes,
    )


def build_vision_bdh_v2_stl10_p12(num_classes: int = 10) -> VisionBDHv2:
    """
    Ablation: patch_size=12 → 64 tokens (8×8 grid). Same token count as CIFAR baseline.
    """
    config = BDHConfig(
        n_layer=6,
        n_embd=256,
        n_head=4,
        vocab_size=256,
        mlp_internal_dim_multiplier=64,
    )
    return VisionBDHv2(
        bdh_config=config,
        img_size=96,
        patch_size=12,
        num_classes=num_classes,
    )


def build_vision_bdh_v2_cifar10(num_classes: int = 10) -> VisionBDHv2:
    """Original CIFAR-10 config for verification."""
    config = BDHConfig(
        n_layer=6,
        n_embd=256,
        n_head=4,
        vocab_size=256,
        mlp_internal_dim_multiplier=64,
    )
    return VisionBDHv2(
        bdh_config=config,
        img_size=32,
        patch_size=4,
        num_classes=num_classes,
    )
