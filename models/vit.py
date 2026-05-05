"""
ViT-Tiny baseline for STL-10 experiments.

Architecture matches the original repo's ViT-Tiny:
- 12 independent transformer layers (NOT recurrent like BDH)
- Multi-head attention with softmax
- Standard MLP blocks
- Learned positional embedding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 3, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x


class ViTMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ViTAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ViTMLP(dim, mlp_ratio, dropout)
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTTiny(nn.Module):
    """
    ViT-Tiny baseline

    Config: embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0
    Parameters: ~5.4M for CIFAR-10, ~5.7M for CIFAR-100
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        # 12 INDEPENDENT transformer layers (contrast: BDH uses 1 shared block × 6)
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ViT-Tiny] img={img_size}×{img_size}, patch={patch_size}×{patch_size}, "
              f"tokens={self.n_patches}, params={total_params/1e6:.2f}M")

    def _init_weights(self):
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, n_patches, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)           # (B, n_patches+1, D)
        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Use CLS token for classification
        return self.head(x[:, 0])


def build_vit_tiny_stl10(num_classes: int = 10) -> ViTTiny:
    """ViT-Tiny for STL-10 (96×96), patch_size=8 → 144 tokens."""
    return ViTTiny(
        img_size=96,
        patch_size=8,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
    )


def build_vit_tiny_cifar10(num_classes: int = 10) -> ViTTiny:
    """ViT-Tiny for CIFAR-10 (32×32)"""
    return ViTTiny(
        img_size=32,
        patch_size=4,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
    )