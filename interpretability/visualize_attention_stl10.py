"""
Attention visualization for Vision-BDH v2 on STL-10 images.

Generates attention maps at each recurrent depth (layer 1, middle, final).

Produces:
  - attention_maps_stl10/bdh_attn_{class}_{layer}.png — per-class attention heatmaps
  - attention_maps_stl10/comparison_{class}.png — BDH vs ViT-Tiny side by side
  - attention_maps_stl10/attention_grid.png — all classes overview figure
"""

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.datasets import STL10

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vision_bdh_v2 import build_vision_bdh_v2_stl10
from models.vit import build_vit_tiny_stl10
from data_stl10 import get_val_transform, STL10_CLASSES
from utils import CheckpointManager, setup_gpu


OUTPUT_DIR = "./attention_maps_stl10"

STL10_CLASSES_NICE = [
    "Airplane", "Bird", "Car", "Cat", "Deer",
    "Dog", "Horse", "Monkey", "Ship", "Truck",
]


# ─────────────────────────────────────────────
# BDH Attention extraction
# ─────────────────────────────────────────────

@torch.no_grad()
def get_bdh_attention_maps(model, x: torch.Tensor) -> list:
    """
    Extract attention weight matrices from BDH at each recurrent step.
    Returns list of (n_head, T, T) averaged across batch.
    """
    model.eval()
    B = x.shape[0]
    n_layer = model.n_layer
    attn_layer = model.bdh_block.attn
    ln = model.bdh_block.ln

    x_tok = model.patch_embed(x) + model.pos_embed
    x_cur = x_tok

    all_attn = []
    for step in range(n_layer):
        # Pre-LN
        x_norm = ln(x_cur)
        T = x_norm.shape[1]
        x_h = x_norm.view(B, T, attn_layer.n_head, attn_layer.head_dim).transpose(1, 2)
        x_rope = attn_layer.rope(x_h, T)

        # Sparse latent (Q=K)
        x_latent = torch.einsum('bnti,nio->bnto', x_rope, attn_layer.encoder)
        x_latent = F.relu(x_latent)

        # Raw attention (B, nh, T, T)
        attn = torch.einsum('bnti,bnsi->bnts', x_latent, x_latent)
        # Normalize for visualization
        attn_vis = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        all_attn.append(attn_vis.mean(0).cpu())  # (nh, T, T)

        # Forward through block
        x_cur = model.bdh_block(x_cur)

    return all_attn  # List of n_layer tensors, each (nh, T, T)


@torch.no_grad()
def get_vit_attention_maps(model, x: torch.Tensor) -> list:
    """
    Extract attention maps from ViT-Tiny at each layer.
    Returns list of (n_head, T+1, T+1) tensors.
    """
    model.eval()
    B = x.shape[0]
    n_head = model.blocks[0].attn.num_heads

    x_tok = model.patch_embed(x).flatten(2).transpose(1, 2)
    cls = model.cls_token.expand(B, -1, -1)
    x_cur = torch.cat([cls, x_tok], dim=1) + model.pos_embed

    all_attn = []
    for block in model.blocks:
        x_norm = block.norm1(x_cur)
        B_cur, T, C = x_norm.shape
        head_dim = C // n_head

        qkv = block.attn.qkv(x_norm).reshape(B_cur, T, 3, n_head, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)  # (B, nh, T+1, T+1)
        all_attn.append(attn.mean(0).cpu())  # (nh, T+1, T+1)

        x_cur = block(x_cur)

    return all_attn


# ─────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────

def attention_to_heatmap(attn_matrix: torch.Tensor, patch_grid: int, center_patch: int) -> np.ndarray:
    """
    Extract attention FROM center patch TO all other patches.
    attn_matrix: (n_head, T, T) or (n_head, T+1, T+1)
    Returns (patch_grid, patch_grid) numpy array.
    """
    # Average over heads
    attn_mean = attn_matrix.mean(0)  # (T, T) or (T+1, T+1)

    T = attn_mean.shape[0]
    if T == patch_grid * patch_grid + 1:
        # ViT: first token is CLS — skip it
        attn_mean = attn_mean[1:, 1:]  # (T, T)

    # Attention FROM center patch
    center_attn = attn_mean[center_patch].numpy()  # (T,)
    center_attn = center_attn.reshape(patch_grid, patch_grid)
    center_attn = (center_attn - center_attn.min()) / (center_attn.max() - center_attn.min() + 1e-8)
    return center_attn


def denormalize_stl10(tensor: torch.Tensor) -> np.ndarray:
    """Reverse STL-10 normalization for display."""
    mean = torch.tensor([0.4409, 0.4279, 0.3868]).view(3, 1, 1)
    std  = torch.tensor([0.2683, 0.2610, 0.2687]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


# ─────────────────────────────────────────────
# Main visualization
# ─────────────────────────────────────────────

def visualize_per_class(
    bdh_model,
    vit_model,
    dataset: STL10,
    device: torch.device,
    n_examples: int = 2,
    patch_size: int = 8,
    img_size: int = 96,
):
    """Generate side-by-side attention maps for BDH and ViT on STL-10."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    patch_grid = img_size // patch_size  # 12 for p=8
    center_patch = (patch_grid // 2) * patch_grid + (patch_grid // 2)

    # Group images by class
    by_class = {i: [] for i in range(10)}
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if len(by_class[label]) < n_examples:
            by_class[label].append((img, idx))
        if all(len(v) >= n_examples for v in by_class.values()):
            break

    # BDH uses layers 1 (first), n//2 (middle), n (last)
    n_bdh_layers = bdh_model.n_layer
    vit_layers_to_show = [0, len(vit_model.blocks) // 2, -1]
    bdh_layers_to_show = [0, n_bdh_layers // 2 - 1, n_bdh_layers - 1]

    for cls_idx, class_name in enumerate(STL10_CLASSES_NICE):
        examples = by_class[cls_idx]
        if not examples:
            continue

        for ex_i, (img_tensor, img_idx) in enumerate(examples):
            x = img_tensor.unsqueeze(0).to(device)
            img_np = denormalize_stl10(img_tensor)

            # Get attention maps
            bdh_maps = get_bdh_attention_maps(bdh_model, x)
            vit_maps = get_vit_attention_maps(vit_model, x)

            # ── Plot: BDH ──────────────────────────
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(f"Vision-BDH v2 — {class_name} (STL-10, 96×96)", fontsize=13, fontweight='bold')

            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            layer_labels = ["Layer 1 (early)", f"Layer {n_bdh_layers//2} (middle)", f"Layer {n_bdh_layers} (final)"]
            for ax_i, (layer_idx, label) in enumerate(zip(bdh_layers_to_show, layer_labels)):
                heatmap = attention_to_heatmap(bdh_maps[layer_idx], patch_grid, center_patch)
                heatmap_up = F.interpolate(
                    torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float(),
                    size=(img_size, img_size), mode="bilinear", align_corners=False
                ).squeeze().numpy()

                axes[ax_i + 1].imshow(img_np, alpha=0.6)
                axes[ax_i + 1].imshow(heatmap_up, cmap="hot", alpha=0.6, vmin=0, vmax=1)
                axes[ax_i + 1].set_title(label, fontsize=10)
                axes[ax_i + 1].axis("off")

            plt.tight_layout()
            save_path = os.path.join(OUTPUT_DIR, f"bdh_attention_{class_name.lower()}_{ex_i}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            # ── Plot: ViT-Tiny ─────────────────────
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(f"ViT-Tiny — {class_name} (STL-10, 96×96)", fontsize=13, fontweight='bold')

            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            vit_labels = ["Layer 1 (early)", f"Layer {len(vit_model.blocks)//2} (middle)", "Layer 12 (final)"]
            for ax_i, (layer_idx, label) in enumerate(zip(vit_layers_to_show, vit_labels)):
                # ViT has CLS token — offset
                T_full = patch_grid * patch_grid + 1
                heatmap = attention_to_heatmap(vit_maps[layer_idx], patch_grid, center_patch)
                heatmap_up = F.interpolate(
                    torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float(),
                    size=(img_size, img_size), mode="bilinear", align_corners=False
                ).squeeze().numpy()

                axes[ax_i + 1].imshow(img_np, alpha=0.6)
                axes[ax_i + 1].imshow(heatmap_up, cmap="hot", alpha=0.6, vmin=0, vmax=1)
                axes[ax_i + 1].set_title(label, fontsize=10)
                axes[ax_i + 1].axis("off")

            plt.tight_layout()
            save_path = os.path.join(OUTPUT_DIR, f"vit_attention_{class_name.lower()}_{ex_i}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"✓ Saved attention maps for {class_name} (example {ex_i})")

    print(f"\nAll attention maps saved to: {OUTPUT_DIR}/")


def generate_comparison_figure(classes_to_show=("airplane", "bird", "ship", "cat")):
    """Create paper-ready figure comparing BDH vs ViT attention side by side."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(len(classes_to_show), 7, figure=fig, wspace=0.05, hspace=0.3)

    # Headers
    col_titles = ["Image", "BDH Early", "BDH Mid", "BDH Final", "ViT Early", "ViT Mid", "ViT Final"]
    for col, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.axis("off")

    for row, cls_name in enumerate(classes_to_show):
        # Load pre-generated maps
        for col_offset, model_prefix in enumerate(["bdh", "vit"]):
            for layer_i in range(3):
                path = os.path.join(OUTPUT_DIR, f"{model_prefix}_attention_{cls_name}_0.png")
                if os.path.exists(path):
                    img = plt.imread(path)
                    ax = fig.add_subplot(gs[row, col_offset * 3 + layer_i + 1])
                    # Crop to just the attention panel (approximate)
                    ax.imshow(img)
                    ax.axis("off")

    plt.suptitle("Vision-BDH vs ViT-Tiny: Attention Patterns on STL-10 (96×96)",
                 fontsize=14, fontweight="bold", y=1.02)
    save_path = os.path.join(OUTPUT_DIR, "comparison_figure.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison figure saved: {save_path}")


def main(args):
    device = setup_gpu()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load BDH model
    print("Loading Vision-BDH v2...")
    bdh_model = build_vision_bdh_v2_stl10(num_classes=10).to(device)
    bdh_ckpt = CheckpointManager("./checkpoints_bdh_stl10_p8")
    bdh_ckpt.load_best(bdh_model)
    bdh_model.eval()

    # Load ViT-Tiny model
    print("Loading ViT-Tiny...")
    vit_model = build_vit_tiny_stl10(num_classes=10).to(device)
    vit_ckpt = CheckpointManager("./checkpoints_vit_stl10")
    vit_ckpt.load_best(vit_model)
    vit_model.eval()

    # Load STL-10 test images for visualization
    transform = get_val_transform(img_size=96)
    dataset = STL10(root="./data_stl10", split="test", download=True, transform=transform)

    print("\nGenerating attention maps...")
    visualize_per_class(
        bdh_model=bdh_model,
        vit_model=vit_model,
        dataset=dataset,
        device=device,
        n_examples=args.n_examples,
    )

    if args.comparison_figure:
        generate_comparison_figure()

    print(f"\n✓ Done! Attention maps saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate attention visualizations on STL-10")
    parser.add_argument("--n_examples", type=int, default=2,
                        help="Number of examples per class (default: 2)")
    parser.add_argument("--comparison_figure", action="store_true",
                        help="Also generate combined comparison figure")
    main(parser.parse_args())