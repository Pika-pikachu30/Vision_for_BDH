"""
verify_setup.py — Run this BEFORE any training to confirm everything works.

Tests:
  1. Model builds and forward pass works
  2. STL-10 data loads correctly
  3. GPU is available and model fits in memory
  4. Estimated training time


"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import time

print("=" * 60)
print("  Vision-BDH STL-10 — Setup Verification")
print("=" * 60)


def check(label: str, fn):
    try:
        result = fn()
        print(f"  ✓ {label}")
        return result
    except Exception as e:
        print(f"  ✗ {label}: {e}")
        raise


# ── 1. Device ─────────────────────────────────────────────────────────────────
device = check("GPU/CPU detection", lambda: (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
))
if device.type == "cuda":
    print(f"    GPU: {torch.cuda.get_device_name(0)} | "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
else:
    print("    WARNING: No GPU. Training will be very slow (100× slower).")


# ── 2. Model imports ──────────────────────────────────────────────────────────
check("Import models", lambda: (
    __import__("models.vision_bdh_v2", fromlist=["build_vision_bdh_v2_stl10"]),
    __import__("models.vit", fromlist=["build_vit_tiny_stl10"]),
))


# ── 3. BDH model forward pass ─────────────────────────────────────────────────
from models.vision_bdh_v2 import build_vision_bdh_v2_stl10, build_vision_bdh_v2_stl10_p12
from models.vit import build_vit_tiny_stl10
from utils import count_params

def test_bdh_p8():
    model = build_vision_bdh_v2_stl10(num_classes=10).to(device)
    x = torch.randn(2, 3, 96, 96).to(device)
    out = model(x)
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"
    n = count_params(model)
    print(f"    BDH v2 (p=8): {n/1e6:.2f}M params | output shape: {out.shape}")
    return model

bdh_model = check("BDH v2 forward pass (patch=8, 96×96)", test_bdh_p8)


def test_bdh_p12():
    model = build_vision_bdh_v2_stl10_p12(num_classes=10).to(device)
    x = torch.randn(2, 3, 96, 96).to(device)
    out = model(x)
    assert out.shape == (2, 10)
    return model

check("BDH v2 forward pass (patch=12, ablation)", test_bdh_p12)


def test_vit():
    model = build_vit_tiny_stl10(num_classes=10).to(device)
    x = torch.randn(2, 3, 96, 96).to(device)
    out = model(x)
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"
    n = count_params(model)
    print(f"    ViT-Tiny (p=8): {n/1e6:.2f}M params | output shape: {out.shape}")
    return model

vit_model = check("ViT-Tiny forward pass (patch=8, 96×96)", test_vit)


# ── 4. STL-10 data ────────────────────────────────────────────────────────────
def test_data():
    from data_stl10 import get_stl10_loaders
    train_loader, val_loader, test_loader = get_stl10_loaders(
        data_root="./data_stl10", batch_size=4, val_split=0.1
    )
    batch = next(iter(train_loader))
    images, labels = batch
    assert images.shape == (4, 3, 96, 96), f"Unexpected shape: {images.shape}"
    print(f"    Batch: images={images.shape}, labels={labels.shape}")
    print(f"    Pixel range: [{images.min():.2f}, {images.max():.2f}]")
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = check("STL-10 dataset loading", test_data)


# ── 5. Training speed benchmark ───────────────────────────────────────────────
def benchmark_speed():
    from torch.optim import AdamW
    from utils import train_one_epoch, get_cosine_schedule_with_warmup
    import torch.nn as nn

    model = build_vision_bdh_v2_stl10(num_classes=10).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 10, 100)
    loss_fn = nn.CrossEntropyLoss()

    # Time 1 epoch (just first few batches)
    model.train()
    t0 = time.time()
    n_batches = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        n_batches += 1
        if n_batches >= 10:
            break

    elapsed = time.time() - t0
    batches_per_sec = n_batches / elapsed
    # Estimate full epoch time
    full_epoch_batches = len(train_loader)
    epoch_time_est = full_epoch_batches / batches_per_sec
    total_50_epochs = 50 * epoch_time_est / 3600

    print(f"    Speed: {batches_per_sec:.1f} batches/sec | "
          f"Epoch est: {epoch_time_est:.0f}s | "
          f"50 epochs est: {total_50_epochs:.1f}h")

    if total_50_epochs > 5:
        print(f"    ⚠ Estimated {total_50_epochs:.1f}h total. Consider reducing epochs or using GPU accelerator.")
    else:
        print(f"    ✓ Estimated training time is reasonable: {total_50_epochs:.1f}h for 50 epochs")

check("Training speed benchmark", benchmark_speed)


# ── 6. Memory check ───────────────────────────────────────────────────────────
if device.type == "cuda":
    def check_memory():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    GPU memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved / {total:.1f}GB total")
        if reserved > total * 0.9:
            print("    ⚠ WARNING: Nearly out of GPU memory! Reduce batch_size.")
        else:
            print(f"    ✓ Memory OK ({reserved/total*100:.0f}% used)")
    check("GPU memory check", check_memory)


print("\n" + "=" * 60)
print("  ALL CHECKS PASSED — Ready to train!")
print("=" * 60)
print("\nNext steps:")
print("  python run_all_experiments.py --exp 1   # Start with BDH")
print("  python run_all_experiments.py --exp 2   # Then ViT baseline")
print("  python run_all_experiments.py --exp 4 --fraction 1.0 --model bdh")
print("    # Or run label efficiency for one fraction at a time")