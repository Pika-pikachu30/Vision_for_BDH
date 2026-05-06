"""
Experiment 1: Vision-BDH v2 on STL-10 (96×96, patch_size=8 → 144 tokens)

This is the PRIMARY experiment of the paper.
Architecture change from CIFAR: img_size=96, patch_size=8 (2 lines changed).

"""

import os
import sys
import time
import argparse

import torch
from torch import nn
from torch.optim import AdamW

# Add parent to path when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vision_bdh_v2 import build_vision_bdh_v2_stl10
from data_stl10 import get_stl10_loaders
from utils import (
    get_cosine_schedule_with_warmup,
    MetricsLogger,
    CheckpointManager,
    train_one_epoch,
    evaluate,
    setup_gpu,
    count_params,
    print_experiment_header,
)


CONFIG = {
    # Architecture (only 2 values changed from CIFAR!)
    "img_size": 96,          # Was 32 for CIFAR
    "patch_size": 8,         # Was 4 for CIFAR → same token density
    "n_tokens": 144,         # (96/8)² = 144 tokens
    "num_classes": 10,

    "epochs": 50,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "warmup_steps": 500,     # Scaled from 1000 (half data → half warmup)
    "grad_clip": 1.0,
    "val_split": 0.1,

    # Paths
    "data_root": "./data_stl10",
    "checkpoint_dir": "./checkpoints_bdh_stl10_p8",
    "log_file": "./checkpoints_bdh_stl10_p8/metrics_bdh_stl10_p8.csv",
}


def main(args):
    print_experiment_header("Vision-BDH v2 on STL-10 (patch_size=8)", CONFIG)

    device = setup_gpu()

    # ── Model ──────────────────────────────────
    model = build_vision_bdh_v2_stl10(num_classes=CONFIG["num_classes"])
    print(f"Parameters: {count_params(model)/1e6:.2f}M")

    # torch.compile for speed (optional, falls back gracefully)
    try:
        model = torch.compile(model, backend="aot_eager")
        print("✓ Model compiled (aot_eager)")
    except Exception as e:
        print(f"  Compilation skipped: {e}")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    # ── Data ───────────────────────────────────
    train_loader, val_loader, test_loader = get_stl10_loaders(
        data_root=CONFIG["data_root"],
        batch_size=CONFIG["batch_size"],
        val_split=CONFIG["val_split"],
        img_size=CONFIG["img_size"],
    )

    # ── Scheduler ──────────────────────────────
    num_training_steps = CONFIG["epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=num_training_steps,
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ── Checkpoint & logging ───────────────────
    ckpt_manager = CheckpointManager(CONFIG["checkpoint_dir"])
    logger = MetricsLogger(CONFIG["log_file"])

    start_epoch = 0
    if args.resume:
        start_epoch = ckpt_manager.load_latest(model, optimizer)

    # ── Training loop ──────────────────────────
    print(f"\nTraining for {CONFIG['epochs']} epochs...")
    best_val_acc = 0.0

    for epoch in range(start_epoch, CONFIG["epochs"]):
        t0 = time.time()

        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            grad_clip=CONFIG["grad_clip"],
            epoch=epoch,
            total_epochs=CONFIG["epochs"],
        )

        val_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print("-" * 70)
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
              f"Loss: {avg_loss:.4f} | Val: {val_acc:.2f}% | "
              f"Best: {best_val_acc:.2f}% | Time: {epoch_time:.1f}s")
        print("-" * 70)

        logger.log({
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "val_accuracy": round(val_acc, 2),
            "epoch_time_sec": round(epoch_time, 1),
            "learning_rate": round(lr, 7),
        })

        ckpt_manager.save(model, optimizer, epoch, val_acc)

    # ── Final test evaluation ──────────────────
    print("\n" + "=" * 70)
    print("  Final Test Evaluation (Best Checkpoint)")
    print("=" * 70)

    best_val = ckpt_manager.load_best(model)
    test_acc = evaluate(model, test_loader, device)

    print(f"Best Val Accuracy : {best_val:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # Save final result summary
    summary = {
        "experiment": "Vision-BDH v2 STL-10 patch_size=8",
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val,
        "params_M": count_params(model) / 1e6,
        "img_size": CONFIG["img_size"],
        "patch_size": CONFIG["patch_size"],
        "n_tokens": CONFIG["n_tokens"],
        "epochs": CONFIG["epochs"],
    }

    import json
    summary_path = os.path.join(CONFIG["checkpoint_dir"], "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision-BDH v2 on STL-10")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    main(args)
