"""
Experiment 3 (Ablation): Vision-BDH v2 on STL-10 with patch_size=12.

patch_size=12 → 64 tokens (8×8 grid) — same as CIFAR-10 (4×4 patches on 32×32).
Comparison with patch_size=8 (144 tokens) shows effect of token count on accuracy.
"""

import os
import sys
import time
import argparse
import json

import torch
from torch import nn
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vision_bdh_v2 import build_vision_bdh_v2_stl10_p12
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
    "img_size": 96,
    "patch_size": 12,        # Ablation: coarser patches (64 tokens like CIFAR)
    "n_tokens": 64,
    "num_classes": 10,
    "epochs": 50,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "warmup_steps": 500,
    "grad_clip": 1.0,
    "val_split": 0.1,
    "data_root": "./data_stl10",
    "checkpoint_dir": "./checkpoints_bdh_stl10_p12",
    "log_file": "./checkpoints_bdh_stl10_p12/metrics_bdh_stl10_p12.csv",
}


def main(args):
    print_experiment_header(
        "Vision-BDH v2 on STL-10 — Ablation: patch_size=12 (64 tokens)", CONFIG
    )

    device = setup_gpu()

    model = build_vision_bdh_v2_stl10_p12(num_classes=CONFIG["num_classes"])
    print(f"Parameters: {count_params(model)/1e6:.2f}M")

    try:
        model = torch.compile(model, backend="aot_eager")
        print("✓ Model compiled")
    except Exception as e:
        print(f"  Compilation skipped: {e}")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    train_loader, val_loader, test_loader = get_stl10_loaders(
        data_root=CONFIG["data_root"],
        batch_size=CONFIG["batch_size"],
        val_split=CONFIG["val_split"],
        img_size=CONFIG["img_size"],
    )

    num_training_steps = CONFIG["epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=num_training_steps,
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    ckpt_manager = CheckpointManager(CONFIG["checkpoint_dir"])
    logger = MetricsLogger(CONFIG["log_file"])

    start_epoch = 0
    if args.resume:
        start_epoch = ckpt_manager.load_latest(model, optimizer)

    best_val_acc = 0.0
    for epoch in range(start_epoch, CONFIG["epochs"]):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            grad_clip=CONFIG["grad_clip"], epoch=epoch, total_epochs=CONFIG["epochs"],
        )
        val_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        best_val_acc = max(best_val_acc, val_acc)

        print("-" * 70)
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} | "
              f"Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}% | Time: {epoch_time:.1f}s")
        print("-" * 70)

        logger.log({
            "epoch": epoch + 1, "train_loss": round(avg_loss, 4),
            "val_accuracy": round(val_acc, 2), "epoch_time_sec": round(epoch_time, 1),
            "learning_rate": round(lr, 7),
        })
        ckpt_manager.save(model, optimizer, epoch, val_acc)

    print("\n" + "=" * 70)
    best_val = ckpt_manager.load_best(model)
    test_acc = evaluate(model, test_loader, device)
    print(f"Best Val Accuracy : {best_val:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    summary = {
        "experiment": "Vision-BDH v2 STL-10 patch_size=12 (ablation)",
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val,
        "params_M": count_params(model) / 1e6,
        "img_size": CONFIG["img_size"],
        "patch_size": CONFIG["patch_size"],
        "n_tokens": CONFIG["n_tokens"],
        "epochs": CONFIG["epochs"],
    }
    with open(os.path.join(CONFIG["checkpoint_dir"], "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())