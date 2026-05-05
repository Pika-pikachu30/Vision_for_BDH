"""
Experiment 4: Label Efficiency — THE NOVEL CONTRIBUTION.

Trains both Vision-BDH v2 and ViT-Tiny with 10%, 25%, 50%, 100% of STL-10 training data.
Produces the "accuracy vs data fraction" figure
"""

import os
import sys
import time
import json
import argparse

import torch
from torch import nn
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vision_bdh_v2 import build_vision_bdh_v2_stl10
from models.vit import build_vit_tiny_stl10
from data_stl10 import get_stl10_fraction_loader, get_stl10_loaders
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

# Data fractions to test — core of the label efficiency experiment
FRACTIONS = [0.10, 0.25, 0.50, 1.00]

# Scale epochs with data (more data → same total compute)
EPOCHS_PER_FRACTION = {
    0.10: 100,  # 10% data → train longer to reach convergence
    0.25: 80,
    0.50: 60,
    1.00: 50,   # Same as main experiment
}

BASE_CONFIG = {
    "img_size": 96,
    "patch_size": 8,
    "num_classes": 10,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "warmup_steps": 200,    # Fewer steps for fractional data
    "grad_clip": 1.0,
    "data_root": "./data_stl10",
}


def train_model_on_fraction(
    model_name: str,
    model: torch.nn.Module,
    fraction: float,
    device: torch.device,
    results_dir: str,
) -> dict:
    """Train a model on `fraction` of STL-10 and return results."""

    epochs = EPOCHS_PER_FRACTION[fraction]
    exp_name = f"{model_name}_frac{int(fraction*100):03d}"
    checkpoint_dir = os.path.join(results_dir, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {model_name} | {fraction*100:.0f}% data | {epochs} epochs")
    print(f"{'='*70}")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=BASE_CONFIG["lr"],
                      weight_decay=BASE_CONFIG["weight_decay"])

    if fraction == 1.0:
        train_loader, val_loader, test_loader = get_stl10_loaders(
            data_root=BASE_CONFIG["data_root"],
            batch_size=BASE_CONFIG["batch_size"],
            img_size=BASE_CONFIG["img_size"],
        )
    else:
        train_loader, val_loader, test_loader = get_stl10_fraction_loader(
            fraction=fraction,
            data_root=BASE_CONFIG["data_root"],
            batch_size=BASE_CONFIG["batch_size"],
            img_size=BASE_CONFIG["img_size"],
        )

    num_training_steps = epochs * len(train_loader)
    warmup = min(BASE_CONFIG["warmup_steps"], num_training_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, num_training_steps)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    logger = MetricsLogger(
        os.path.join(checkpoint_dir, "metrics.csv"),
        extra_fields=["fraction", "model"]
    )
    ckpt_manager = CheckpointManager(checkpoint_dir)

    best_val_acc = 0.0
    epoch_curves = []  # For plotting

    for epoch in range(epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            grad_clip=BASE_CONFIG["grad_clip"],
            epoch=epoch, total_epochs=epochs,
            log_every=50,
        )
        val_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        best_val_acc = max(best_val_acc, val_acc)

        epoch_curves.append({"epoch": epoch + 1, "val_accuracy": val_acc, "train_loss": avg_loss})

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  [{exp_name}] Ep {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")

        logger.log({
            "epoch": epoch + 1, "train_loss": round(avg_loss, 4),
            "val_accuracy": round(val_acc, 2), "epoch_time_sec": round(epoch_time, 1),
            "learning_rate": round(lr, 7),
            "fraction": fraction, "model": model_name,
        })
        ckpt_manager.save(model, optimizer, epoch, val_acc)

    # Final test eval
    best_val = ckpt_manager.load_best(model)
    test_acc = evaluate(model, test_loader, device)
    print(f"\n  [{exp_name}] FINAL TEST: {test_acc:.2f}% (best val: {best_val:.2f}%)")

    result = {
        "model": model_name,
        "fraction": fraction,
        "n_train_samples": len(train_loader.dataset),
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val,
        "params_M": count_params(model) / 1e6,
        "epochs": epochs,
        "learning_curves": epoch_curves,
    }

    with open(os.path.join(checkpoint_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def main(args):
    print_experiment_header(
        "Label Efficiency Experiment — BDH vs ViT-Tiny (STL-10)",
        {"fractions": FRACTIONS, **BASE_CONFIG}
    )

    device = setup_gpu()
    results_dir = "./results_label_efficiency"
    os.makedirs(results_dir, exist_ok=True)

    # Determine which experiments to run
    fractions_to_run = [args.fraction] if args.fraction else FRACTIONS
    models_to_run = [args.model] if args.model else ["bdh", "vit"]

    all_results = []

    for fraction in fractions_to_run:
        for model_name in models_to_run:
            if model_name == "bdh":
                model = build_vision_bdh_v2_stl10(num_classes=10)
            else:
                model = build_vit_tiny_stl10(num_classes=10)

            result = train_model_on_fraction(
                model_name=model_name,
                model=model,
                fraction=fraction,
                device=device,
                results_dir=results_dir,
            )
            all_results.append(result)

    # Save consolidated results
    summary_path = os.path.join(results_dir, "all_results.json")
    with open(summary_path, "w") as f:
        # Remove learning_curves from summary to keep it readable
        summary = [
            {k: v for k, v in r.items() if k != "learning_curves"}
            for r in all_results
        ]
        json.dump(summary, f, indent=2)
    print(f"\nAll results saved to: {summary_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  LABEL EFFICIENCY SUMMARY")
    print("=" * 70)
    print(f"{'Fraction':>10} {'Model':>15} {'Test Acc':>10} {'N Train':>10}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['fraction']*100:>9.0f}% {r['model']:>15} "
              f"{r['test_accuracy']:>9.2f}% {r['n_train_samples']:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Efficiency Experiment")
    parser.add_argument("--fraction", type=float, default=None,
                        help="Run only this fraction (0.1, 0.25, 0.5, 1.0). Default: all.")
    parser.add_argument("--model", type=str, default=None,
                        choices=["bdh", "vit"],
                        help="Run only this model. Default: both.")
    args = parser.parse_args()
    main(args)