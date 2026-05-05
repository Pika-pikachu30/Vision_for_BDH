"""
Shared training utilities for all Vision-BDH STL-10 experiments.
Keeps training scripts DRY — all experiments import from here.
"""

import os
import csv
import glob
import math
import time
import json
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# Learning rate schedule
# ─────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup → cosine decay. Identical to Pika 2025."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# ─────────────────────────────────────────────
# Metrics logging
# ─────────────────────────────────────────────

class MetricsLogger:
    """CSV logger matching Pika 2025's format, extended with extra fields."""

    def __init__(self, log_path: str, extra_fields: list = None):
        self.log_path = log_path
        base_fields = ["epoch", "train_loss", "val_accuracy", "epoch_time_sec", "learning_rate"]
        self.fields = base_fields + (extra_fields or [])

        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(self.fields)

    def log(self, values: Dict[str, Any]):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(values)


# ─────────────────────────────────────────────
# Checkpoint management
# ─────────────────────────────────────────────

class CheckpointManager:
    """Save / load checkpoints, tracking best validation accuracy."""

    def __init__(self, checkpoint_dir: str, keep_best: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.keep_best = keep_best
        self.best_val_acc = 0.0
        self.best_path = ""
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_accuracy: float,
        extra: Optional[Dict] = None,
    ) -> str:
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

        if val_accuracy > self.best_val_acc:
            self.best_val_acc = val_accuracy
            self.best_path = os.path.join(
                self.checkpoint_dir, f"best_model_acc{val_accuracy:.2f}.pth"
            )
            torch.save(payload, self.best_path)

        return path

    def load_latest(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load most recent checkpoint. Returns start_epoch."""
        checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth")))
        if not checkpoints:
            print("No checkpoints found — starting fresh.")
            return 0

        path = checkpoints[-1]
        print(f"Resuming from: {path}")
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"] + 1

    def load_best(self, model: nn.Module) -> float:
        """Load the best checkpoint. Returns best val accuracy."""
        # Find all best_model files and pick highest accuracy
        bests = glob.glob(os.path.join(self.checkpoint_dir, "best_model_acc*.pth"))
        if not bests:
            # Fall back to all checkpoints
            bests = sorted(glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth")))
        if not bests:
            print("No best checkpoint found.")
            return 0.0

        # Pick highest val_accuracy
        best_acc = 0.0
        best_path = bests[0]
        for p in bests:
            ckpt = torch.load(p, map_location="cpu")
            if ckpt.get("val_accuracy", 0) > best_acc:
                best_acc = ckpt["val_accuracy"]
                best_path = p

        print(f"Loading best model: {best_path} (val_acc={best_acc:.2f}%)")
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        return best_acc


# ─────────────────────────────────────────────
# Training / evaluation loops
# ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
    epoch: int = 0,
    total_epochs: int = 0,
    log_every: int = 100,
) -> float:
    model.train()
    total_loss = 0.0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (i + 1) % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch+1}/{total_epochs} | Batch {i+1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | LR: {lr:.6f}"
            )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_topk(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    k: int = 5,
) -> tuple:
    """Returns (top1_acc, topk_acc)."""
    model.eval()
    correct_1 = correct_k = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        # Top-1
        preds = logits.argmax(dim=1)
        correct_1 += (preds == labels).sum().item()

        # Top-k
        _, top_k = logits.topk(k, dim=1)
        labels_exp = labels.view(-1, 1).expand_as(top_k)
        correct_k += (top_k == labels_exp).any(dim=1).sum().item()

        total += labels.size(0)

    return 100.0 * correct_1 / total, 100.0 * correct_k / total


# ─────────────────────────────────────────────
# GPU setup
# ─────────────────────────────────────────────

def setup_gpu() -> torch.device:
    """Configure GPU for best performance, matching Pika 2025."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU found, training on CPU (will be very slow).")
    return device


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_experiment_header(title: str, config: dict):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 70)