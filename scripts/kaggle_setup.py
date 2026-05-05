"""
kaggle_setup.py — Run this first cell on Kaggle to set up the environment.

Paste this as the first cell of your Kaggle notebook.
Then run: !python run_all_experiments.py --exp 1
"""

# ── Install deps ─────────────────────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# All required; most pre-installed on Kaggle
deps = ["torch", "torchvision", "matplotlib", "numpy"]
for dep in deps:
    install(dep)

print("✓ Dependencies ready")

# ── Verify GPU ───────────────────────────────────────────────────────────────
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n✓ Setup complete. Run experiments with:")
print("  !python run_all_experiments.py --exp 1   # BDH on STL-10 (~3h)")
print("  !python run_all_experiments.py --exp 2   # ViT baseline (~3h)")
print("  !python run_all_experiments.py --exp 3   # Patch ablation (~3h)")
print("  !python run_all_experiments.py --exp 4   # Label efficiency (~4h)")
print("  !python run_all_experiments.py --exp all # Everything (10-14h)")