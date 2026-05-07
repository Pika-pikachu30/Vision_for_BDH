# Vision-BDH Beyond 32×32: STL-10 Experiments

Extends [Pika 2025 (takzen/vision-bdh)](https://github.com/takzen/vision-bdh) to STL-10 (96×96, 5000 samples).

## Architecture change from CIFAR → STL-10

```python
# CIFAR-10 (Pika 2025)
VisionBDHv2(img_size=32, patch_size=4)  # 64 tokens

# STL-10 (Ours) — 2 lines changed
VisionBDHv2(img_size=96, patch_size=8)  # 144 tokens
```

## Experiments

| # | Script | Purpose |
|---|--------|---------|
| 1 | `train_bdh_stl10.py` | Vision-BDH v2 on STL-10 (main result) |
| 2 | `train_vit_stl10.py` | ViT-Tiny baseline |
| 3 | `train_bdh_stl10_ablation_patch.py` | Ablation: patch_size=12 |
| 4 | `train_label_efficiency.py` | accuracy vs data fraction |

## Quick start

```bash
# 1. Verify setup
python verify_setup.py

# 2. Run all experiments (10-14h on T4 GPU)
python run_all_experiments.py --exp all

# Or run individually
python run_all_experiments.py --exp 1   # BDH (~3h)
python run_all_experiments.py --exp 2   # ViT (~3h)
python run_all_experiments.py --exp 3   # Patch ablation (~3h)
python run_all_experiments.py --exp 4   # Label efficiency (~4h)

# 3. Generate figures
python analysis/analyze_stl10.py

# 4. Generate attention visualizations (after experiments 1+2)
python interpretability/visualize_attention_stl10.py
```

## Project structure

```
vision-bdh-stl10/
├── models/
│   ├── bdh.py                    # BDH config, attention, block
│   ├── vision_bdh_v2.py          # VisionBDHv2 (flexible img_size)
│   └── vit.py                    # ViT-Tiny baseline
├── data_stl10.py                 # STL-10 data loading + augmentation
├── utils.py                      # Shared training utilities
├── train_bdh_stl10.py            # Exp 1: BDH on STL-10
├── train_vit_stl10.py            # Exp 2: ViT baseline
├── train_bdh_stl10_ablation_patch.py  # Exp 3: patch ablation
├── train_label_efficiency.py     # Exp 4: label efficiency
├── run_all_experiments.py        # Master runner
├── verify_setup.py               # Pre-flight check
├── analysis/
│   └── analyze_stl10.py         # All figures
├── interpretability/
│   └── visualize_attention_stl10.py  # Attention maps
└── scripts/
    └── kaggle_setup.py           # Kaggle environment setup
```

## Key Results

### Main Comparison (full STL-10, 50 epochs)
| Model       | Test Acc | Params | 
|-------------|----------|--------|
| ViT-Tiny    | 56.69%   | 5.4M   |
| Vision-BDH (ours) | 53.04% | 3.2M  |

### Attention Mechanism Ablation
| Variant            | Test Acc |
|--------------------|----------|
| BDH (no modification) | 51.54% |
| + Root-N scaling   | 51.95%   |
| + Gating only      | 53.04%   |
| + Both             | 51.95%   |

### Label Efficiency (BDH vs ViT across data fractions)
| Data Fraction | BDH Val | BDH Test | ViT Val | ViT Test | Winner (Test) |
|---------------|---------|----------|---------|----------|---------------|
| 10% (450 samples) | 40.00% | **37.24%** | 37.40% | 34.65% | **BDH** |
| 25% (1120 samples) | 42.80% | 41.67% | 46.40% | **42.36%** | ViT |
| 50% (2250 samples) | 51.00% | 47.69% | 52.40% | **50.80%** | ViT |

**Finding:** BDH outperforms ViT-Tiny at low data despite 40% fewer parameters,
suggesting parameter efficiency acts as implicit regularization under data scarcity.


## Citation
If you use this work, please cite:
```bitex
@misc{aarsh2026visionbdhstl10,
  author = {Aarsh Verma},
  title  = {Vision-BDH Beyond 32x32: STL-10 Experiments},
  year   = {2026},
  url    = {https://github.com/Pika-pikachu30/Vision_for_BDH}
}
```

```bibtex
@software{pika2025visionbdh,
  author = {Krzysztof Pika},
  title = {Vision-BDH: Adapting Baby Dragon Hatchling for Computer Vision},
  year = {2025},
  url = {https://github.com/takzen/vision-bdh},
}
```
