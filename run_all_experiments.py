"""
run_all_experiments.py 

Run individual experiments:
  python run_all_experiments.py --exp 1
  python run_all_experiments.py --exp 2
  python run_all_experiments.py --exp 3
  python run_all_experiments.py --exp 4  [--fraction 1.0 --model bdh]

Run everything:
  python run_all_experiments.py --exp all
"""

import os
import sys
import time
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_experiment(name: str, fn, *args, **kwargs):
    """Wrapper: run an experiment, time it, save status."""
    print(f"\n{'#'*70}")
    print(f"# STARTING: {name}")
    print(f"{'#'*70}\n")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\n✓ COMPLETED: {name} in {elapsed/3600:.1f}h ({elapsed:.0f}s)")
        return {"status": "success", "result": result, "elapsed_s": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n✗ FAILED: {name} — {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e), "elapsed_s": elapsed}


def main(args):
    status_log = {}

    # ── Experiment 1: BDH on STL-10 ───────────────────────────────────────
    if args.exp in ("all", "1"):
        from train_bdh_stl10 import main as exp1_main

        class Args1:
            resume = args.resume
        status_log["exp1_bdh_stl10"] = run_experiment(
            "Exp 1: Vision-BDH v2 on STL-10 (patch=8)",
            exp1_main, Args1()
        )

    # ── Experiment 2: ViT-Tiny baseline ───────────────────────────────────
    if args.exp in ("all", "2"):
        from train_vit_stl10 import main as exp2_main

        class Args2:
            resume = args.resume
        status_log["exp2_vit_stl10"] = run_experiment(
            "Exp 2: ViT-Tiny on STL-10 (Baseline)",
            exp2_main, Args2()
        )

    # ── Experiment 3: Patch ablation ──────────────────────────────────────
    if args.exp in ("all", "3"):
        from train_bdh_stl10_ablation_patch import main as exp3_main

        class Args3:
            resume = args.resume
        status_log["exp3_patch_ablation"] = run_experiment(
            "Exp 3: BDH Ablation patch_size=12",
            exp3_main, Args3()
        )

    # ── Experiment 4: Label efficiency ────────────────────────────────────
    if args.exp in ("all", "4"):
        from train_label_efficiency import main as exp4_main

        class Args4:
            fraction = args.fraction
            model = args.model
        status_log["exp4_label_efficiency"] = run_experiment(
            "Exp 4: Label Efficiency (BDH vs ViT @ 10/25/50/100% data)",
            exp4_main, Args4()
        )

    # ── Analysis ──────────────────────────────────────────────────────────
    if args.exp in ("all", "analysis"):
        from analysis.analyze_stl10 import main as analysis_main

        class ArgsA:
            plot = "all"
        status_log["analysis"] = run_experiment(
            "Analysis: Generate all figures",
            analysis_main, ArgsA()
        )

    # Save status
    with open("./experiment_status.json", "w") as f:
        json.dump(status_log, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("  EXPERIMENT STATUS SUMMARY")
    print("=" * 70)
    total_time = 0
    for name, result in status_log.items():
        icon = "✓" if result["status"] == "success" else "✗"
        elapsed_h = result["elapsed_s"] / 3600
        total_time += result["elapsed_s"]
        print(f"  {icon} {name:40s} {elapsed_h:.2f}h")
    print(f"  {'Total':40s} {total_time/3600:.2f}h")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all Vision-BDH STL-10 experiments")
    parser.add_argument(
        "--exp", type=str, default="all",
        choices=["all", "1", "2", "3", "4", "analysis"],
        help="Which experiment to run (default: all)"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--fraction", type=float, default=None,
                        help="For exp 4: data fraction (0.1, 0.25, 0.5, 1.0)")
    parser.add_argument("--model", type=str, default=None, choices=["bdh", "vit"],
                        help="For exp 4: which model to run")
    args = parser.parse_args()
    main(args)