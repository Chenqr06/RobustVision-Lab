import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (DATA_DIR, OUT_DIR, RUN_DIR, config_hash, git_commit,
                    save_json, stamp)
from dataio.cifar import get_loaders
from experiments import analysis, figures
from experiments.grid import grid_items, run_grid, tag
from models import load_checkpoint

RES_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--force", action="store_true", help="重算已缓存的概率")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt or OUT_DIR / "checkpoints" / f"{args.model}_seed42.pt")
    if not ckpt.exists():
        raise SystemExit(f"找不到 {ckpt}")

    started = time.time()
    _, _, test_loader = get_loaders(root=DATA_DIR, num_workers=2)
    model = load_checkpoint(ckpt, args.model)

    cache = OUT_DIR / "probs" / args.model
    grid = run_grid(model, test_loader, cache, force=args.force)

    e1 = analysis.e1_table(grid, args.model)
    analysis.write_csv(RES_DIR / args.model / "e1_single.csv", e1, analysis.E1_FIELDS)

    e2 = analysis.e2_table(grid, args.model)
    analysis.write_csv(RES_DIR / args.model / "e2_composite.csv", e2, analysis.E2_FIELDS)

    e4, worst_key = analysis.e4_bins(grid, args.model)
    analysis.write_csv(RES_DIR / args.model / "e4_reliability.csv", e4, analysis.E4_FIELDS)

    e5 = analysis.e5_curves(grid, args.model)
    analysis.write_csv(RES_DIR / args.model / "e5_risk_coverage.csv", e5, analysis.E5_FIELDS)

    e6 = analysis.e6_table(grid, args.model)
    analysis.write_csv(RES_DIR / args.model / "e6_selective.csv", e6, analysis.E6_FIELDS)

    for row in e1:
        rid = f"e1_{args.model}_{row['perturbation']}_s{row['strength']:.1f}"
        save_json({
            "run_id": rid,
            "experiment": "E1",
            "model": args.model,
            "checkpoint": str(ckpt.relative_to(ckpt.parents[2])),
            "dataset": "cifar10",
            "split": "test",
            "n_samples": 10000,
            "perturbation": {"name": row["perturbation"], "strength": row["strength"]},
            "metrics": {k: row[k] for k in ("accuracy", "ece15", "ece_adaptive", "mce",
                                            "brier", "aurc", "risk_at_0.8", "risk_at_0.9")},
            "git_commit": git_commit(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "timestamp": stamp(),
        }, RUN_DIR / "E1" / f"{rid}.json")

    if not args.skip_figures:
        figures.robustness_curve(grid, args.model, FIG_DIR / f"{args.model}_robustness.png")
        figures.composite_bars(e2, args.model, FIG_DIR / f"{args.model}_composite.png")
        figures.reliability_plot(grid, args.model, worst_key, FIG_DIR / f"{args.model}_reliability.png")
        figures.risk_coverage_plot(grid, args.model, FIG_DIR / f"{args.model}_risk_coverage.png")

    print(f"最差条件：{tag(*worst_key)}")
    print(f"E1 {len(e1)} 行 / E2 {len(e2)} 行 / E4 {len(e4)} 行 / E5 {len(e5)} 行 / E6 {len(e6)} 行")
    print(f"耗时 {time.time() - started:.1f}s，结果在 {RES_DIR / args.model}，图在 {FIG_DIR}")


if __name__ == "__main__":
    main()
