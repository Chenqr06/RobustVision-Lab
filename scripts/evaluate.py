import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DATA_DIR, RUN_DIR, save_json, stamp
from dataio.cifar import get_loaders
from metrics import accuracy, brier, ece, mce, risk_coverage
from models import load_checkpoint


def collect(model, loader):
    ps, ys = [], []
    for x, y in loader:
        # 指标函数统一在 CPU 上算，省得设备对不上
        ps.append(model.predict_proba(x).cpu())
        ys.append(y)
    return torch.cat(ps), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()

    _, val_loader, test_loader = get_loaders(root=DATA_DIR, num_workers=2)
    loader = test_loader if args.split == "test" else val_loader

    model = load_checkpoint(args.ckpt, args.model)
    probs, y = collect(model, loader)

    rc = risk_coverage(probs, y)
    out = {
        "model": args.model,
        "split": args.split,
        "n_samples": int(y.numel()),
        "accuracy": accuracy(probs, y),
        "ece15": ece(probs, y, 15),
        "ece_adaptive": ece(probs, y, 15, adaptive=True),
        "mce": mce(probs, y, 15),
        "brier": brier(probs, y),
        "aurc": rc["aurc"],
        "risk_at_0.8": rc["risk_at_0.8"],
        "risk_at_0.9": rc["risk_at_0.9"],
        "timestamp": stamp(),
    }
    for k, v in out.items():
        print(f"{k:15s} {v}")

    save_json(out, RUN_DIR / "eval" / f"{args.model}_{args.split}.json")


if __name__ == "__main__":
    main()
