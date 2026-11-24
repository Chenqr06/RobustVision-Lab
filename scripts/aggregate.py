"""把多个 seed / 多个模型的结果合成主表。"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import OUT_DIR
from experiments import figures

RES = OUT_DIR / "results"
SUM = RES / "summary"


def split_run(name: str):
    model, seed = name.rsplit("_seed", 1)
    return model, int(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    runs = sorted(d for d in RES.iterdir() if d.is_dir() and (d / "e1_single.csv").exists())
    if not runs:
        raise SystemExit("results 下没有可用的 run")

    SUM.mkdir(parents=True, exist_ok=True)

    e1_all, clean_all, e7_all = [], [], []
    for d in runs:
        model, seed = split_run(d.name)
        e1 = pd.read_csv(d / "e1_single.csv")
        e1["run"], e1["seed"] = d.name, seed
        e1_all.append(e1)
        c = pd.read_csv(d / "e0_clean.csv")
        c["run"] = d.name
        clean_all.append(c)
        t = pd.read_csv(d / "e7_temperature.csv")
        t["run"], t["seed"] = d.name, seed
        e7_all.append(t)

    e1_all = pd.concat(e1_all, ignore_index=True)
    clean_all = pd.concat(clean_all, ignore_index=True)
    e7_all = pd.concat(e7_all, ignore_index=True)

    clean_all.to_csv(SUM / "clean_by_run.csv", index=False)

    agg = (e1_all.groupby(["model", "perturbation", "strength"])
           .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                ece_mean=("ece15", "mean"), runs=("run", "nunique"))
           .reset_index())
    agg["acc_std"] = agg["acc_std"].fillna(0.0)
    agg.to_csv(SUM / "e1_mean_std.csv", index=False)

    clean_sum = (clean_all.groupby("model")
                 .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                      ece_mean=("ece15", "mean"), ece_std=("ece15", "std"),
                      aurc_mean=("aurc", "mean"), brier_mean=("brier", "mean"),
                      runs=("run", "nunique"))
                 .reset_index())
    clean_sum.to_csv(SUM / "clean_summary.csv", index=False)

    temp_sum = (e7_all.groupby(["model", "condition"])
                .agg(temp=("temperature", "mean"),
                     ece_before=("ece_before", "mean"), ece_after=("ece_after", "mean"))
                .reset_index())
    temp_sum.to_csv(SUM / "temperature_summary.csv", index=False)

    print("== 干净数据（多种子）==")
    print(clean_sum.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("== 各扰动 s=1.0 ==")
    tail = agg[agg.strength == 1.0].pivot_table(index="perturbation", columns="model",
                                                values="acc_mean")
    print(tail.to_string(float_format=lambda v: f"{v:.4f}"))

    if not args.no_figures:
        fig_dir = OUT_DIR / "figures"
        figures.model_comparison(agg, fig_dir / "model_comparison.png")
        print(f"\n图：{fig_dir / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
