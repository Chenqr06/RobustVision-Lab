from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import reliability_bins, risk_coverage

from .grid import COMPOSITES, SINGLE, STRENGTHS

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def robustness_curve(grid, model_name, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    feas = np.array([0.0] + STRENGTHS)
    clean_probs, clean_labels = grid[("clean", 0.0)]
    clean_acc = (clean_probs.argmax(1) == clean_labels).float().mean().item()
    for name in SINGLE:
        ys = []
        for s in STRENGTHS:
            probs, labels = grid[(name, s)]
            ys.append((probs.argmax(1) == labels).float().mean().item())
        ax.plot(feas, [clean_acc] + ys, marker="o", ms=3, lw=1.4,
                label=name.replace("_", " "))
    ax.set_xlabel("strength")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name}: accuracy under single perturbations")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    _save(fig, path)


def composite_bars(e2_rows, model_name, path):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    names = list(dict.fromkeys(r["composite"] for r in e2_rows))
    x = np.arange(len(names))
    w = 0.25
    for i, s in enumerate([0.4, 0.6, 0.8, 1.0]):
        comp = [next(r["composite_acc"] for r in e2_rows if r["composite"] == n and r["strength"] == s) for n in names]
        pred = [next(r["product_of_parts"] for r in e2_rows if r["composite"] == n and r["strength"] == s) for n in names]
        ax.bar(x + (i - 1.5) * w, comp, w * 0.9, label=f"measured s={s}", alpha=0.85)
        ax.plot(x + (i - 1.5) * w, pred, "k_", ms=13, mew=1.8,
                label="product of parts" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name}: composite vs independent product")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, path)


def reliability_plot(grid, model_name, worst_key, path, n_bins=15):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    for key, label, color in ((("clean", 0.0), "clean", "#2b6cb0"),
                              (worst_key, f"{worst_key[0].replace('_', ' ')} s={worst_key[1]}", "#c53030")):
        probs, labels = grid[key]
        rep = reliability_bins(probs, labels, n_bins)
        mask = rep["bin_count"] >= 30
        ax.plot(rep["bin_conf"][mask], rep["bin_acc"][mask], "o-", ms=4, lw=1.2,
                color=color, label=f"{label} (ECE {rep['ece']:.3f})")
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name}: reliability")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _save(fig, path)


def risk_coverage_plot(grid, model_name, path, top_k=3):
    singles = {k: v for k, v in grid.items() if k[0] in SINGLE}
    ranked = sorted(singles.keys(),
                    key=lambda k: (singles[k][0].argmax(1) == singles[k][1]).float().mean().item())
    chosen = [("clean", ("clean", 0.0))] + [(k, k) for k in ranked[:top_k]]

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for label, key in chosen:
        probs, labels = grid[key]
        rc = risk_coverage(probs, labels)
        ax.plot(rc["coverage"], rc["risk"], lw=1.4,
                label=f"{label if label == 'clean' else label[0].replace('_', ' ') + f' s={label[1]}'} (AURC {rc['aurc']:.3f})")
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (1 - accuracy)")
    ax.set_xlim(0, 1)
    ax.set_title(f"{model_name}: risk-coverage")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, path)
