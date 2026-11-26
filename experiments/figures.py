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


def normalized_damage(grid, model_name, path):
    """每条线都归一化到 [1, 0]：1 是干净准确率，0 是它自己在 s=1.0 的水平。
    这样比的是"掉得多快"，不是"掉多少"，跨扰动才有可比性。"""
    feas = np.array([0.0] + STRENGTHS)
    clean_probs, clean_labels = grid[("clean", 0.0)]
    clean_acc = (clean_probs.argmax(1) == clean_labels).float().mean().item()

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name in SINGLE:
        accs = [clean_acc]
        for s in STRENGTHS:
            probs, labels = grid[(name, s)]
            accs.append((probs.argmax(1) == labels).float().mean().item())
        floor = accs[-1]
        span = clean_acc - floor
        if span < 1e-6:
            continue
        ys = [(a - floor) / span for a in accs]
        ax.plot(feas, ys, marker="o", ms=3, lw=1.4, label=name.replace("_", " "))
    ax.set_xlabel("strength")
    ax.set_ylabel("remaining damage (1 → 0)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{model_name}: how fast each perturbation bites")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    _save(fig, path)


def model_comparison(agg, path):
    """agg 是 aggregate.py 产出的 mean/std 表，一个模型一个子图。"""
    models = sorted(agg["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models) + 1.2, 3.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, models):
        sub = agg[agg["model"] == m]
        runs = int(sub["runs"].max())
        for name in SINGLE:
            s = sub[sub["perturbation"] == name].sort_values("strength")
            if not len(s):
                continue
            ax.plot(s["strength"], s["acc_mean"], marker="o", ms=3, lw=1.3,
                    label=name.replace("_", " "))
            ax.fill_between(s["strength"], s["acc_mean"] - s["acc_std"],
                            s["acc_mean"] + s["acc_std"], alpha=0.15, lw=0)
        ax.set_title(f"{m} ({runs} seed{'s' if runs > 1 else ''})", fontsize=9)
        ax.set_xlabel("strength")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(fontsize=6.5, ncol=2, frameon=False, loc="lower left")
    _save(fig, path)


def retention_bars(agg, path):
    """s=1.0 的相对保持率，横向柱状图，一个扰动两组。"""
    sub = agg[agg.strength == 1.0]
    models = sorted(sub["model"].unique())
    order = (sub[sub.model == models[0]].set_index("perturbation")["acc_rel"]
             .sort_values().index.tolist())
    y = np.arange(len(order))
    h = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for i, m in enumerate(models):
        s = sub[sub.model == m].set_index("perturbation").reindex(order)
        ax.barh(y + (i - (len(models) - 1) / 2) * h, s["acc_rel"], h * 0.92,
                label=m, alpha=0.88)
    ax.set_yticks(y, [o.replace("_", " ") for o in order])
    ax.set_xlabel("accuracy retained at s=1.0  (acc / acc_clean)")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
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
