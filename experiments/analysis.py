import csv
from pathlib import Path

import numpy as np
import torch

from metrics import accuracy, brier, ece, mce, reliability_bins, risk_coverage

from .grid import COMPOSITES, SINGLE, STRENGTHS


def summarize(probs, labels):
    rc = risk_coverage(probs, labels)
    return {
        "accuracy": accuracy(probs, labels),
        "ece15": ece(probs, labels, 15),
        "ece_adaptive": ece(probs, labels, 15, adaptive=True),
        "mce": mce(probs, labels, 15),
        "brier": brier(probs, labels),
        "aurc": rc["aurc"],
        "risk_at_0.8": rc["risk_at_0.8"],
        "risk_at_0.9": rc["risk_at_0.9"],
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in fields} for r in rows])


def e1_table(grid, model_name):
    rows = []
    for name in SINGLE:
        for s in STRENGTHS:
            probs, labels = grid[(name, s)]
            r = summarize(probs, labels)
            rows.append({"model": model_name, "perturbation": name, "strength": s, **r})
    return rows


def e2_table(grid, model_name):
    rows = []
    for name in COMPOSITES:
        members = COMPOSITES[name]
        for s in STRENGTHS:
            comp_probs, labels = grid[(name, s)]
            comp_acc = accuracy(comp_probs, labels)
            parts = [accuracy(*grid[(m, s)]) for m in members]
            predicted = float(np.prod(parts))
            rows.append({
                "model": model_name,
                "composite": name,
                "members": "+".join(members),
                "strength": s,
                "composite_acc": comp_acc,
                "product_of_parts": predicted,
                "interaction": comp_acc - predicted,
            })
    return rows


def e4_bins(grid, model_name, n_bins=15):
    """干净数据 + 准确率最低的扰动条件，各出一份分箱统计。"""
    clean_probs, labels = grid[("clean", 0.0)]
    clean = reliability_bins(clean_probs, labels, n_bins)

    worst = min(((k, accuracy(*v)) for k, v in grid.items() if k[0] in SINGLE),
                key=lambda kv: kv[1])
    worst_probs, worst_labels = grid[worst[0]]
    bad = reliability_bins(worst_probs, worst_labels, n_bins)

    rows = []
    for cond, rep in (("clean", clean), (f"{worst[0][0]}_s{worst[0][1]:.1f}", bad)):
        for m in range(n_bins):
            rows.append({
                "model": model_name,
                "condition": cond,
                "bin": m,
                "count": int(rep["bin_count"][m]),
                "bin_conf": round(float(rep["bin_conf"][m]), 6),
                "bin_acc": round(float(rep["bin_acc"][m]), 6),
            })
    return rows, worst[0]


def e5_curves(grid, model_name, top_k=3):
    singles = {k: v for k, v in grid.items() if k[0] in SINGLE}
    ranked = sorted(singles.keys(), key=lambda k: accuracy(*singles[k]))
    chosen = [("clean", ("clean", 0.0))] + [(k, k) for k in ranked[:top_k]]

    rows = []
    for label, key in chosen:
        probs, labels = grid[key]
        rc = risk_coverage(probs, labels)
        for c, r in zip(rc["coverage"], rc["risk"]):
            rows.append({
                "model": model_name,
                "condition": label,
                "coverage": round(float(c), 3),
                "risk": round(float(r), 6),
                "aurc": round(rc["aurc"], 6),
            })
    return rows


def e6_table(grid, model_name, strength=0.6):
    rows = []
    for name in SINGLE + list(COMPOSITES):
        key = (name, strength)
        if key not in grid:
            continue
        probs, labels = grid[key]
        rc = risk_coverage(probs, labels)
        rows.append({
            "model": model_name,
            "perturbation": name,
            "strength": strength,
            "accuracy": accuracy(probs, labels),
            "risk_at_0.8": rc["risk_at_0.8"],
            "risk_at_0.9": rc["risk_at_0.9"],
            "aurc": rc["aurc"],
        })
    return rows


def e7_temperature(grid, model_name, val_probs, val_labels):
    """在验证集上拟合一个温度，看它对 test 各条件的校准能修好多少。"""
    from metrics.temperature import apply_temperature, fit_temperature

    t = fit_temperature(val_probs, val_labels)
    rows = [("clean", 0.0)] + [(n, 1.0) for n in SINGLE + list(COMPOSITES)]
    out = []
    for key in rows:
        if key not in grid:
            continue
        probs, labels = grid[key]
        name = key[0] if key[0] == "clean" else f"{key[0]}_s{key[1]:.1f}"
        out.append({
            "model": model_name,
            "condition": name,
            "temperature": round(t, 4),
            "accuracy": accuracy(probs, labels),
            "ece_before": ece(probs, labels, 15),
            "ece_after": ece(apply_temperature(probs, t), labels, 15),
        })
    return out, t


CLEAN_FIELDS = ["model", "seed", "accuracy", "ece15", "ece_adaptive", "mce",
                "brier", "aurc", "risk_at_0.8", "risk_at_0.9"]
E1_FIELDS = ["model", "perturbation", "strength", "accuracy", "ece15", "ece_adaptive",
             "mce", "brier", "aurc", "risk_at_0.8", "risk_at_0.9"]
E2_FIELDS = ["model", "composite", "members", "strength", "composite_acc",
             "product_of_parts", "interaction"]
E4_FIELDS = ["model", "condition", "bin", "count", "bin_conf", "bin_acc"]
E5_FIELDS = ["model", "condition", "coverage", "risk", "aurc"]
E6_FIELDS = ["model", "perturbation", "strength", "accuracy", "risk_at_0.8",
             "risk_at_0.9", "aurc"]
E7_FIELDS = ["model", "condition", "temperature", "accuracy", "ece_before", "ece_after"]
