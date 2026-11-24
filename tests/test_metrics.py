import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import (accuracy, brier, ece, mce, per_class_accuracy,
                     reliability_bins, risk_coverage)

K = 10


def make_probs(confs, preds, k=K):
    """按给定的 top-1 置信度和预测类别造一个合法的概率张量。"""
    n = len(confs)
    p = torch.full((n, k), 0.0)
    for i, (c, pred) in enumerate(zip(confs, preds)):
        rest = (1 - c) / (k - 1)
        p[i] = rest
        p[i, pred] = c
    return p


def test_ece_perfectly_calibrated_is_zero():
    # 分箱里置信度均值恰好等于正确率
    confs = [0.9, 0.9, 0.9, 0.9]
    preds = [0, 1, 2, 3]
    y = torch.tensor([0, 1, 2, 9])           # 3/4 正确，置信 0.9 不对齐
    assert ece(make_probs(confs, preds), y, n_bins=1) == pytest.approx(0.15, abs=1e-5)

    confs = [0.75] * 4
    y = torch.tensor([0, 1, 2, 9])           # 3/4 = 0.75，完全对齐
    assert ece(make_probs(confs, preds), y, n_bins=1) == pytest.approx(0.0, abs=1e-5)


def test_ece_all_wrong_high_confidence_is_one():
    probs = make_probs([1.0] * 5, [0] * 5)
    y = torch.tensor([1, 2, 3, 4, 5])
    assert ece(probs, y, n_bins=15) == pytest.approx(1.0, abs=1e-5)


def test_confidence_one_goes_to_last_bin():
    probs = make_probs([1.0] * 3, [0, 1, 2])
    y = torch.tensor([0, 1, 2])
    rep = reliability_bins(probs, y, n_bins=15)
    assert rep["bin_count"][-1] == 3
    assert rep["bin_count"][:-1].sum() == 0


def test_hand_computed_ece():
    # conf [0.9, 0.8, 0.2, 0.1]，correct [1, 0, 1, 0]，2 箱
    # bin1: conf 均值 0.85, acc 0.5 -> 0.35 ; bin0: 0.15 vs 0.5 -> 0.35
    probs = make_probs([0.9, 0.8, 0.2, 0.1], [0, 0, 0, 0])
    y = torch.tensor([0, 5, 0, 7])
    assert ece(probs, y, n_bins=2) == pytest.approx(0.35, abs=1e-6)


def test_mce_at_least_ece():
    g = torch.Generator().manual_seed(0)
    probs = torch.softmax(torch.randn(200, K, generator=g) * 2, 1)
    y = torch.randint(0, K, (200,), generator=g)
    assert mce(probs, y, 15) >= ece(probs, y, 15) - 1e-9


def test_adaptive_ece_runs_and_is_sane():
    g = torch.Generator().manual_seed(1)
    probs = torch.softmax(torch.randn(500, K, generator=g) * 3, 1)
    y = torch.randint(0, K, (500,), generator=g)
    eq = ece(probs, y, 10, adaptive=True)
    rep = reliability_bins(probs, y, 10, adaptive=True)
    assert 0 <= eq <= 1
    assert rep["bin_count"].sum() == 500
    assert rep["bin_count"].max() - rep["bin_count"].min() <= 1


def test_brier_perfect_and_uniform():
    onehot = torch.eye(K)
    y = torch.arange(K)
    assert brier(onehot, y) == pytest.approx(0.0, abs=1e-6)

    uniform = torch.full((K, K), 1 / K)
    assert brier(uniform, y) == pytest.approx(0.9, abs=1e-6)


def test_risk_at_full_coverage_equals_error():
    g = torch.Generator().manual_seed(2)
    probs = torch.softmax(torch.randn(100, K, generator=g) * 2, 1)
    y = torch.randint(0, K, (100,), generator=g)
    rc = risk_coverage(probs, y)
    assert rc["risk"][-1] == pytest.approx(1 - accuracy(probs, y), abs=1e-6)


def test_aurc_prefers_correct_ranking():
    n = 200
    y = torch.randint(0, K, (n,), generator=(g := torch.Generator().manual_seed(3)))

    # 好的排序：正确样本的置信度高
    good_conf = torch.linspace(1.0, 0.5, n)
    preds = y.clone()
    preds[: n // 2] = (y[: n // 2] + 1) % K        # 一半是错的
    order = torch.argsort(good_conf)
    preds = preds[order]

    probs_good = make_probs(torch.sort(good_conf).values.tolist(), preds.tolist())
    probs_bad = make_probs(torch.sort(good_conf, descending=True).values.tolist(), preds.tolist())

    assert risk_coverage(probs_good, y)["aurc"] < risk_coverage(probs_bad, y)["aurc"]


def test_per_class_accuracy_shape():
    probs = torch.softmax(torch.randn(50, K, generator=torch.Generator().manual_seed(4)), 1)
    y = torch.randint(0, K, (50,), generator=torch.Generator().manual_seed(5))
    acc = per_class_accuracy(probs, y)
    assert acc.shape == (K,)
    assert np.all((acc >= 0) & (acc <= 1) | np.isnan(acc))
