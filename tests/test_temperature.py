import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import ece
from metrics.temperature import apply_temperature, fit_temperature, nll

K = 10


def _overconfident(n=400, scale=4.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, K, (n,), generator=g)
    logits = torch.randn(n, K, generator=g) * scale
    logits[torch.arange(n), y] += 0.5          # 稍微偏向正确类，让 acc 不至于全是 1/10
    return torch.softmax(logits, 1), y


def test_temperature_one_is_identity():
    probs, _ = _overconfident()
    assert torch.allclose(apply_temperature(probs, 1.0), probs, atol=1e-6)


def test_fitting_reduces_nll_on_calibration_set():
    probs, y = _overconfident()
    t = fit_temperature(probs, y)
    assert nll(apply_temperature(probs, t), y) <= nll(probs, y) + 1e-6


def test_overconfident_model_gets_temperature_above_one():
    probs, y = _overconfident(scale=6.0)
    t = fit_temperature(probs, y)
    assert t > 1.0, f"过度自信的模型温度应大于 1，实际 {t}"


def test_temperature_does_not_change_predictions():
    probs, _ = _overconfident()
    t = fit_temperature(probs, _overconfident()[1])
    assert torch.equal(probs.argmax(1), apply_temperature(probs, t).argmax(1))


def test_temperature_reduces_ece_on_overconfident_model():
    probs, y = _overconfident(scale=6.0)
    t = fit_temperature(probs, y)
    assert ece(apply_temperature(probs, t), y, 15) < ece(probs, y, 15)
