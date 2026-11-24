import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perturbations import COMPOSITES, REGISTRY, Composite, apply

ALL = list(REGISTRY)


def _img(n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, 32, 32, generator=g)


@pytest.mark.parametrize("name", ALL)
def test_identity_at_zero(name):
    x = _img(seed=1)
    assert torch.allclose(apply(name, x, 0.0, seed=0), x, atol=1e-6)


@pytest.mark.parametrize("name", ALL)
def test_shape_and_range(name):
    x = _img()
    y = apply(name, x, 1.0, seed=0)
    assert y.shape == x.shape
    assert y.min() >= 0 and y.max() <= 1
    assert not torch.isnan(y).any()


@pytest.mark.parametrize("name", ALL)
def test_does_not_mutate_input(name):
    x = _img(seed=7)
    before = x.clone()
    apply(name, x, 0.8, seed=0)
    assert torch.equal(x, before)


@pytest.mark.parametrize("name", ALL)
def test_same_seed_same_output(name):
    x = _img()
    a = apply(name, x, 0.6, seed=123)
    b = apply(name, x, 0.6, seed=123)
    assert torch.allclose(a, b)


@pytest.mark.parametrize("name", ALL)
def test_changes_pixels(name):
    x = _img()
    y = apply(name, x, 0.6, seed=3)
    assert (y - x).abs().mean() > 1e-4, f"{name} 在 s=0.6 几乎没改动图像"


def test_stronger_means_bigger_change():
    x = _img(seed=5)
    for name in ["gaussian_noise", "brightness", "contrast"]:
        weak = (apply(name, x, 0.2, seed=0) - x).abs().mean()
        strong = (apply(name, x, 1.0, seed=0) - x).abs().mean()
        assert strong > weak, f"{name} 强度加大后变化反而变小"


def test_occlusion_area_grows_with_strength():
    x = torch.ones(1, 3, 32, 32)
    small = (apply("occlusion", x, 0.2, seed=0) != 1).sum()
    large = (apply("occlusion", x, 1.0, seed=0) != 1).sum()
    assert large > small


def test_composite_order_matters():
    x = _img(seed=9)
    a = Composite(["gaussian_noise", "gaussian_blur"])(x, 0.8, seed=0)
    b = Composite(["gaussian_blur", "gaussian_noise"])(x, 0.8, seed=0)
    assert not torch.allclose(a, b, atol=1e-3)


def test_composite_by_name():
    x = _img()
    y = Composite("triple")(x, 0.5, seed=0)
    assert y.shape == x.shape
    assert len(COMPOSITES["triple"]) == 3


def test_unknown_names_raise():
    with pytest.raises(KeyError):
        apply("nope", _img(), 0.5)
    with pytest.raises(KeyError):
        Composite("nope")
