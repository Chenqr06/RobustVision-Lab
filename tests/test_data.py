import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataio.cifar import denormalize, get_datasets, load_npz

# 数据不进仓库，clone 下来没跑过 download_data.py 时这些测试直接跳过
HAS_DATA = (ROOT / "data" / "cifar10.npz").exists()
pytestmark = pytest.mark.skipif(not HAS_DATA, reason="data/cifar10.npz 不存在，先跑 scripts/download_data.py")



def test_npz_shapes():
    tr_x, tr_y, te_x, te_y = load_npz()
    assert tr_x.shape == (50000, 32, 32, 3)
    assert te_x.shape == (10000, 32, 32, 3)
    assert tr_x.dtype == np.uint8
    assert tr_y.min() == 0 and tr_y.max() == 9


def test_class_balance():
    _, tr_y, _, _ = load_npz()
    counts = np.bincount(tr_y, minlength=10)
    assert counts.sum() == 50000
    assert counts.min() >= 4900, f"类别不均衡：{counts}"


def test_split_disjoint():
    train, val, test = get_datasets()
    assert len(train) == 45000 and len(val) == 5000 and len(test) == 10000
    tr_idx = set(train.indices)
    va_idx = set(val.indices)
    assert not (tr_idx & va_idx)
    assert len(tr_idx | va_idx) == 50000


def test_item_shape():
    train, _, _ = get_datasets()
    x, y = train[0]
    assert x.shape == (3, 32, 32)
    assert x.dtype == torch.float32
    assert 0 <= y < 10


def test_denormalize_in_range():
    x = torch.randn(4, 3, 32, 32)
    y = denormalize(x)
    assert y.min() >= 0 and y.max() <= 1
