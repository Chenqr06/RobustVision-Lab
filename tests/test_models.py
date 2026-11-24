import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.base import build_model, pretrained_report, WrappedModel


def test_forward_shape():
    net = build_model("resnet18", pretrained=False).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_stem_adapted():
    net = build_model("resnet18", pretrained=False)
    assert net.conv1.kernel_size == (3, 3)
    assert net.conv1.stride == (1, 1)
    assert isinstance(net.maxpool, torch.nn.Identity)


def test_pretrained_layers_mostly_loaded():
    loaded, total = pretrained_report("resnet18")
    assert loaded >= total - 4, f"只匹配了 {loaded}/{total}，第一层改造不该牵连这么多"


def test_predict_proba_sums_to_one():
    m = WrappedModel(build_model("resnet18", pretrained=False), "resnet18", device="cpu")
    p = m.predict_proba(torch.randn(3, 3, 32, 32))
    assert p.shape == (3, 10)
    assert torch.allclose(p.sum(1), torch.ones(3), atol=1e-5)


def test_eval_mode_deterministic():
    net = build_model("resnet18", pretrained=False).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        a = net(x)
        b = net(x)
    assert torch.equal(a, b)


def test_unknown_model_raises():
    try:
        build_model("vgg16", pretrained=False)
    except KeyError:
        return
    raise AssertionError("未知模型名应该抛 KeyError")
