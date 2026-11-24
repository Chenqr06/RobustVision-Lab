import torch
import torchvision
from torch import nn

ARCHS = {
    "resnet18": torchvision.models.resnet18,
    "resnet50": torchvision.models.resnet50,
}


def _cifar_stem(m):
    # 32x32 进来的图撑不住 7x7 stride2 + maxpool，第一层换小、去掉池化
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def build_model(name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    if name not in ARCHS:
        raise KeyError(f"未知模型 {name}，可选 {list(ARCHS)}")
    weights = "DEFAULT" if pretrained else None
    model = ARCHS[name](weights=weights)
    model = _cifar_stem(model)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def pretrained_report(name: str, num_classes: int = 10) -> tuple[int, int]:
    """返回 (用上预训练的层数, 总层数)，写进实验记录用。"""
    ref = ARCHS[name](weights="DEFAULT").state_dict()
    own = build_model(name, num_classes).state_dict()
    loaded = sum(1 for k, v in ref.items() if k in own and own[k].shape == v.shape)
    return loaded, len(own)


class WrappedModel:
    """统一接口，测指标的时候只认 predict_proba。"""

    def __init__(self, net: nn.Module, name: str, device: str = "cuda"):
        self.net = net.to(device).eval()
        self.name = name
        self.device = device
        self.num_classes = net.fc.out_features

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x.to(self.device)), dim=1).float()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x).argmax(dim=1)


def load_checkpoint(path, name: str, device: str = "cuda") -> WrappedModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = build_model(name, num_classes=ckpt.get("num_classes", 10), pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    return WrappedModel(net, name, device)
