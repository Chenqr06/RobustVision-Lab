import numpy as np
import torch


def _bin_index(conf: torch.Tensor, n_bins: int, adaptive: bool) -> torch.Tensor:
    n = conf.numel()
    if not adaptive:
        # 右端点归最后一箱，conf=1.0 不能越界
        return torch.clamp((conf * n_bins).long(), max=n_bins - 1)
    order = torch.argsort(conf, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(n, device=conf.device)
    return torch.clamp(rank * n_bins // n, max=n_bins - 1)


def reliability_bins(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15,
                     adaptive: bool = False) -> dict:
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    idx = _bin_index(conf, n_bins, adaptive)

    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    for m in range(n_bins):
        mask = idx == m
        n_m = int(mask.sum())
        counts[m] = n_m
        if n_m:
            accs[m] = correct[mask].mean().item()
            confs[m] = conf[mask].mean().item()

    n = conf.numel()
    gaps = np.abs(accs - confs)
    weights = counts / max(n, 1)
    nonempty = counts > 0
    return {
        "bin_acc": accs,
        "bin_conf": confs,
        "bin_count": counts,
        "ece": float((weights * gaps).sum()),
        "mce": float(gaps[nonempty].max()) if nonempty.any() else 0.0,
        "n": n,
    }


def ece(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15, adaptive: bool = False) -> float:
    return reliability_bins(probs, y, n_bins, adaptive)["ece"]


def mce(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    return reliability_bins(probs, y, n_bins)["mce"]


def brier(probs: torch.Tensor, y: torch.Tensor) -> float:
    # 多分类求和版，范围 [0,2]，别跟二分类的 [0,1] 混
    onehot = torch.nn.functional.one_hot(y, probs.shape[1]).float()
    return ((probs - onehot) ** 2).sum(1).mean().item()
