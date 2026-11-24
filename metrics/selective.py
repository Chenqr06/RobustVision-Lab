import numpy as np
import torch


def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def risk_coverage(probs: torch.Tensor, y: torch.Tensor, grid=None) -> dict:
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    order = torch.argsort(conf, descending=True)
    cum_acc = correct[order].cumsum(0) / torch.arange(1, conf.numel() + 1,
                                                    device=conf.device, dtype=torch.float32)
    n = conf.numel()

    if grid is None:
        grid = np.concatenate([[0.0], np.arange(0.05, 1.0001, 0.05)])
    grid = np.asarray(grid, dtype=float)
    ks = np.clip(np.ceil(grid * n).astype(int), 1, n)
    risk = 1.0 - cum_acc.cpu().numpy()[ks - 1]

    def at(c):
        k = int(np.clip(np.ceil(c * n), 1, n))
        return float(1.0 - cum_acc[k - 1].item())

    return {
        "coverage": grid,
        "risk": risk,
        "aurc": _trapz(risk, grid),
        "risk_at_0.8": at(0.8),
        "risk_at_0.9": at(0.9),
    }


def risk_at_coverage(probs: torch.Tensor, y: torch.Tensor, coverage: float) -> float:
    conf, pred = probs.max(1)
    n = conf.numel()
    k = int(np.clip(np.ceil(coverage * n), 1, n))
    keep = torch.topk(conf, k).indices
    return float(1.0 - (pred[keep] == y[keep]).float().mean().item())
