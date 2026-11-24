import torch
import torch.nn.functional as F


def _logits(probs: torch.Tensor) -> torch.Tensor:
    # log softmax 去掉常数项后与原始 logits 等价，直接拿来做温度缩放没问题
    return torch.log(probs.clamp_min(1e-8))


def fit_temperature(probs: torch.Tensor, y: torch.Tensor,
                    lo: float = 0.5, hi: float = 3.0, steps: int = 251) -> float:
    """在验证集上搜一个温度，让 NLL 最小。网格搜索比 LBFGS 稳，代价可忽略。"""
    logits = _logits(probs)
    best_t, best_nll = 1.0, float("inf")
    for t in torch.linspace(lo, hi, steps):
        nll = F.cross_entropy(logits / t, y).item()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def apply_temperature(probs: torch.Tensor, t: float) -> torch.Tensor:
    return torch.softmax(_logits(probs) / t, dim=1)


def nll(probs: torch.Tensor, y: torch.Tensor) -> float:
    return F.cross_entropy(_logits(probs), y).item()
