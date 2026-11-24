import numpy as np
import torch


def accuracy(probs: torch.Tensor, y: torch.Tensor) -> float:
    return (probs.argmax(1) == y).float().mean().item()


def confusion_matrix(probs: torch.Tensor, y: torch.Tensor, k: int = 10) -> np.ndarray:
    pred = probs.argmax(1).cpu().numpy()
    y = y.cpu().numpy()
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def per_class_accuracy(probs: torch.Tensor, y: torch.Tensor, k: int = 10) -> np.ndarray:
    cm = confusion_matrix(probs, y, k)
    total = cm.sum(1)
    return np.divide(cm.diagonal(), total, out=np.full(k, np.nan), where=total > 0)
