import numpy as np
import torch

from dataio.cifar import denormalize, normalize
from perturbations import REGISTRY, apply
from perturbations.composite import COMPOSITES, Composite

STRENGTHS = [0.2, 0.4, 0.6, 0.8, 1.0]
SINGLE = list(REGISTRY)
FIXED_SEED = 42


def grid_items():
    """clean 一条 + 7 单一 × 5 强度 + 4 复合 × 5 强度。"""
    items = [("clean", None, 0.0)]
    for name in SINGLE:
        items += [(name, name, s) for s in STRENGTHS]
    for name in COMPOSITES:
        items += [(name, name, s) for s in STRENGTHS]
    return items


def _perturb(pert, x, strength):
    if pert in COMPOSITES:
        return Composite(pert)(x, strength, seed=FIXED_SEED)
    return apply(pert, x, strength, seed=FIXED_SEED)


@torch.no_grad()
def run_one(model, loader, pert, strength, seed=FIXED_SEED):
    """扰动在显存里做，别在 CPU 上算完再搬过去。"""
    dev = getattr(model, "device", "cpu")
    ps, ys = [], []
    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        if pert is not None:
            x = normalize(_perturb(pert, denormalize(x), strength))
        ps.append(model.predict_proba(x).float().cpu())
        ys.append(y)
    return torch.cat(ps), torch.cat(ys)


def tag(name, strength):
    return f"{name}_s{strength:.1f}"


def run_grid(model, loader, cache_dir, force=False, log=print):
    """把整个网格的概率存下来，后面算指标不用再过一遍模型。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    items = grid_items()
    for i, (name, pert, s) in enumerate(items, 1):
        path = cache_dir / f"{tag(name, s)}.npz"
        if path.exists() and not force:
            d = np.load(path)
            out[(name, s)] = (torch.from_numpy(d["probs"]), torch.from_numpy(d["labels"]))
            continue
        probs, labels = run_one(model, loader, pert, s)
        np.savez(path, probs=probs.numpy(), labels=labels.numpy())
        out[(name, s)] = (probs, labels)
        log(f"[{i}/{len(items)}] {tag(name, s)}")
    return out
