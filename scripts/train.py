import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (CKPT_DIR, DATA_DIR, RUN_DIR, config_hash, git_commit,
                    load_config, save_json, set_seed, stamp)
from dataio.cifar import get_loaders
from models.base import build_model, pretrained_report


def evaluate(net, loader, device):
    net.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=True):
                out = net(x)
                loss_sum += F.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total, loss_sum / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None,
                    help="覆盖 dataloader worker 数，崩溃排查时设 0")
    ap.add_argument("--resume", action="store_true",
                    help="从 _last.pt 继续，优化器与调度器状态一并恢复")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.workers is not None:
        cfg["num_workers"] = args.workers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])

    train_loader, val_loader, _ = get_loaders(
        root=DATA_DIR, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"]
    )

    net = build_model(cfg["name"], pretrained=True).to(device)
    loaded, total = pretrained_report(cfg["name"])
    print(f"预训练权重 {loaded}/{total} 层匹配", flush=True)

    opt = SGD(net.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
              weight_decay=cfg["weight_decay"], nesterov=True)
    sched = CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    ckpt_path = CKPT_DIR / f"{cfg['name']}_seed{cfg['seed']}.pt"
    last_path = CKPT_DIR / f"{cfg['name']}_seed{cfg['seed']}_last.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    start_epoch = 1
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        net.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_acc = state["best_acc"]
        print(f"续训：从 epoch {start_epoch} 开始，best {best_acc:.4f}", flush=True)

    started = time.time()
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        net.train()
        run_loss = 0.0
        seen = 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg["amp"]):
                out = net(x)
                loss = F.cross_entropy(out, y, label_smoothing=cfg["label_smoothing"])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * y.numel()
            seen += y.numel()
        sched.step()

        val_acc, val_loss = evaluate(net, val_loader, device)
        flag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": net.state_dict(),
                "model": cfg["name"],
                "num_classes": 10,
                "epoch": epoch,
                "val_acc": val_acc,
                "config": cfg,
            }, ckpt_path)
            flag = " *"

        # 每轮都落盘，崩了最多丢一个 epoch
        torch.save({
            "state_dict": net.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }, last_path)

        print(f"[{epoch:02d}/{cfg['epochs']}] loss {run_loss / seen:.4f} | "
              f"val {val_acc:.4f} | {time.time() - t0:.1f}s{flag}", flush=True)

    record = {
        "run_id": f"train_{cfg['name']}_seed{cfg['seed']}",
        "experiment": "train",
        "model": cfg["name"],
        "checkpoint": str(ckpt_path.relative_to(Path(__file__).resolve().parent.parent)),
        "best_val_acc": best_acc,
        "epochs": cfg["epochs"],
        "resumed_from": start_epoch - 1 if start_epoch > 1 else None,
        "pretrained_layers": f"{loaded}/{total}",
        "config": cfg,
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": stamp(),
        "duration_s": round(time.time() - started, 1),
    }
    save_json(record, RUN_DIR / "train" / f"{record['run_id']}.json")
    print(f"best val acc {best_acc:.4f} | {record['duration_s']}s | {ckpt_path}")


if __name__ == "__main__":
    main()
