import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
RUN_DIR = OUT_DIR / "runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def git_commit() -> str:
    head = ROOT / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    ref = head.read_text().strip()
    if ref.startswith("ref:"):
        p = ROOT / ".git" / ref.split(" ", 1)[1]
        return p.read_text().strip()[:7] if p.exists() else "unknown"
    return ref[:7]


def save_json(obj: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
