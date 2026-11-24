"""下载 CIFAR-10 并转成本地 npz。官方源在国内基本下不动，走 hf-mirror。"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
NPZ = DATA / "cifar10.npz"

BASE = "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text"
FILES = {
    "train": f"{BASE}/train-00000-of-00001.parquet",
    "test": f"{BASE}/test-00000-of-00001.parquet",
}


def fetch(name: str, url: str) -> Path:
    dest = RAW / f"{name}.parquet"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"{name} 已存在，跳过下载")
        return dest
    print(f"下载 {name} ...", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done % (10 << 20) < (1 << 20):
                    print(f"  {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
    tmp.rename(dest)
    return dest


def to_arrays(path: Path):
    df = pd.read_parquet(path)
    imgs = np.stack([
        np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB"))
        for b in df["img"]
    ])
    labels = df["label"].to_numpy().astype(np.uint8)
    return imgs, labels


def main():
    if NPZ.exists():
        print(f"{NPZ} 已存在，不用重复转")
        return

    for name, url in FILES.items():
        fetch(name, url)

    print("转换 train ...", flush=True)
    tr_x, tr_y = to_arrays(RAW / "train.parquet")
    print("转换 test ...", flush=True)
    te_x, te_y = to_arrays(RAW / "test.parquet")

    assert tr_x.shape == (50000, 32, 32, 3), tr_x.shape
    assert te_x.shape == (10000, 32, 32, 3), te_x.shape

    np.savez(NPZ, train_images=tr_x, train_labels=tr_y,
             test_images=te_x, test_labels=te_y)
    print(f"完成：train {tr_x.shape} / test {te_x.shape} -> {NPZ} ({NPZ.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
