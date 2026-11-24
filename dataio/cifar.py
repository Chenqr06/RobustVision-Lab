from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

TRAIN_SIZE = 45000
SPLIT_SEED = 42
NPZ_NAME = "cifar10.npz"

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class CifarArray(Dataset):
    """直接读 npz 里的 uint8 数组，绕开 torchvision 的下载和 pickle 解码。"""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])


def load_npz(root="data"):
    path = Path(root) / NPZ_NAME
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，先跑：python scripts/download_data.py")
    d = np.load(path)
    return d["train_images"], d["train_labels"], d["test_images"], d["test_labels"]


def get_datasets(root="data"):
    tr_x, tr_y, te_x, te_y = load_npz(root)

    train_ds = CifarArray(tr_x, tr_y, train_transform())
    val_ds = CifarArray(tr_x, tr_y, eval_transform())
    test_ds = CifarArray(te_x, te_y, eval_transform())

    g = torch.Generator().manual_seed(SPLIT_SEED)
    idx = torch.randperm(len(tr_y), generator=g)
    return (
        Subset(train_ds, idx[:TRAIN_SIZE].tolist()),
        Subset(val_ds, idx[TRAIN_SIZE:].tolist()),
        test_ds,
    )


def get_loaders(root="data", batch_size=128, num_workers=4):
    train, val, test = get_datasets(root)
    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, **kw),
        DataLoader(val, batch_size=256, shuffle=False, **kw),
        DataLoader(test, batch_size=256, shuffle=False, **kw),
    )
