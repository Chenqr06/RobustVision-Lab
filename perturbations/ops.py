import math

import torch
import torch.nn.functional as F

# 所有扰动吃 [0,1] 的 float 张量 (N,3,32,32)，s=0 必须原样返回


def _check(x: torch.Tensor) -> None:
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"要 (N,3,H,W)，拿到 {tuple(x.shape)}")


def gaussian_noise(x, s, seed=0):
    _check(x)
    if s <= 0:
        return x.clone()
    g = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    return (x + 0.15 * s * noise).clamp(0, 1)


def _gauss1d(k, sigma, device, dtype):
    t = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    w = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    return w / w.sum()


def gaussian_blur(x, s, seed=0):
    _check(x)
    if s <= 0:
        return x.clone()
    k = 2 * round(1 + 3 * s) + 1
    sigma = 0.1 + 2.0 * s
    w = _gauss1d(k, sigma, x.device, x.dtype)
    kx = torch.zeros(3, 1, 1, k, device=x.device, dtype=x.dtype)
    kx[:, 0, 0, :] = w
    ky = torch.zeros(3, 1, k, 1, device=x.device, dtype=x.dtype)
    ky[:, 0, :, 0] = w
    x = F.conv2d(x, kx, padding=(0, k // 2), groups=3)
    x = F.conv2d(x, ky, padding=(k // 2, 0), groups=3)
    return x.clamp(0, 1)


def _line_kernel(k, angle, device, dtype):
    r = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    dist = (yy * math.sin(angle) - xx * math.cos(angle)).abs()
    ker = (dist <= 0.8).to(dtype)
    if ker.sum() == 0:
        ker[k // 2, k // 2] = 1
    return ker / ker.sum()


def motion_blur(x, s, seed=0):
    _check(x)
    if s <= 0:
        return x.clone()
    k = 2 * round(1 + 5 * s) + 1
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(x.shape[0]):
        ang = float(torch.rand(1, generator=g)) * math.pi
        ker = _line_kernel(k, ang, x.device, x.dtype)
        w = ker.expand(3, 1, k, k).contiguous()
        out.append(F.conv2d(x[i:i + 1], w, padding=k // 2, groups=3))
    return torch.cat(out).clamp(0, 1)


def brightness(x, s, seed=0):
    _check(x)
    if s <= 0:
        return x.clone()
    return (x * (1 + 0.8 * s)).clamp(0, 1)


def contrast(x, s, seed=0):
    _check(x)
    if s <= 0:
        return x.clone()
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * (1 - 0.8 * s) + m).clamp(0, 1)


def occlusion(x, s, seed=0):
    _check(x)
    if s <= 0:
        return x.clone()
    side = int(math.floor(x.shape[-1] * math.sqrt(0.05 + 0.45 * s)))
    side = max(1, min(side, x.shape[-1]))
    g = torch.Generator().manual_seed(seed)
    out = x.clone()
    for i in range(x.shape[0]):
        top = int(torch.randint(0, x.shape[-2] - side + 1, (1,), generator=g))
        left = int(torch.randint(0, x.shape[-1] - side + 1, (1,), generator=g))
        out[i, :, top:top + side, left:left + side] = 0.5
    return out


def jpeg(x, s, seed=0):
    _check(x)
    if s <= 0:
        return x.clone()
    from torchvision.io import decode_jpeg, encode_jpeg

    q = int(round(100 - 90 * s))
    u = (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    out = [decode_jpeg(encode_jpeg(u[i], quality=q)).to(x.dtype) / 255.0 for i in range(x.shape[0])]
    return torch.stack(out).to(x.device).clamp(0, 1)


REGISTRY = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "brightness": brightness,
    "contrast": contrast,
    "occlusion": occlusion,
    "jpeg": jpeg,
}


def apply(name: str, x: torch.Tensor, strength: float, seed: int = 0) -> torch.Tensor:
    if name not in REGISTRY:
        raise KeyError(f"没有 {name}，可选 {list(REGISTRY)}")
    return REGISTRY[name](x, strength, seed)
