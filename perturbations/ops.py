import math

import torch
import torch.nn.functional as F

# 所有扰动吃 [0,1] 的 float 张量 (N,3,32,32)，s=0 必须原样返回
#
# 强度不是线性公式，而是一张参数表：s 从 0.2 到 1.0 对应 5 个标定过的档位，
# 中间线性插值。这样每种扰动在 s=1.0 时破坏力大致相当，跨扰动才能横向比。

ANCHORS = [0.2, 0.4, 0.6, 0.8, 1.0]

NOISE_SIGMA = [0.03, 0.07, 0.12, 0.18, 0.26]
BLUR_K = [3, 5, 7, 9, 11]
BLUR_SIGMA = [0.5, 0.8, 1.2, 1.6, 2.0]
MOTION_K = [3, 5, 7, 9, 11]
BRIGHT_FACTOR = [0.94, 0.82, 0.62, 0.38, 0.16]
CONTRAST_FACTOR = [0.90, 0.74, 0.55, 0.33, 0.12]
OCC_AREA = [0.03, 0.08, 0.16, 0.28, 0.44]
JPEG_QUALITY = [85, 60, 35, 18, 8]


def _check(x: torch.Tensor) -> None:
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"要 (N,3,H,W)，拿到 {tuple(x.shape)}")


def _lerp(table, s):
    if s <= ANCHORS[0]:
        return float(table[0])
    if s >= ANCHORS[-1]:
        return float(table[-1])
    for i in range(len(ANCHORS) - 1):
        if ANCHORS[i] <= s <= ANCHORS[i + 1]:
            t = (s - ANCHORS[i]) / (ANCHORS[i + 1] - ANCHORS[i])
            return float(table[i] + t * (table[i + 1] - table[i]))
    return float(table[-1])


def gaussian_noise(x, s, seed=0):
    _check(x)
    if s <= 0:
        return x.clone()
    g = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    return (x + _lerp(NOISE_SIGMA, s) * noise).clamp(0, 1)


def _gauss1d(k, sigma, device, dtype):
    t = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    w = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    return w / w.sum()


def gaussian_blur(x, s, seed=0):
    _check(x)
    if s <= 0:
        return x.clone()
    k = int(_lerp(BLUR_K, s))
    k += 1 - k % 2                      # 核必须是奇数
    sigma = _lerp(BLUR_SIGMA, s)
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
    k = int(_lerp(MOTION_K, s))
    k += 1 - k % 2
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
    return (x * _lerp(BRIGHT_FACTOR, s)).clamp(0, 1)


def contrast(x, s, seed=0):
    _check(x)
    if s <= 0:
        return x.clone()
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * _lerp(CONTRAST_FACTOR, s) + m).clamp(0, 1)


def occlusion(x, s, seed=0):
    _check(x)
    if s <= 0:
        return x.clone()
    area = _lerp(OCC_AREA, s)
    side = int(round(x.shape[-1] * math.sqrt(area)))
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

    q = int(round(_lerp(JPEG_QUALITY, s)))
    q = max(1, min(q, 100))
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
