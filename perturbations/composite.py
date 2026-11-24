from .ops import REGISTRY, apply

COMPOSITES = {
    "noise_blur": ["gaussian_noise", "gaussian_blur"],
    "blur_bright": ["gaussian_blur", "brightness"],
    "noise_occ": ["gaussian_noise", "occlusion"],
    "triple": ["brightness", "contrast", "gaussian_blur"],
}


class Composite:
    def __init__(self, members):
        if isinstance(members, str):
            if members not in COMPOSITES:
                raise KeyError(f"没有 {members}，可选 {list(COMPOSITES)}")
            members = COMPOSITES[members]
        bad = [m for m in members if m not in REGISTRY]
        if bad:
            raise KeyError(f"未知扰动 {bad}")
        self.members = list(members)

    def __call__(self, x, strength, seed=0):
        for i, name in enumerate(self.members):
            x = apply(name, x, strength, seed=seed + i * 1000)
        return x

    def __repr__(self):
        return f"Composite({' -> '.join(self.members)})"


def build(spec) -> Composite:
    return Composite(spec) if not isinstance(spec, Composite) else spec
