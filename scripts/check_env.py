"""环境自检：确认 GPU 可用、各模型能跑多大 batch。"""

import sys
import torch
import torchvision


def main():
    print(f"python  {sys.version.split()[0]}")
    print(f"torch   {torch.__version__}")
    print(f"tv      {torchvision.__version__}")
    print(f"cuda    {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("没检测到 GPU，后面不用测了")
        return
    print(f"device  {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"显存    {total / 1024**3:.1f} GB 总 / {free / 1024**3:.1f} GB 空闲")

    for name in ["resnet18", "resnet50", "mobilenet_v3_large"]:
        model = getattr(torchvision.models, name)(weights="DEFAULT").cuda().eval()
        ok = []
        for bs in [8, 16, 32, 64, 128, 256]:
            try:
                x = torch.randn(bs, 3, 224, 224, device="cuda")
                with torch.no_grad():
                    model(x)
                ok.append(bs)
                del x
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
        print(f"{name:20s} 最大 batch {max(ok) if ok else 0}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
