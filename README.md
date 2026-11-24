# RobustVision Lab

视觉模型鲁棒性、置信度校准与选择性预测的实验平台。

围绕四个问题做一套评估流程，重点在测，不在训：

- RQ1 图像分布偏移下模型性能怎么变
- RQ2 单一扰动 vs 复合扰动的差异
- RQ3 模型置信度能否反映真实正确率
- RQ4 低置信度时拒绝预测能否降低错误风险

> 主仓库：https://github.com/Chenqr06/RobustVision-Lab

细节都在 [PROJECT_SPEC.md](PROJECT_SPEC.md)：数据集划分、扰动参数、指标定义、实验矩阵。
代码里没有规范外的东西。

## 状态

| 模块 | 状态 |
|---|---|
| 规范 PROJECT_SPEC.md | 完成 |
| 数据 / 模型 / 训练 | 完成，resnet18 已训练 |
| 扰动模块（7 单一 + 4 复合） | 完成，40 个测试 |
| 指标模块（Acc / ECE / MCE / Brier / Risk–Coverage） | 完成，10 个测试 |
| 实验 E1–E6 | 未开始 |
| Streamlit 平台 | 未开始 |

## Baseline

ResNet18，CIFAR-10 test 集 10000 张，40 epoch，单种子 42。
数据来自 `outputs/runs/eval/resnet18_test.json`，不是手填的。

| 指标 | 值 |
|---|---|
| Accuracy | 0.9123 |
| ECE (15 bins, equal-width) | 0.0386 |
| ECE (15 bins, adaptive) | 0.0386 |
| MCE | 0.2437 |
| Brier（多分类求和版，范围 [0,2]） | 0.1346 |
| AURC | 0.0122 |
| Risk @ 80% coverage | 0.0194 |
| Risk @ 90% coverage | 0.0446 |

两个 ECE 几乎一样，说明分箱方式在这里影响不大。但 MCE 0.24 挺扎眼——有个箱子
偏了二十多个点。光看 ECE 会以为模型校准得还行，所以三个数一起报。

## 环境

Python 3.13 + PyTorch 2.11 (CUDA 12.8)，RTX 4060 Laptop 8GB。

```bash
python -m venv .venv
.venv/Scripts/activate                # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-dev.txt
```

Windows 上 pypi 默认的 torch wheel 是 CPU 版，必须走 `download.pytorch.org`。

## 跑起来

```bash
python scripts/download_data.py                                    # 下载并转 npz
python scripts/train.py --config configs/resnet18.yaml             # 40 epoch，约 10 分钟
python scripts/evaluate.py --ckpt outputs/checkpoints/resnet18_seed42.pt --model resnet18
pytest                                                             # 61 个测试
```

数据走 hf-mirror 的 parquet 镜像（官方源在国内实测约 90 KB/s，不可用），
详见 `scripts/download_data.py`。

## 目录

```
configs/        训练与实验配置（YAML）
data/           CIFAR-10 npz 与原始 parquet，不进仓库
dataio/         数据集加载与归一化
models/         模型构建与统一 predict_proba 接口
perturbations/  7 种单一扰动 + 4 种复合
metrics/        Accuracy / ECE / MCE / Brier / Risk–Coverage / AURC
experiments/    实验编排（E1–E6）
visualization/  图表
app/            Streamlit 界面
tests/          单元测试
outputs/        实验结果、checkpoint、图片，不进仓库
```

## 数据集

CIFAR-10。数据本身不提交仓库，使用请遵守原始许可条款。
`scripts/download_data.py` 会自动下载并转成本地 npz，与官方图像内容一致。

## 许可证

MIT
