# RobustVision Lab

视觉模型鲁棒性、置信度校准与选择性预测的实验平台。

围绕四个问题搭一套评估流程，不是又一个图片分类 demo：

- RQ1 图像分布偏移下模型性能怎么变
- RQ2 单一扰动 vs 复合扰动的差异
- RQ3 模型置信度能否反映真实正确率
- RQ4 低置信度时拒绝预测能否降低错误风险

> 主仓库：https://github.com/Chenqr06/RobustVision-Lab

## 状态

**Stage 1：设计与规范完成。** 研究问题、数据集划分、扰动参数、指标定义、实验矩阵、
模块接口都已在 [PROJECT_SPEC.md](PROJECT_SPEC.md) 中定死。业务代码尚未开始。

| 项 | 状态 |
|---|---|
| 规范（PROJECT_SPEC.md） | 完成 |
| 数据 / 模型模块 | 未开始 |
| 扰动模块 | 未开始 |
| 指标模块 | 未开始 |
| 实验代码 | 未开始 |
| 可视化平台 | 未开始 |

## 环境

Python 3.13 + PyTorch (CUDA 12.8)，RTX 4060 Laptop 8GB。

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-dev.txt
```

注意 Windows 上 pypi 默认的 torch wheel 是 CPU 版，必须走 `download.pytorch.org`。

## 目录

```
configs/        实验配置（YAML）
data/           CIFAR-10 缓存，不进仓库
models/         模型统一封装
perturbations/  单一扰动与复合扰动
metrics/        Accuracy / ECE / Brier / Risk-Coverage
experiments/    实验脚本与记录
visualization/  图表
app/            Streamlit 界面
tests/          单元测试
outputs/        实验结果与图片，不进仓库
```

## 数据集

CIFAR-10，首次运行自动下载到 `data/`。数据集本身不提交仓库。
使用时请遵守其原始许可条款。

## 许可证

MIT
