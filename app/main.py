import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataio.cifar import CLASSES, load_npz
from perturbations import REGISTRY, Composite, apply
from perturbations.composite import COMPOSITES

OUT = ROOT / "outputs"
RES = OUT / "results"
FIG = OUT / "figures"

st.set_page_config(page_title="RobustVision Lab", page_icon="◼", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1150px;}
      div[data-testid="stMetricValue"] {font-size: 1.5rem;}
      h1 {font-size: 1.7rem; margin-bottom: .2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def available_models() -> list[str]:
    if not RES.exists():
        return []
    return sorted([d.name for d in RES.iterdir() if (d / "e1_single.csv").exists()])


@st.cache_resource
def test_images(n: int = 8):
    try:
        _, _, te_x, te_y = load_npz(ROOT / "data")
    except FileNotFoundError:
        return None, None
    return torch.from_numpy(te_x[:n]), torch.from_numpy(te_y[:n]).long()


def to_pil(x: torch.Tensor, size: int = 128) -> Image.Image:
    arr = (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.NEAREST)


def show_png(path: Path, caption: str = ""):
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"还没生成 {path.name}，先跑 scripts/run_experiment.py")


def fig_path(run: str, kind: str) -> Path:
    # run 目录叫 resnet18_seed42，图叫 resnet18_s42_*.png
    return FIG / f"{run.replace('_seed', '_s')}_{kind}.png"


def run_label(run: str) -> str:
    if "_seed" not in run:
        return run
    m, s = run.rsplit("_seed", 1)
    return f"{m} · seed {s}"


models = available_models()
if not models:
    st.warning("outputs/results 下没有结果。先跑：python scripts/run_experiment.py --model resnet18")
    st.stop()

st.sidebar.title("RobustVision Lab")
page = st.sidebar.radio("页面", [
    "Overview", "Image Perturbation", "Model Comparison",
    "Calibration", "Selective Prediction", "Experiment History",
])
model = st.sidebar.selectbox("run", models, format_func=run_label)
st.sidebar.caption("数据：CIFAR-10 test 10000 张")

e1 = read_csv(RES / model / "e1_single.csv")

if page == "Overview":
    st.title("Overview")
    clean_df = read_csv(RES / model / "e0_clean.csv")
    c1, c2, c3, c4 = st.columns(4)
    if len(clean_df):
        c1.metric("Accuracy (clean)", f"{float(clean_df.accuracy[0]):.3f}")
        c2.metric("ECE (15 bins)", f"{float(clean_df.ece15[0]):.3f}")
        c3.metric("Brier", f"{float(clean_df.brier[0]):.3f}")
        c4.metric("AURC", f"{float(clean_df.aurc[0]):.3f}")

    st.subheader("Single perturbations")
    show_png(fig_path(model, "robustness"))

    st.subheader("归一化后的破坏速度")
    show_png(fig_path(model, "normalized"),
             "每条线按自身在 s=1.0 的损失归一化：比的是掉得多快，不是掉多少")

    st.subheader("最脆弱 / 最稳健")
    if len(e1):
        worst = e1.loc[e1.accuracy.idxmin()]
        best = e1.loc[e1.accuracy.idxmax()]
        a, b = st.columns(2)
        a.metric("最差条件", f"{worst.perturbation} s={worst.strength}", f"{worst.accuracy:.3f}")
        b.metric("最好条件", f"{best.perturbation} s={best.strength}", f"{best.accuracy:.3f}")

elif page == "Image Perturbation":
    st.title("Image Perturbation")
    imgs, labels = test_images()
    if imgs is None:
        st.error("找不到 data/cifar10.npz，先跑 scripts/download_data.py")
        st.stop()

    left, right = st.columns([1, 3])
    with left:
        idx = st.slider("图片序号", 0, len(imgs) - 1, 0)
        kind = st.radio("扰动类型", ["单扰动", "复合扰动"])
        pool = list(REGISTRY) if kind == "单扰动" else list(COMPOSITES)
        name = st.selectbox("选择", pool)
        s = st.slider("强度", 0.0, 1.0, 0.6, 0.1)
        seed = st.number_input("随机种子", 0, 9999, 42, step=1)

    x = imgs[idx:idx + 1]
    with right:
        st.caption(f"真实类别：{CLASSES[int(labels[idx])]}")
        cols = st.columns(4, gap="small")
        cols[0].image(to_pil(x[0]), caption="原图")
        for i, sd in enumerate([seed, seed + 1, seed + 2], start=1):
            y = apply(name, x, s, seed=sd) if kind == "单扰动" else Composite(name)(x, s, seed=sd)
            cols[i].image(to_pil(y[0]), caption=f"seed {sd}")

elif page == "Model Comparison":
    st.title("Model Comparison")
    if len(models) < 2:
        st.info("目前只有一个 run，训完 resnet50 或 seed 43/44 之后这里会自动出对比。")
    default = [m for m in models if m.endswith("seed42")][:2] or models[:2]
    chosen = st.multiselect("对比哪些 run", models, default=default, format_func=run_label)
    rows = []
    for m in chosen:
        df = read_csv(RES / m / "e1_single.csv")
        if len(df):
            rows.append(df.assign(run=run_label(m)))
    if rows:
        all_e1 = pd.concat(rows)
        pick = st.multiselect("扰动", sorted(all_e1.perturbation.unique()),
                              default=sorted(all_e1.perturbation.unique())[:3])
        if pick:
            sub = all_e1[all_e1.perturbation.isin(pick)]
            pivot = sub.pivot_table(index="strength", columns=["perturbation", "run"],
                                    values="accuracy")
            st.line_chart(pivot)
        base = {run_label(m): read_csv(RES / m / "e0_clean.csv") for m in chosen}
        clean = {k: float(v.accuracy[0]) for k, v in base.items() if len(v)}
        tail = all_e1[all_e1.strength == 1.0].pivot_table(index="perturbation", columns="run",
                                                          values="accuracy")
        st.caption("s=1.0 的准确率，括号里是除以各自干净准确率后的保持率")
        show = tail.copy()
        for c in show.columns:
            show[c] = [f"{v:.3f} ({v / clean[c]:.2f})" if c in clean else f"{v:.3f}"
                       for v in tail[c]]
        st.dataframe(show, width="stretch")

elif page == "Calibration":
    st.title("Calibration")
    show_png(fig_path(model, "reliability"), "可靠性图：干净数据 vs 最差扰动条件")
    e4 = read_csv(RES / model / "e4_reliability.csv")
    if len(e4):
        conds = e4.condition.unique().tolist()
        cond = st.selectbox("条件", conds)
        sub = e4[e4.condition == cond][["bin", "count", "bin_conf", "bin_acc"]]
        st.dataframe(sub, width="stretch", hide_index=True)
        st.caption(
            "ECE 对分箱敏感，所以同时报 MCE 和等频 ECE。"
            "置信度与实际正确率脱节越远，越说明模型在扰动下过度自信。"
        )

    e7 = read_csv(RES / model / "e7_temperature.csv")
    if len(e7):
        st.subheader("Post-hoc 温度缩放")
        t = float(e7.temperature.iloc[0])
        st.caption(f"在验证集上拟合 T = {t:.2f}，直接用到 test 的所有条件上")
        show = e7[["condition", "accuracy", "ece_before", "ece_after"]].copy()
        for c in ("accuracy", "ece_before", "ece_after"):
            show[c] = show[c].round(3)
        show["降幅"] = (e7.ece_before - e7.ece_after).round(3)
        st.dataframe(show, width="stretch", hide_index=True)
        clean_row = e7[e7.condition == "clean"].iloc[0]
        st.markdown(
            f"干净数据 ECE 从 {clean_row.ece_before:.3f} 降到 {clean_row.ece_after:.3f}，"
            f"修得很好。但最差的扰动只从 {e7.ece_before.max():.3f} 降到 "
            f"{e7.loc[e7.ece_before.idxmax(), 'ece_after']:.3f}——"
            "一个标量温度救不了分布偏移下的过度自信。"
        )

elif page == "Selective Prediction":
    st.title("Selective Prediction")
    show_png(fig_path(model, "risk_coverage"), "低置信度样本被拒绝后的风险-覆盖率曲线")
    e6 = read_csv(RES / model / "e6_selective.csv")
    if len(e6):
        st.subheader("强度 0.6 下的选择性预测")
        show = e6.copy()
        for c in ("accuracy", "risk_at_0.8", "risk_at_0.9", "aurc"):
            show[c] = show[c].round(4)
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption("risk@0.8 表示只保留置信度最高的 80% 样本时，剩下的错误率。")

else:
    st.title("Experiment History")
    run_files = sorted((OUT / "runs").rglob("*.json"))
    if not run_files:
        st.info("outputs/runs 下还没有记录。")
    else:
        st.caption(f"共 {len(run_files)} 条记录，全部由脚本写入，没有手填的")
        kinds = sorted({p.parent.name for p in run_files})
        kind = st.selectbox("实验", kinds)
        files = [p for p in run_files if p.parent.name == kind]
        df = pd.DataFrame([pd.read_json(p, typ="series") for p in files])
        drop = [c for c in ("config", "metrics") if c in df.columns]
        st.dataframe(df.drop(columns=drop), width="stretch", hide_index=True)
