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


models = available_models()
if not models:
    st.warning("outputs/results 下没有结果。先跑：python scripts/run_experiment.py --model resnet18")
    st.stop()

st.sidebar.title("RobustVision Lab")
page = st.sidebar.radio("页面", [
    "Overview", "Image Perturbation", "Model Comparison",
    "Calibration", "Selective Prediction", "Experiment History",
])
model = st.sidebar.selectbox("模型", models)
st.sidebar.caption("数据：CIFAR-10 test 10000 张，单种子 42")

e1 = read_csv(RES / model / "e1_single.csv")

if page == "Overview":
    st.title("Overview")
    c1, c2, c3, c4 = st.columns(4)
    base = pd.read_json(OUT / "runs" / "eval" / f"{model}_test.json", typ="series")
    c1.metric("Accuracy (clean)", f"{float(base['accuracy']):.3f}")
    c2.metric("ECE (15 bins)", f"{float(base['ece15']):.3f}")
    c3.metric("Brier", f"{float(base['brier']):.3f}")
    c4.metric("AURC", f"{float(base['aurc']):.3f}")

    st.subheader("Single perturbations")
    show_png(FIG / f"{model}_robustness.png")

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
        st.info("目前只有一个模型的结果，训练 resnet50 之后这里会自动出现对比。")
    rows = []
    for m in models:
        df = read_csv(RES / m / "e1_single.csv")
        if len(df):
            rows.append(df.assign(model=m))
    if rows:
        all_e1 = pd.concat(rows)
        pick = st.multiselect("扰动", sorted(all_e1.perturbation.unique()),
                              default=sorted(all_e1.perturbation.unique())[:3])
        if pick:
            sub = all_e1[all_e1.perturbation.isin(pick)]
            pivot = sub.pivot_table(index="strength", columns=["perturbation", "model"],
                                    values="accuracy")
            st.line_chart(pivot)
        st.dataframe(
            all_e1.groupby(["model", "perturbation"]).accuracy.mean().round(4).unstack(),
            width="stretch",
        )

elif page == "Calibration":
    st.title("Calibration")
    show_png(FIG / f"{model}_reliability.png", "可靠性图：干净数据 vs 最差扰动条件")
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

elif page == "Selective Prediction":
    st.title("Selective Prediction")
    show_png(FIG / f"{model}_risk_coverage.png", "低置信度样本被拒绝后的风险-覆盖率曲线")
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
