# PROJECT_SPEC — RobustVision Lab

Version 0.1 · 2026-11-24
Status: approved for implementation (Stage 2 onward)

Decisions that are painful to reverse once the code exists.

If something here turns out wrong, change this file before the code.

---

## 1. Scope

Build a reproducible evaluation harness that answers four questions about image
classifiers under perturbation:

| ID | Question | Operationalised as |
|----|----------|--------------------|
| RQ1 | How does accuracy move under different distribution shifts? | 7 single perturbations × 5 strengths × 2 models, reported as accuracy drop vs clean |
| RQ2 | Is a composition of perturbations different from the sum of its parts? | 4 composites vs the product of their single-perturbation accuracies |
| RQ3 | Does confidence track correctness? | ECE, MCE, Brier, reliability diagrams |
| RQ4 | Can rejecting low-confidence predictions cut error? | Risk–coverage curve, AURC, risk at 80/90% coverage |

Non-goals: new architectures, new training methods, adversarial attacks,
state-of-the-art robustness numbers, ImageNet-scale evaluation. This is a
measurement project, not a method project.

---

## 2. Dataset

**CIFAR-10**, downloaded once into a local npz.

- 50,000 train / 10,000 test, 32×32 RGB, 10 classes.
- Split: 45,000 train / 5,000 validation, carved from the 50k with a fixed
  permutation (seed 42). **Test set is touched only for the final tables** —
  no threshold, bin count, or hyper-parameter is tuned on it.
- Preprocessing: `ToTensor()` only. Normalisation uses CIFAR-10 statistics
  (`mean=(0.4914,0.4822,0.4465)`, `std=(0.2023,0.1994,0.2010)`).
- Training augmentation: random crop 32 with padding 4, random horizontal flip.
  No augmentation at evaluation time.

**Data source.** `torchvision`'s default mirror is `cs.toronto.edu`, which
measured ~90 KB/s here — 30+ minutes for 170 MB. A proxy made it worse, 78 B/s.
So `scripts/download_data.py` pulls the HuggingFace mirror of CIFAR-10
(`uoft-cs/cifar10`, parquet, peaked at 2.5 MB/s through hf-mirror.com) and
converts it to `data/cifar10.npz` (uint8, 184 MB). Loading goes through
`dataio.CifarArray` rather than `torchvision.datasets`, which drops the pickle
layer and the download-on-import. The arrays are the same pixels as the
official release; anyone with fast access to the tarball can unpack it and
compare.

Files live under `data/`, never committed (`.gitignore`d).

**Decision: keep 32×32, do not upsample to 224.** Upsampling adds no
information; it only exists to suit ImageNet weights. It also multiplies
inference cost by 49×, and the perturbation grid runs 7×5×2 forward passes over
10,000 images. Training at 32×32 gets ResNet18 to 91.2% test accuracy in 40
epochs — high enough that a perturbation-induced drop means something.

---

## 3. Models

Two backbones, both `torchvision` ImageNet-pretrained:

| Key | Architecture | CIFAR adaptation |
|-----|--------------|------------------|
| `resnet18` | ResNet-18 | `conv1` → 3×3, stride 1, padding 1, bias=False; `maxpool` → Identity |
| `resnet50` | ResNet-50 | same |

The first stem layer is re-initialised because the shape no longer matches;
everything else loads from the checkpoint. ResNet18 came out at **119 of 122
tensors matched**. That number goes into every run record, so it never has to
be re-measured.

`mobilenet_v3_large` is **deferred**. Its stem downsamples 32×32 to 1×1, which
would need stride surgery on three stages. It is listed as optional in the
backlog; adding it is not allowed to block Stages 2–6.

Training recipe (identical across models, only the backbone differs):

```yaml
optimizer: sgd          # momentum 0.9, nesterov, weight_decay 5e-4
lr: 0.1                 # cosine annealing to 0
epochs: 40
batch_size: 128
label_smoothing: 0.0
seed: 42
amp: true               # fp16 autocast
```

Checkpoints land in `outputs/checkpoints/<model>_seed<k>.pt` and are gitignored.
A model is only used in experiments if its validation accuracy is recorded in
the run log; no number is ever hand-written.

---

## 4. Perturbations

All take a float tensor `x ∈ [0,1]` of shape `(N, 3, 32, 32)`, never modify it
in place, and are parameterised by `strength s ∈ [0, 1]`, where `s = 0` is the
identity. The RNG is seeded per call, so the same image gets the same corruption
whichever model is being scored.

| Key | Operation | Parameter from `s` |
|-----|-----------|--------------------|
| `gaussian_noise` | additive Gaussian | `σ = 0.15 · s` |
| `gaussian_blur` | Gaussian kernel | `k = 2·round(1+3s)+1`, `σ = 0.1 + 2.0s` |
| `motion_blur` | directional convolution | kernel length `2·round(1+5s)+1`, angle drawn per image from the seeded RNG |
| `brightness` | multiplicative | `factor = 1 + 0.8s` (brighten only; darkening is `contrast`'s job) |
| `contrast` | blend toward mean | `factor = 1 - 0.8s` |
| `occlusion` | grey square, random position | side `= floor(32 · sqrt(0.05 + 0.45s))` |
| `jpeg` | encode/decode | quality `= round(100 - 90s)` |

Strength grid: `s ∈ {0.2, 0.4, 0.6, 0.8, 1.0}`, plus `s = 0` as the clean
baseline. Every perturbation is applied to the **whole test set**, not a
sample, for RQ1.

Composites (RQ2), applied in this fixed order:

| Key | Pipeline |
|-----|----------|
| `noise_blur` | `gaussian_noise` → `gaussian_blur` |
| `blur_bright` | `gaussian_blur` → `brightness` |
| `noise_occ` | `gaussian_noise` → `occlusion` |
| `triple` | `brightness` → `contrast` → `gaussian_blur` |

In a composite, all members share the same `s`. Random-order pipelines are a
backlog item, not part of the main results.

**Why fixed order**: order changes the result (blur-then-noise ≠
noise-then-blur), and only a fixed order gives a reproducible number. Order
sensitivity gets one sentence in the report, not an experiment.

---

## 5. Metrics

Let `p ∈ Δ^9` be the softmax output, `ŷ = argmax p`, `conf = max p`, `y` the
label, `n` the sample count.

**Accuracy** — `mean(ŷ == y)`. Top-1 only; CIFAR-10 has 10 classes so Top-5 is
meaningless here. Also reported: per-class accuracy and the 10×10 confusion
matrix.

**ECE** (expected calibration error), equal-width bins:

```
ECE = Σ_m (|B_m| / n) · |acc(B_m) − conf(B_m)|
```

`M = 15` bins by default, bin index `min(floor(conf · M), M−1)` so that
`conf = 1.0` lands in the last bin instead of an out-of-range 16th. Empty bins
contribute 0. Also report **MCE** (`max` over non-empty bins) and an
**adaptive/equal-frequency ECE**. ECE shifts with bin count, and one number is
easy to over-read.

**Brier score** (multi-class, mean squared error against the one-hot target):

```
Brier = (1/n) Σ_i Σ_k (p_ik − y_ik)²
```

Note this is the *summed* multi-class variant, so its range is `[0, 2]`, not
`[0, 1]`. Spell that out in the README, or anyone who only knows the binary
version will read 0.13 as a bug.

**Risk–coverage** — sort samples by `conf` descending, sweep coverage
`c ∈ {0.05, 0.10, …, 1.00}`, take the top `⌈c·n⌉` samples, define
`risk(c) = 1 − accuracy(accepted)`. Report:

- the full curve (plotted),
- `AURC` = area under `risk(c)`, trapezoid, over `c ∈ [0,1]`,
- `risk@0.8` and `risk@0.9` — single numbers for tables.

Lower AURC is better. A model that is both accurate and well-calibrated gets a
flat curve near its error rate; an overconfident one gets an L-shape.

---

## 6. Experiments

All runs write one JSON to `outputs/runs/<experiment>/<run_id>.json` and are
launched by a script in `scripts/`, never by hand-editing state in a notebook.

| ID | Content | Inputs | Est. time |
|----|---------|--------|-----------|
| E1 | Single perturbation sweep | 2 models × 7 perturbations × 5 strengths | ~4 min |
| E2 | Composite vs single | 2 models × 4 composites × 5 strengths | ~2 min |
| E3 | Model comparison | aggregates E1/E2 | 0 |
| E4 | Calibration | ECE/MCE/Brier/reliability on clean + worst perturbed | ~1 min |
| E5 | Risk–coverage | AURC + curve per model × condition | ~1 min |
| E6 | Selective prediction under shift | risk@coverage per perturbation type | ~1 min |

Time estimates assume a trained checkpoint already exists; they cover inference
only. Training happens once per model (Stage 2). Measured on the 4060:
**14.4 s/epoch, 9.9 min for 40 epochs** at batch 128 — about 3× faster than the
first estimate, which assumed 32×32 would be slower than it is.

RQ2 needs a reference point, so the report also computes, for each composite,
`Π accuracy of its members` and compares it against the measured composite
accuracy. The gap is the interaction effect. No claim of "composition is worse
than the sum" is made unless that gap is consistent across all five strengths
and both models.

---

## 7. Interfaces

Fixed now so Stage 2–5 can be written in parallel without merge pain.

```python
# models/base.py
class VisionModel(Protocol):
    name: str
    num_classes: int
    def predict_proba(self, x: Tensor) -> Tensor:   # (N, C) softmax, float32
        ...
    def predict(self, x: Tensor) -> Tensor:         # (N,) int64
        ...

# perturbations/base.py
class Perturbation(Protocol):
    name: str
    def __call__(self, x: Tensor, strength: float, seed: int = 0) -> Tensor:
        ...                                          # (N,3,32,32) in [0,1], returns a new tensor

# perturbations/composite.py
class Composite:
    def __init__(self, members: list[str]) -> None: ...
    def __call__(self, x: Tensor, strength: float, seed: int = 0) -> Tensor: ...

# metrics/classification.py
def accuracy(logits_or_probs: Tensor, y: Tensor) -> float: ...
def confusion_matrix(probs: Tensor, y: Tensor, k: int = 10) -> np.ndarray: ...

# metrics/calibration.py
def ece(probs: Tensor, y: Tensor, n_bins: int = 15, adaptive: bool = False) -> float: ...
def brier(probs: Tensor, y: Tensor) -> float: ...
def reliability_bins(probs: Tensor, y: Tensor, n_bins: int = 15) -> dict: ...

# metrics/selective.py
def risk_coverage(probs: Tensor, y: Tensor, grid: np.ndarray | None = None) -> dict: ...
```

Every function takes `probs` (already softmaxed), not logits. One convention,
applied everywhere.

---

## 8. Run record

```json
{
  "run_id": "e1_resnet18_gaussian_noise_s0.4_seed42",
  "experiment": "E1",
  "model": "resnet18",
  "checkpoint": "outputs/checkpoints/resnet18_seed42.pt",
  "dataset": "cifar10",
  "split": "test",
  "n_samples": 10000,
  "perturbation": {"name": "gaussian_noise", "strength": 0.4, "seed": 42},
  "metrics": {"accuracy": 0.0, "ece15": 0.0, "ece_adaptive": 0.0,
              "brier": 0.0, "aurc": 0.0, "risk_at_0.8": 0.0},
  "config_hash": "sha256:...",
  "git_commit": "abc1234",
  "torch": "2.11.0+cu128",
  "gpu": "RTX 4060 Laptop",
  "timestamp": "2026-11-24T21:30:00+08:00",
  "duration_s": 12.4
}
```

`config_hash` is the SHA-256 of the resolved YAML. Two runs with the same hash
and the same git commit must produce the same metrics; if they don't, that is a
bug, not noise.

---

## 9. Testing

`pytest`, before every commit. These cases have to exist:

| Test | Why |
|------|-----|
| ECE of a perfectly calibrated synthetic set is ≈ 0 | catches bin-boundary mistakes |
| ECE of all-wrong-max-confidence is ≈ 1 | catches sign errors |
| `conf == 1.0` lands in the last bin, not an index error | the classic off-by-one |
| Brier of one-hot perfect predictions is 0, uniform is 0.9 | pins the `[0,2]` range |
| Risk–coverage at `c=1.0` equals `1 − accuracy` | ties the two metrics together |
| AURC is monotone: a strictly better-ranked model has lower AURC | catches a reversed sort |
| Perturbation with `s=0` returns the input unchanged (within 1e-6) | identity contract |
| Same seed → identical perturbed tensor; different seed → different | reproducibility contract |
| `eval()` mode returns identical logits on repeated calls | catches dropout / BN drift |

Perturbation tests run on synthetic tensors; they must not require the dataset
to be downloaded.

---

## 10. Risks

| Risk | Mitigation |
|------|-----------|
| ECE quietly wrong (the most common mistake in this literature) | the calibration tests above, plus a hand-computed 4-sample example |
| Pretrained weights partially discarded at the stem | record the matched-tensor count per run (119/122 for ResNet18); a from-scratch run would quantify the gap |
| CIFAR-10 test set gets used during development | only the validation split is touched before the final tables; each run record names its split |
| Composite interaction claimed without evidence | E2 reports the per-strength gap only; "worse than the sum" needs the gap to hold across 5 strengths × 2 models |
| Report overfitted to a lucky seed | final tables use seeds {42, 43, 44}; single-seed numbers are labelled exploratory |
| 8 GB VRAM | batch 128 at 32×32 stays far below the limit — ResNet18 trained for 40 epochs without pressure |

## 11. Acceptance

Stage is done when, on a fresh clone:

1. `pip install -r requirements.txt` + torch cu128, then
2. `python scripts/download_data.py` fetches CIFAR-10, then
3. `python scripts/train.py --config configs/resnet18.yaml` produces a
   checkpoint with validation accuracy recorded, then
4. `python scripts/run_experiment.py --experiment E1` writes JSON records, then
5. `streamlit run app/main.py` shows the results without crashing, then
6. `pytest` is green.

Every number in the README comes from a run record in `outputs/runs/`.

---

## 12. Backlog (not now)

- `mobilenet_v3_large` (stem surgery)
- random-order composites
- CIFAR-10-C as an external, published shift baseline
- temperature scaling as a post-hoc calibration baseline
- confidence histogram on the app's Overview page
- ablation: same recipe trained from scratch, to price the pretrained init
