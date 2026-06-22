# Provenance-Aware Image Reconstruction

*Labels which pixels in an AI-restored image were constrained by the measurement — and which the model invented to fill the gap — with calibrated uncertainty.*

---

## The question Moon Mode never answered

In 2019, photographers testing the Huawei P30 noticed something that sparked a months-long debate: photos of the Moon showed sharply defined craters even when the raw sensor data contained only a blurry white disc. The camera used AI super-resolution. The question that no one gave a *quantitative* answer to was:

> **How much of that detail was recovered from what the sensor actually measured — and how much was the model's prior expectation of what a Moon should look like?**

It remained a debate rather than a measurement. Not because the question is unanswerable in principle, but because answering it requires knowing the degradation operator **A** — the sensor's point-spread function, the compression, the noise model. Huawei's pipeline is a black box. **This project does not analyse it, and makes no claim about it.**

What we do instead: build the instrument the debate lacked, and demonstrate it in a controlled setting where both the operator and the ground truth are known. Any pixel classification the system produces can be verified against the true image, because in our evaluation we hold that truth out and measure against it.

---

## What it does

Every AI image restoration has two kinds of output pixels.

Some pixels are **constrained** by the measurement: the degraded input strongly determines what value they should take. The model has little freedom here; any reasonable algorithm would agree.

Other pixels are **invented**: the input gives the model almost no information about the correct value. The model fills the gap using patterns learned from training data. The result may look exactly right — or exactly wrong — and the input alone cannot tell you which.

This system labels the difference, per pixel:

| Class | Meaning |
|---|---|
| **Recovered** | The measurement constrains this pixel's energy; the model's contribution is small |
| **Invented** | The null-space contribution dominates; the model fabricated this detail |

**Critical distinction: "invented" means *not determined by the input* — not necessarily wrong.** A recovered texture may be a worse reconstruction than an invented one. The claim is epistemic, not evaluative: where a pixel is invented, the measurement provides no ground for preferring one value over another.

For 4× super-resolution with a bicubic downsampling operator, the null space is large (61 440 of 65 536 Fourier components for a 256×256 image). All fine-grained texture in the output is invented by construction. The system shows *which pixels are more vs. less invented*, and how confidently.

### Traditional SR vs. this system

| | Traditional SR | This system |
|---|---|---|
| Output | "Here is a sharp image" | "Here is a sharp image, and here is what I had information about vs. what I invented" |
| Stance | Trust me | Verify me |
| Uncertainty | Hidden in the model | Exposed as a calibrated per-pixel map |
| Inspectable | No | Yes — operator, null-space decomposition, provenance overlay all auditable |

---

## How it works

### Range–null decomposition

For a linear degradation **y = Ax**, the reconstructed output decomposes exactly as:

```
x̂  =  A⁺y  +  (I − A⁺A)·z
       ─────    ─────────────
     range part   null part
     (measured)  (invented)
```

`A⁺y` is determined entirely by the measurement **y** — it is identical across all model runs. `(I − A⁺A)·z` lives in the null space of **A**: any value here is equally consistent with the observation. The model's output fills the null space with plausible-looking detail, but that detail is not recoverable from **y** alone.

This decomposition is textbook. Kawar et al. (DDRM, 2022), Wang et al. (DDNM, 2022), and Chung et al. (DPS, 2022) each use variants of it to steer diffusion models toward measurement-consistent reconstructions. The common thread across these works: *range-space corrections are deterministic; null-space corrections are sampled*.

**The contribution here is not the decomposition** — it is using it as a *provenance artifact*: measuring the null-space fraction per pixel, running an ensemble to quantify how much the null content varies across seeds, and calibrating whether that spread reliably predicts reconstruction error.

### Calibrated uncertainty

An ensemble of N=6 model runs is produced for each input. All members share the same range component `A⁺y`; only the null-space component `(I−A⁺A)·z_i` differs. The per-pixel standard deviation of the ensemble measures how much the model's invented content varies — high spread where the model is uncertain, low spread where it converges.

The system is **calibrated** if high spread reliably signals high actual error. This is measured with three statistics on held-out images:

- **Pearson r** between per-pixel predicted spread and actual |error|
- **Slope** of the regression (slope = 1 is perfect magnitude calibration)
- **Expected Calibration Error (ECE)**

---

## Results

Evaluated on 16 held-out ImageNet patches, 256×256 RGB, with BicubicDownsample(4) as the degradation operator and ResShift as the reconstruction model:

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Pearson r | **+0.9667** | ≥ 0.90 | ✓ PASS |
| Slope | **1.5301** | 0.5 – 2.0 | ✓ PASS |
| ECE | **0.0282** | ≤ 0.30 | ✓ PASS |
| **IS\_CALIBRATED** | **YES** | — | ✓ |

**Honest caveat on slope = 1.53:** The uncertainty ordering is strong — where the ensemble spread is high, reconstruction error is also high (r = +0.97). The absolute magnitude is off by ~1.5×: the model is mildly underconfident. In the lowest-uncertainty bin, actual error is roughly 5× the predicted spread. The system passes the calibration threshold, but perfect magnitude calibration (slope = 1.0) has not been demonstrated at this scale.

A per-pixel provenance overlay is produced for each reconstruction. For the 4× bicubic case, 37–77% of pixels are classified as invented (varies by image content; fine-textured images score higher).

**→ [Interactive demo](demo/index.html)** — Panel A: wipe between polished output and provenance overlay. Panel B: reliability curve + certified metrics. Panel C: seed toggle demonstrating that the range component is fixed while invented content varies; limitation case (r = 0.945, slope = 0.156) shown with explanation.

---

## Reproduce it

The full calibration is gated by a single script:

```bash
git checkout 5f99f84
python falsify.py --fast    # deterministic checks only, ~0.1s, no GPU required
python falsify.py --full    # adds ResShift calibration, ~105s, GPU recommended
```

Expected output of `--full`:

```
Pearson r  : +0.9667   (need ≥ 0.9)   ✓
slope      :  1.5301   (need [0.5, 2.0]) ✓
ECE        :  0.0282   (need ≤ 0.3)   ✓

FALSIFY: GREEN  [full]
```

`--fast` runs five deterministic checks: operator pseudo-inverse correctness (`A·A⁺·A = A` at 1e-12), data consistency (`‖A·x̂ − y‖ ≤ 1e-10` for all outputs), R2 structural invariant (`std(x̂_i) = std(null_i)` — uncertainty comes from null-space spread only), classifier honesty (zero false positives, zero false negatives against a known oracle), and fast calibration on the oracle engine. No GPU or model weights needed.

`--full` adds the ResShift calibration across 16 images × 6 ensemble members. Requires model weights (see Setup).

---

## Scope and limitations

This is an explicitly bounded proof-of-concept. The boundaries matter:

**What is demonstrated:**
- One degradation operator per run (BicubicDownsample(4) in the calibration eval)
- One reconstruction model (ResShift, trained on ImageNet bicubic SR pairs)
- Synthetic evaluation only: **y** is computed from a known **x**, so ground truth is available
- 16 held-out 256×256 patches from ImageNet validation set
- Grayscale (luminance channel); colour processing is an extension

**What is not demonstrated and not claimed:**
- Real camera pipelines: ISP processing, lens blur, JPEG compression, noise, and sensor characteristics are not modelled. A is unknown in a real phone camera; the system cannot be applied without a known operator.
- Nonlinear degradations: the range–null decomposition requires **A** to be linear. Nonlinear operators (many real ISPs, RAW processing pipelines) are named future work.
- The Huawei Moon case specifically: no claim is made about any real product or manufacturer's pipeline.
- Manipulation detection: the system classifies pixels as invented or recovered relative to a known **A** and **y**. It does not detect tampering, compositing, or adversarial modification.

**Where the problem generalises (future scope, not current capability):**
Medical imaging (MRI undersampling reconstruction), satellite and remote-sensing SR, microscopy deconvolution, forensic imaging — any domain where a linear forward model **A** is known and provenance of the reconstruction matters. These are the application areas where calibrated per-pixel provenance would be most consequential. None of them are handled by this repo today.

---

## Setup

### Environment

The environment was assembled with non-standard dependency management (MPS wheels, `--no-deps` installs) to resolve Apple Silicon / PyTorch / ResShift conflicts. A standard `pip install -r requirements.txt` will likely not work out of the box on all platforms.

```bash
python -m venv .venv
source .venv/bin/activate
# Install PyTorch for MPS (Apple Silicon) or CUDA (Linux/Windows GPU):
# See https://pytorch.org/get-started/locally/ for your platform.
pip install torch torchvision
# ResShift dependencies (install individually if conflicts arise):
pip install basicsr omegaconf einops
```

The falsify `--fast` mode (deterministic checks, no ResShift) has no GPU requirement and minimal dependencies: `numpy`, `scipy`.

### Model weights

Weights are **not included in the repository** (ResShift checkpoint: 456 MB; VQ autoencoder: 211 MB). Place them in `weights/`:

```
weights/
  resshift_bicsrx4_s4.pth      # ResShift bicubic ×4 checkpoint
  autoencoder_vq_f4.pth        # VQ autoencoder
```

The ResShift weights are distributed by the original authors under their licence.
Download links are on the [ResShift GitHub releases page](https://github.com/zsyOAOA/ResShift/releases).

### ResShift model code

`engine/resshift.py` imports from `vendor/ResShift/`. The full ResShift source is **not bundled** in this repo (only a handful of stubs are tracked). To run the engine or `falsify.py --full`, populate `vendor/ResShift/` first:

```bash
git clone https://github.com/zsyOAOA/ResShift vendor/ResShift
```

`falsify.py --fast` (deterministic checks only) does **not** require ResShift code or weights.

### What is and isn't in the repo

| Path | In repo | Reason |
|---|---|---|
| `operators/`, `engine/`, `layer/` | ✓ | Core science |
| `falsify.py` | ✓ | Reproducibility gate |
| `demo/` | ✓ | Pre-computed artefacts |
| `eval/` | ✓ | Generation scripts |
| `vendor/ResShift/` | ✗ (partial) | Clone separately — see above |
| `weights/` | ✗ | Too large; obtain from ResShift releases |
| `.venv/` | ✗ | Virtualenv |
| `output/` | ✗ | Intermediate generated files |

---

## Project structure

```
operators/     — Linear degradation operators (BoxDownsample, BicubicDownsample, MaskOp, CircularBlur)
engine/        — Reconstruction engines (ResShift wrapper, OracleEngine for tests)
layer/         — Provenance layer: rectify, ensemble_stats, classify, calibration
eval/          — Calibration and demo generation scripts
tests/         — Test suite (R7: operator correctness; calibration; classify)
demo/          — Static HTML demo + pre-computed assets
falsify.py     — Project gate: exit 0 + FALSIFY: GREEN only if all checks pass
vendor/        — Stubs only; clone ResShift separately (see Setup)
weights/       — Model checkpoints (not in repo)
```

---

## Commit anchor

`git checkout 5f99f84` gives the exact state that produced the demo assets and the certified calibration numbers. Running `python falsify.py --full` at that commit reproduces r = +0.9667, slope = 1.5301, ECE = 0.0282, IS_CALIBRATED = YES.
