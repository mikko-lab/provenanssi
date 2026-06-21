"""
eval/smoke_resshift.py — ResShift conditioning smoke test.

Goal: verify that ResShift's ensemble spread is SPATIALLY STRUCTURED BY y
(i.e., std is higher where y is less informative), unlike church-256 which
had a uniform prior-driven spread.

Method
------
1. x_gt = 256×256 synthetic (sinusoidal base + checkerboard null).
2. y = BoxDownsample(4) · x_gt  →  64×64 LR measurement.
3. N=6 ResShift samples conditioned on y (different seeds).
4. Report:
   - Conditioning check: ‖A·sample − y‖ for each sample (R3-like check).
   - Per-pixel std map: is it spatially structured by the input or flat?
   - Correlation analysis: std vs reconstruction error per pixel.
   - Mean pairwise diff in null space.
   - Runtime per sample.
5. Save PNGs: x_gt, A⁺y, mean, std, 6 samples.

Critical test: std map spatial structure.
If std is higher in the null space (checkerboard-like pattern, not in the
smooth sinusoidal) and lower in the range space, that's evidence the model
has learned a CONDITIONAL distribution p(x|y) — which is the prerequisite
for calibrated uncertainty estimation.

R4: x_gt used only to make y and for error evaluation.
R5: SHA-256, seed, config logged.
R6: Honest interpretation of what the samples show.
R9: BoxDownsample(4) for this model's native scale.
"""
from __future__ import annotations
import os
import struct
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from operators.superres import BoxDownsample
from engine.resshift import ResShiftEngine

# ---------------------------------------------------------------------------

HR_SIZE    = 256
SCALE      = 4
N_ENSEMBLE = 6
SEED_GT    = 42

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


# ---------------------------------------------------------------------------
# PNG writer (stdlib only)

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    c = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", c)


def _save_png(arr: np.ndarray, path: str, auto_scale: bool = True) -> None:
    if auto_scale:
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo + 1e-12)
    px = (arr.clip(0, 1) * 255).astype(np.uint8)
    h, w = px.shape
    raw = b"".join(b"\x00" + bytes(row) for row in px)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", ihdr))
        f.write(_png_chunk(b"IDAT", zlib.compress(raw)))
        f.write(_png_chunk(b"IEND", b""))


# ---------------------------------------------------------------------------

def _make_gt(size: int = HR_SIZE, seed: int = SEED_GT) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H, W = size, size
    phase_y = rng.uniform(0, np.pi)
    phase_x = rng.uniform(0, np.pi)
    y_lin = np.linspace(0, 2 * np.pi * 1.5, H)[:, None]
    x_lin = np.linspace(0, 2 * np.pi * 1.5, W)[None, :]
    base = 0.35 * (np.sin(2 * y_lin + phase_y) * np.cos(3 * x_lin + phase_x)
                   + np.sin(y_lin) * np.sin(2 * x_lin)) + 0.5
    checker = ((np.indices((H, W)).sum(axis=0) % 2) * 2.0 - 1.0) * 0.25
    return np.clip(base + checker, 0.0, 1.0).astype(np.float64)


# ---------------------------------------------------------------------------

def run(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 64)
    log("Provenanssi — ResShift conditioning smoke test")
    log("=" * 64)
    log()

    op     = BoxDownsample(SCALE)
    engine = ResShiftEngine(op)
    log()

    log("Configuration (R5)")
    log(f"  model        : resshift_bicsrx4_s4  (UNetModelSwin + VQModelTorch)")
    log(f"  model SHA    : {engine._ckpt_sha256}")
    log(f"  ae SHA       : {engine._ae_sha256}")
    log(f"  conditioning : model.forward(x, t, lq=y)  — cond_lq=True")
    log(f"  operator     : BoxDownsample({SCALE})")
    log(f"  HR size      : {HR_SIZE}×{HR_SIZE}")
    log(f"  LR size      : {HR_SIZE//SCALE}×{HR_SIZE//SCALE}")
    log(f"  diffusion    : 4-step ResShift exponential schedule")
    log(f"  N ensemble   : {N_ENSEMBLE}")
    log(f"  GT seed      : {SEED_GT}")
    log()

    # Ground truth and LR
    x_gt = _make_gt()
    y    = op.forward(x_gt)    # (R4: x_gt used ONLY to make y)
    log(f"Test image     : {HR_SIZE}×{HR_SIZE} synthetic (sinusoidal + checkerboard null)")
    log(f"LR observation : {y.shape[0]}×{y.shape[1]}")
    log()

    # ------------------------------------------------------------------
    # 1. Time one sample

    log("Timing one sample ...")
    t0 = time.perf_counter()
    s_time = engine.restore(y)
    t1 = time.perf_counter()
    t_one = t1 - t0
    log(f"  1 sample (4 ResShift steps) : {t_one:.1f}s  ({t_one/4*1000:.0f} ms/step)")
    log(f"  projected N=6 time          : {t_one*6:.0f}s")
    log()

    # ------------------------------------------------------------------
    # 2. Conditioning check: ‖A·sample − y‖ for each sample

    log("Conditioning check: ‖A·sample − y‖ ...")
    residuals = []
    for i in range(N_ENSEMBLE):
        s = engine._sample(y, seed=i)
        Ax = op.forward(s)
        residual = float(np.max(np.abs(Ax - y)))
        residuals.append(residual)
        log(f"  seed={i}  max‖A·x − y‖ = {residual:.6f}")

    log()
    mean_res = float(np.mean(residuals))
    log(f"  mean residual = {mean_res:.6f}")
    log(f"  NOTE: ResShift does NOT enforce A·x = y (no DDNM correction here).")
    log(f"  The residual reflects how close the conditional prior keeps samples to y.")
    log(f"  Non-zero residuals are expected (model is not perfectly data-consistent).")
    log()

    # ------------------------------------------------------------------
    # 3. N=6 ensemble

    log(f"Running N={N_ENSEMBLE} ensemble ...")
    members, t_ens = engine.timed_ensemble(y, N_ENSEMBLE)
    log(f"  Total time : {t_ens:.1f}s  ({t_ens/N_ENSEMBLE:.1f}s/sample)")

    stack = np.stack(members)          # (6, H, W)
    mean_x = stack.mean(axis=0)
    std_x  = stack.std(axis=0)

    log(f"  std map stats:")
    log(f"    mean std = {std_x.mean():.5f}")
    log(f"    max  std = {std_x.max():.5f}")
    log(f"    min  std = {std_x.min():.5f}")
    log()

    # Pairwise diffs
    n_pairs, sum_diff = 0, 0.0
    for i in range(N_ENSEMBLE):
        for j in range(i + 1, N_ENSEMBLE):
            sum_diff += float(np.mean(np.abs(stack[i] - stack[j])))
            n_pairs += 1
    avg_pw = sum_diff / n_pairs
    log(f"  avg pairwise |diff| = {avg_pw:.5f}  ({n_pairs} pairs)")
    log()

    # ------------------------------------------------------------------
    # 4. CRITICAL: Is std spatially structured by y?

    pinv_y = op.pinv(y)   # nearest-neighbour upsampled LR (range-space image)

    # Range-space mask: pixels where BoxDownsample "strongly constrains" x.
    # BoxDownsample(4) averages 4×4 = 16 pixels per LR pixel.
    # The range space of BoxDownsample(4) is the set of images constant within
    # each 4×4 block. So for any single pixel, whether it's strongly constrained
    # depends on the block average — all 16 pixels in a block see the same
    # y-constraint. The null space is the within-block variation.
    # std in range-space pixels should be lower (y constrains the block mean).
    # std in null-space pixels should be higher (model must invent within-block texture).

    # Null-space component of each sample: x - A⁺·A·x = x - op.project(x)
    null_parts = np.stack([s - op.project(s) for s in members])  # (6, H, W)
    null_std   = null_parts.std(axis=0)
    range_std  = (stack - null_parts).std(axis=0)   # std in range component only

    log("Spatial structure analysis (KEY TEST)")
    log(f"  null-space std   : mean={null_std.mean():.5f}  max={null_std.max():.5f}")
    log(f"  range-space std  : mean={range_std.mean():.5f}  max={range_std.max():.5f}")
    log(f"  ratio null/range : {null_std.mean() / (range_std.mean() + 1e-9):.2f}x")
    log()

    # std correlation with pixel error (R4: x_gt available for evaluation only)
    pixel_error = np.abs(mean_x - x_gt)
    corr = float(np.corrcoef(std_x.ravel(), pixel_error.ravel())[0, 1])
    log(f"  Pearson r(std, |mean_x - x_gt|) = {corr:.4f}")
    log(f"    > 0: std predicts where error is high  (calibrated signal)")
    log(f"    ≈ 0: std flat, uncorrelated with error (uncalibrated, like TTA)")
    log(f"    < 0: inverted (perverse)")
    log()

    # ------------------------------------------------------------------
    # 5. Save PNGs

    p = lambda n: os.path.join(out_dir, n)
    _save_png(x_gt,                     p("rs_01_x_gt.png"),       auto_scale=False)
    _save_png(pinv_y,                   p("rs_02_pinv_y.png"),      auto_scale=False)
    _save_png(mean_x,                   p("rs_03_mean.png"),        auto_scale=False)
    _save_png(std_x,                    p("rs_04_std.png"),         auto_scale=True)
    _save_png(null_std,                 p("rs_05_null_std.png"),    auto_scale=True)
    _save_png(range_std,                p("rs_06_range_std.png"),   auto_scale=True)
    _save_png(pixel_error,              p("rs_07_pixel_error.png"), auto_scale=True)
    for i, s in enumerate(members):
        _save_png(s,                    p(f"rs_08_sample_{i}.png"), auto_scale=False)

    log("Saved PNGs (output/):")
    log("  rs_01_x_gt.png       — ground truth 256×256")
    log("  rs_02_pinv_y.png     — A⁺y (nearest-neighbour, no null content)")
    log("  rs_03_mean.png       — ensemble mean (6 ResShift samples)")
    log("  rs_04_std.png        — per-pixel std (auto-scaled)")
    log("  rs_05_null_std.png   — std in null-space component (should be HIGH)")
    log("  rs_06_range_std.png  — std in range-space component (should be LOW)")
    log("  rs_07_pixel_error.png— |mean_x − x_gt| (ground-truth error map)")
    for i in range(N_ENSEMBLE):
        log(f"  rs_08_sample_{i}.png  — ResShift sample seed={i}")
    log()

    # ------------------------------------------------------------------
    # 6. R6 Honest assessment

    log("=" * 64)
    log("R6 — Honest interpretation")
    log("=" * 64)
    log()

    null_dominates = null_std.mean() > range_std.mean() * 1.5
    std_corr_positive = corr > 0.05

    if null_dominates and std_corr_positive:
        log("  ASSESSMENT: STRUCTURED UNCERTAINTY — conditioning is working.")
        log(f"  null_std {null_std.mean():.4f} > range_std {range_std.mean():.4f} (×{null_std.mean()/max(range_std.mean(),1e-9):.1f})")
        log(f"  r(std, error) = {corr:.4f} > 0 — std predicts where error is high.")
        log("  This is the signature of p(x|y): the model invents in the null space")
        log("  (where y gives no constraint) and agrees in the range space (where y does).")
    elif null_dominates:
        log("  ASSESSMENT: PARTIALLY STRUCTURED — null/range split correct, but")
        log(f"  std-vs-error correlation r={corr:.4f} is weak/flat.")
        log("  The null-space spread is there, but doesn't track error magnitude yet.")
        log("  Possible cause: prior mismatch (model trained on ImageNet, not synthetic).")
    elif std_corr_positive:
        log("  ASSESSMENT: ERROR-CORRELATED STD — r > 0, but null/range split not clear.")
        log("  May still be valid; the spatial structure may appear in other metrics.")
    else:
        log("  ASSESSMENT: FLAT/UNCORRELATED — similar to church-256.")
        log(f"  null_std={null_std.mean():.4f}  range_std={range_std.mean():.4f}  r={corr:.4f}")
        log("  Possible causes:")
        log("  (a) Prior mismatch (ImageNet model on synthetic checkerboard)")
        log("  (b) 4-step schedule too coarse (diversity collapsed)")
        log("  (c) LR image normalisation mismatch")

    log()
    log("  Contrast with church-256:")
    log("  church-256: unconditional. Spread = prior. Flat std. r ≈ 0.")
    log("  ResShift:   conditioned on lq=y. Spread = posterior. Should be structured.")
    log()
    log("  Even if r is low here, the MECHANISM is different: the model has learned")
    log("  (LR, HR) pairs. A domain-matched dataset (or a calibration run on N=30")
    log("  images) would give a more honest r estimate than N=6 here.")

    # ------------------------------------------------------------------
    report_path = os.path.join(out_dir, "smoke_resshift_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log()
    log(f"Report saved to {report_path}")


if __name__ == "__main__":
    run()
