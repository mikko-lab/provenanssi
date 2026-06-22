"""
eval/calibrate_resshift.py — Calibration of ResShift + bicubic degradation (R8).

Question
--------
With degradation MATCHED to ResShift's training (bicubic x4) and evaluated on
ImageNet-distribution natural images, does the rectified ensemble std predict
actual reconstruction error well enough to calibrate?

Prior milestones
----------------
  TTA + BoxDownsample(2):       r ≈ −0.16  (inverted, wrong mechanism)
  church-256 DDPM + Box(2):     r ≈  0.00  (flat, prior-driven)
  ResShift + BoxDownsample(4):  r = +0.21  (first positive, domain mismatch)
  THIS RUN: ResShift + BicubicDownsample(4) + natural images

Method
------
16 natural 256×256 images (vendor/ResShift/testdata/Bicubicx4/gt/,
ILSVRC2012 ImageNet validation — same distribution as ResShift's training)
→ grayscale [0,1]  (luminance: 0.299R + 0.587G + 0.114B)
→ y = BicubicDownsample(4).forward(x_gt)  (circular/FFT bicubic, our operator)
→ ResShiftEngine.ensemble(y, N)  conditioned on lq=y
→ rectify() each member through layer/decompose  (R3: ‖A·x_out−y‖ ≤ 1e-10)
→ pooled calibrate() + is_calibrated()  [both UNCHANGED]

Interior vs edge analysis
-------------------------
Our BicubicDownsample uses circular/FFT boundary; ResShift was trained with
MATLAB imresize (reflect boundary). The mismatch lives within ~16 pixels of
image borders (= 4·scale = kernel half-support in HR). For 256×256 images,
the affected strip is 16 pixels wide on all four sides.
Hypothesis: if circular boundary hurts calibration, r will be LOWER in the
edge strip than in the 224×224 interior. Reported explicitly.

Wall-clock budget
-----------------
  N_IMAGES=16, N_ENSEMBLE=6, ~5.5s/sample → 16×6×5.5 ≈ 528s (≈8.8 min).
  Smaller budget options documented in log; not used here (full run).

Constraints honored
-------------------
R1 : BicubicDownsample(4) for this matched-degradation evaluation.
     (R1 production = BoxDownsample(2); this eval uses matched bicubic.)
R2 : uncertainty = per-pixel ensemble std of x_out_i values.
     Std is computed from null-component spread only (range component
     A⁺y is identical across all members). No pixel content used.
R3 : ‖A·x_out−y‖ ≤ CONSISTENCY_EPS asserted per member per image.
R4 : x_gt used ONLY to compute y (via forward) and evaluate error.
     x_gt is NOT passed to the engine or to rectify().
R5 : image paths, GT SHA, seeds, checkpoint SHA, thresholds all logged.
R6 : HONEST result. Thresholds are fixed; not tuned to produce a verdict.
R9 : BicubicDownsample(4) for this x4 SR calibration evaluation.

Calibration thresholds (FIXED — identical to test_calibrate.py)
----------------------------------------------------------------
  min_pearson_r = 0.9
  max_ece       = 0.3
  min_slope     = 0.5
  max_slope     = 2.0
"""
from __future__ import annotations
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from operators.bicubic import BicubicDownsample
from engine.resshift import ResShiftEngine
from layer.decompose import rectify, CONSISTENCY_EPS
from layer.calibrate import calibrate, is_calibrated

# ---------------------------------------------------------------------------
# Configuration  (R5 — ALL explicit, NO post-hoc tuning)

NATURAL_IMAGES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "vendor", "ResShift",
    "testdata", "Bicubicx4", "gt",
)
SCALE      = 4
H, W       = 256, 256
N_ENSEMBLE = 6           # same as prior smoke test for direct comparison
N_BINS     = 10
MIN_STD    = 1e-6
R3_TOL     = CONSISTENCY_EPS   # 1e-10

# Edge width = 4×scale = half-support of the bicubic kernel in HR pixels.
# Circular vs MATLAB reflect boundary differs within this strip.
EDGE_WIDTH = 4 * SCALE   # = 16 pixels

# is_calibrated() thresholds — IDENTICAL to test_calibrate.py, NOT tuned here
MIN_R    = 0.9
MAX_ECE  = 0.3
MIN_SLOP = 0.5
MAX_SLOP = 2.0

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


# ---------------------------------------------------------------------------
# Helpers

def _load_grayscale(path: str) -> np.ndarray:
    """Load image → grayscale float64 [0,1] at H×W.

    Luminance: L = 0.299R + 0.587G + 0.114B (BT.601 standard).
    Verified shape is 256×256 to match ResShift's input contract.
    """
    img = Image.open(path).convert("RGB")
    if img.size != (W, H):
        raise ValueError(f"Expected {W}×{H}, got {img.size} for {path}")
    arr = np.array(img, dtype=np.float64) / 255.0   # (H, W, 3)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return gray.astype(np.float64)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r of two 1D arrays; returns nan for < 2 elements."""
    if len(a) < 2:
        return float("nan")
    ac = a - a.mean();  bc = b - b.mean()
    denom = np.sqrt((ac**2).sum() * (bc**2).sum())
    return float((ac * bc).sum() / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------

def run() -> None:
    import glob

    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("=" * 64)
    log("Provenanssi — ResShift + BicubicDownsample calibration (R8)")
    log("=" * 64)
    log()

    # ------------------------------------------------------------------
    # 0. Setup operator and engine

    op = BicubicDownsample(SCALE)
    engine = ResShiftEngine(op)
    log()

    log("Configuration (R5)")
    log(f"  model        : resshift_bicsrx4_s4  (UNetModelSwin + VQModelTorch)")
    log(f"  model SHA    : {engine._ckpt_sha256}")
    log(f"  ae SHA       : {engine._ae_sha256}")
    log(f"  conditioning : model.forward(x, t, lq=y)  — cond_lq=True")
    log(f"  operator     : BicubicDownsample(4)  — Keys cubic a=−0.5, circular/FFT")
    log(f"  HR size      : {H}×{W}")
    log(f"  LR size      : {H//SCALE}×{W//SCALE}")
    log(f"  diffusion    : 4-step ResShift exponential schedule")
    log(f"  N_ENSEMBLE   : {N_ENSEMBLE}")
    log(f"  N_BINS       : {N_BINS}")
    log(f"  min_std      : {MIN_STD}")
    log(f"  R3_TOL       : {R3_TOL:.0e}")
    log()
    log("Image source")
    log(f"  dir          : vendor/ResShift/testdata/Bicubicx4/gt/")
    log(f"  content      : ILSVRC2012 ImageNet validation patches")
    log(f"  size         : 256×256 RGB → grayscale  (luminance 0.299R+0.587G+0.114B)")
    log(f"  note         : same distribution as ResShift training — matched domain")
    log()
    log("Wall-clock budget")
    log(f"  N_ENSEMBLE={N_ENSEMBLE} × ~5.5s/sample × N_IMAGES images")
    log()
    log("Thresholds (fixed, identical to test_calibrate.py — NOT tuned here)")
    log(f"  min_pearson_r : {MIN_R}")
    log(f"  max_ece       : {MAX_ECE}")
    log(f"  slope range   : [{MIN_SLOP}, {MAX_SLOP}]")
    log()
    log("R2 confirmation")
    log("  predicted_std = per-pixel std(x_out_i).  x_out_i = A⁺y + null_component_i.")
    log("  Range component A⁺y is IDENTICAL across all ensemble members (depends only")
    log("  on y, not the sample).  std(x_out_i) = std(null_component_i).")
    log("  Uncertainty derives from null-space spread only — never pixel magnitude.")
    log()

    # ------------------------------------------------------------------
    # 1. Discover natural images

    image_paths = sorted(glob.glob(os.path.join(NATURAL_IMAGES_DIR, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {NATURAL_IMAGES_DIR}")
    N_IMAGES = len(image_paths)
    budget_s = N_IMAGES * N_ENSEMBLE * 5.5
    log(f"Images found: {N_IMAGES}")
    log(f"Estimated budget: {N_IMAGES} × {N_ENSEMBLE} × 5.5s ≈ {budget_s:.0f}s ({budget_s/60:.1f} min)")
    log()

    # ------------------------------------------------------------------
    # 2. Time one full image pass first

    log("Timing probe (first image) ...")
    x_gt_probe = _load_grayscale(image_paths[0])
    y_probe = op.forward(x_gt_probe)
    t0 = time.perf_counter()
    _ = engine.restore(y_probe)   # one sample
    t_one = time.perf_counter() - t0
    log(f"  1 sample: {t_one:.1f}s → N={N_ENSEMBLE} per image: {t_one*N_ENSEMBLE:.0f}s")
    log(f"  Revised total estimate: {t_one*N_ENSEMBLE*N_IMAGES:.0f}s ({t_one*N_ENSEMBLE*N_IMAGES/60:.1f} min)")
    log()

    # ------------------------------------------------------------------
    # 3. Main loop: ensemble + rectify per image

    # Per-member collectors (for pooled calibration)
    all_x_outs_per_member: list[list[np.ndarray]] = [[] for _ in range(N_ENSEMBLE)]
    all_x_gt:              list[np.ndarray] = []
    per_image_stats:       list[dict] = []

    # Interior/edge pixel collectors
    all_pred_interior:  list[np.ndarray] = []
    all_err_interior:   list[np.ndarray] = []
    all_pred_edge:      list[np.ndarray] = []
    all_err_edge:       list[np.ndarray] = []

    t_total = 0.0
    max_residual_global = 0.0

    log(f"Running ensembles on {N_IMAGES} images ...")
    log()
    for img_idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        x_gt = _load_grayscale(img_path)   # (256, 256) float64

        # R4: y derived from x_gt; x_gt NOT passed to engine
        y = op.forward(x_gt)               # (64, 64) float64

        # Ensemble (seed per member for reproducibility R5)
        t0 = time.perf_counter()
        x_hats = engine.ensemble(y, N_ENSEMBLE)
        t_img = time.perf_counter() - t0
        t_total += t_img

        # Rectify each member (R3)
        results_i = [rectify(xh, y, op) for xh in x_hats]
        x_outs_i  = [r.x_out for r in results_i]

        # R3 check: assert ‖A·x_out − y‖ ≤ tolerance
        max_res = max(r.residual for r in results_i)
        assert max_res <= R3_TOL, (
            f"R3 violated on image {img_idx} ({img_name}): "
            f"residual={max_res:.3e} > {R3_TOL:.0e}"
        )
        max_residual_global = max(max_residual_global, max_res)

        # Per-image calibration (binned)
        img_cal = calibrate(x_outs_i, x_gt, n_bins=N_BINS, min_predicted_std=MIN_STD)

        # Per-pixel statistics for interior/edge analysis
        stack_i = np.stack(x_outs_i)        # (N, H, W)
        pred_std_i  = stack_i.std(axis=0)   # (H, W)
        actual_err_i = np.abs(stack_i.mean(axis=0) - x_gt)  # (H, W)

        # Interior mask: [EDGE_WIDTH:-EDGE_WIDTH, EDGE_WIDTH:-EDGE_WIDTH]
        interior_mask = np.zeros((H, W), dtype=bool)
        interior_mask[EDGE_WIDTH:-EDGE_WIDTH, EDGE_WIDTH:-EDGE_WIDTH] = True
        edge_mask = ~interior_mask

        # Collect pixels with pred_std above threshold (filter range-space)
        active = pred_std_i >= MIN_STD
        all_pred_interior.append(pred_std_i[interior_mask & active])
        all_err_interior.append(actual_err_i[interior_mask & active])
        all_pred_edge.append(pred_std_i[edge_mask & active])
        all_err_edge.append(actual_err_i[edge_mask & active])

        per_image_stats.append({
            "idx":      img_idx,
            "name":     img_name,
            "gt_sha":   _sha256_array(x_gt),
            "r":        img_cal.pearson_r,
            "slope":    img_cal.slope,
            "ece":      img_cal.ece,
            "n_cal":    img_cal.n_pixels_calibrated,
            "max_res":  max_res,
            "t_s":      t_img,
        })

        all_x_gt.append(x_gt)
        for j, xo in enumerate(x_outs_i):
            all_x_outs_per_member[j].append(xo)

        log(f"  [{img_idx+1:2d}/{N_IMAGES}]  {img_name[:30]:<30}  "
            f"r={img_cal.pearson_r:+.3f}  "
            f"slope={img_cal.slope:+.3f}  "
            f"ECE={img_cal.ece:.3f}  "
            f"max_res={max_res:.1e}  "
            f"({t_img:.0f}s)")

    log()
    log(f"Total time: {t_total:.0f}s  ({t_total/60:.1f} min)")
    log(f"Max R3 residual across all images+members: {max_residual_global:.3e}  (tol: {R3_TOL:.0e})")
    log()

    # ------------------------------------------------------------------
    # 4. Pooled calibration (stack all images along row axis)

    x_outs_pooled = [
        np.vstack(all_x_outs_per_member[j]) for j in range(N_ENSEMBLE)
    ]
    x_gt_pooled = np.vstack(all_x_gt)   # (N_IMAGES*H, W)

    pooled = calibrate(
        x_outs_pooled, x_gt_pooled,
        n_bins=N_BINS, min_predicted_std=MIN_STD,
    )
    verdict = is_calibrated(pooled, MIN_R, MAX_ECE, MIN_SLOP, MAX_SLOP)

    # ------------------------------------------------------------------
    # 5. Interior vs edge analysis (circular-boundary hypothesis)

    pred_int_all  = np.concatenate(all_pred_interior)
    err_int_all   = np.concatenate(all_err_interior)
    pred_edge_all = np.concatenate(all_pred_edge)
    err_edge_all  = np.concatenate(all_err_edge)

    r_interior = _pearson_r(pred_int_all, err_int_all)
    r_edge     = _pearson_r(pred_edge_all, err_edge_all)
    n_interior = len(pred_int_all)
    n_edge     = len(pred_edge_all)

    # ------------------------------------------------------------------
    # 6. Report

    log("=" * 64)
    log("POOLED CALIBRATION RESULT")
    log("=" * 64)
    log(f"  n_images            : {N_IMAGES}")
    log(f"  n_ensemble          : {N_ENSEMBLE}")
    log(f"  n_pixels_calibrated : {pooled.n_pixels_calibrated:,}  "
        f"(of {N_IMAGES * H * W:,} total)")
    log(f"  Pearson r           : {pooled.pearson_r:+.4f}  (need ≥ {MIN_R})")
    log(f"  slope               : {pooled.slope:+.4f}  (need [{MIN_SLOP}, {MAX_SLOP}])")
    log(f"  ECE                 : {pooled.ece:.4f}  (need ≤ {MAX_ECE})")
    log(f"  is_calibrated()     : {'YES' if verdict else 'NO'}")
    log()

    log("Reliability curve (bin_predicted_std → bin_actual_error):")
    log("  bin   pred_std   actual_err   ratio(actual/pred)   n_pixels")
    for b in range(len(pooled.bin_predicted_std)):
        p = pooled.bin_predicted_std[b]
        a = pooled.bin_actual_error[b]
        n = pooled.n_per_bin[b]
        ratio = a / p if p > 0 else float("nan")
        log(f"  {b:2d}    {p:.5f}    {a:.5f}      {ratio:.3f}                {n:6d}")
    log()

    log("Per-image metrics:")
    log(f"  {'img':>3}  {'name':<35}  {'r':>7}  {'slope':>7}  {'ECE':>7}  {'n_cal':>7}  {'max_res':>10}")
    for s in per_image_stats:
        log(f"  {s['idx']+1:3d}  {s['name']:<35}  "
            f"{s['r']:+7.4f}  {s['slope']:+7.4f}  {s['ece']:7.4f}  "
            f"{s['n_cal']:7,}  {s['max_res']:10.2e}")
    log()

    log("Interior vs edge analysis (circular-boundary hypothesis)")
    log(f"  Edge strip: {EDGE_WIDTH} pixels wide on all sides (= 4×scale = kernel half-support)")
    log(f"  Interior : {H - 2*EDGE_WIDTH}×{W - 2*EDGE_WIDTH} = {(H-2*EDGE_WIDTH)*(W-2*EDGE_WIDTH):,} pixels/image")
    log(f"  Edge     : {H*W - (H-2*EDGE_WIDTH)*(W-2*EDGE_WIDTH):,} pixels/image  "
        f"(= {100*(H*W - (H-2*EDGE_WIDTH)*(W-2*EDGE_WIDTH))/(H*W):.1f}% of image)")
    log()
    log(f"  r(std, error) — INTERIOR : {r_interior:+.4f}  (n={n_interior:,})")
    log(f"  r(std, error) — EDGE     : {r_edge:+.4f}  (n={n_edge:,})")
    r_diff = r_interior - r_edge
    log(f"  r_interior − r_edge      : {r_diff:+.4f}")
    if abs(r_diff) < 0.02:
        log("  Interpretation: no meaningful interior/edge difference.")
        log("  Circular boundary mismatch is NOT a significant factor.")
    elif r_diff > 0.02:
        log("  Interpretation: interior has HIGHER r than edge.")
        log("  Consistent with circular-boundary hypothesis: edge pixels see the")
        log(f"  circular/MATLAB mismatch, interior pixels do not.")
        log(f"  Magnitude of effect: Δr = {r_diff:+.4f}")
    else:
        log("  Interpretation: edge has HIGHER r than interior.")
        log("  Inconsistent with circular-boundary hypothesis.")
        log("  Edge structure may help ResShift (e.g. by providing boundary context).")
    log()

    # ------------------------------------------------------------------
    # 7. R6 honest interpretation

    log("=" * 64)
    log("R6 — HONEST INTERPRETATION")
    log("=" * 64)
    log()
    log("Milestone progression:")
    log("  TTA + BoxDownsample(2):       r ≈ −0.16  (inverted — wrong mechanism)")
    log("  church-256 DDPM + Box(2):     r ≈  0.00  (flat — prior-driven, not y-conditioned)")
    log("  ResShift + BoxDownsample(4):  r = +0.21  (first positive — domain mismatch)")
    log(f"  ResShift + BicubicDownsample: r = {pooled.pearson_r:+.4f}  (this run — matched degradation)")
    log()

    r = pooled.pearson_r
    s = pooled.slope
    e = pooled.ece

    if verdict:
        log(f"VERDICT: IS_CALIBRATED = YES")
        log()
        log(f"  All three thresholds met: r={r:+.4f}≥{MIN_R}, "
            f"ECE={e:.4f}≤{MAX_ECE}, slope={s:.4f}∈[{MIN_SLOP},{MAX_SLOP}].")
        log()
        log("  The rectified ResShift ensemble spread IS calibrated: higher predicted std")
        log("  predicts higher actual reconstruction error, at the correct magnitude.")
        log()
        log("  Mechanistic interpretation:")
        log("  ResShift samples the conditional posterior p(x|y) where y is the bicubic")
        log("  LR image. The null-space component of each rectified sample (the invented")
        log("  high-frequency texture beyond what y constrains) varies between seeds.")
        log("  Where y is ambiguous (complex texture, many plausible HR completions),")
        log("  the posterior spread is large AND the reconstruction error is large.")
        log("  Where y strongly constrains x (smooth regions, clean edges), the spread")
        log("  is small AND the error is small. This is calibrated uncertainty.")
        log()
        log("  ResShift + bicubic + natural images crosses the calibration threshold.")
    else:
        log(f"VERDICT: IS_CALIBRATED = NO")
        log()
        failing = []
        if r < MIN_R:
            failing.append(f"Pearson r = {r:+.4f} (need ≥ {MIN_R})")
        if e > MAX_ECE:
            failing.append(f"ECE = {e:.4f} (need ≤ {MAX_ECE})")
        if not (MIN_SLOP <= s <= MAX_SLOP):
            failing.append(f"slope = {s:+.4f} (need [{MIN_SLOP}, {MAX_SLOP}])")

        log("  Which criteria failed:")
        for f_msg in failing:
            log(f"    ✗  {f_msg}")
        log()

        if r > 0.21:
            log(f"  r = {r:+.4f} > +0.21 (prior ResShift+Box milestone): IMPROVEMENT.")
            log("  The matched bicubic degradation improved correlation vs domain mismatch,")
            log("  but calibration threshold (r≥0.9) was not reached.")
        elif r <= 0.21:
            log(f"  r = {r:+.4f} ≤ +0.21: NO improvement over ResShift+Box baseline.")

        log()
        log("  Analysis of remaining gap:")
        if s < MIN_SLOP:
            log(f"  Slope={s:+.4f} < {MIN_SLOP}: OVERCONFIDENT.")
            log("  The model's spread understates actual error.")
            log("  ResShift hallucinations are more wrong than its spread suggests.")
        elif s > MAX_SLOP:
            log(f"  Slope={s:+.4f} > {MAX_SLOP}: UNDERCONFIDENT.")
            log("  The model's spread overstates actual error (conservative).")

        log()
        log("  Possible causes of remaining miscalibration:")
        log("  (a) ResShift's 4-step schedule collapses diversity vs the full diffusion chain.")
        log("  (b) Grayscale conversion loses chromatic information that ResShift used for SR.")
        log("  (c) Circular/FFT boundary in our operator differs from ResShift's MATLAB training.")
        log(f"      (Interior r = {r_interior:+.4f} vs edge r = {r_edge:+.4f}; Δ = {r_diff:+.4f})")
        log("  (d) N=6 ensemble is too small to accurately estimate the true posterior std.")

    log()
    log("=" * 64)
    log("KEY FINDING SUMMARY")
    log("=" * 64)
    log(f"  Pearson r : {r:+.4f}")
    log(f"  Slope     : {s:+.4f}")
    log(f"  ECE       : {e:.4f}")
    log(f"  Verdict   : {'IS_CALIBRATED' if verdict else 'NOT CALIBRATED'}")
    log(f"  Interior r: {r_interior:+.4f}  vs  Edge r: {r_edge:+.4f}  (Δ={r_diff:+.4f})")
    log(f"  R3 max residual: {max_residual_global:.2e}  (tol: {R3_TOL:.0e})")
    log(f"  Runtime: {t_total:.0f}s ({t_total/60:.1f} min)  /  {N_IMAGES} images × {N_ENSEMBLE} members")

    # ------------------------------------------------------------------
    # 8. Save report
    report_path = os.path.join(OUTPUT_DIR, "calibration_resshift_bicubic_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log()
    log(f"Report saved to {report_path}")


if __name__ == "__main__":
    run()
