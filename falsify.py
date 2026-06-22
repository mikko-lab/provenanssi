#!/usr/bin/env python3
"""
falsify.py — Provenanssi project gate (§1, §9).

Runs the §1 kill-test end to end and emits a single GREEN / RED verdict.
EXIT 0 + "FALSIFY: GREEN" only if every check passes.
EXIT 1 + "FALSIFY: RED" with the first failing check named, otherwise.

Checks
------
1. OPERATOR CORRECTNESS (R7): A·A⁺·A = A and (A⁺A)² = A⁺A at 1e-12,
   all four operators (Box, Mask, Deblur, Bicubic).
   Logic reused from tests/test_operators.py.

2. DATA CONSISTENCY (R3): rectify output satisfies ‖A·x_out−y‖ ≤ 1e-10
   for every operator (oracle) + ResShift real output (full mode).
   NOTE: BicubicDownsample R3 was only asserted in eval/calibrate_resshift.py
   (eval script), never in a standalone test. First CI assertion here.

3. R2 STRUCTURAL INVARIANT: std(x_out_i) == std(null_component_i) pixel-by-
   pixel at tol=1e-12. Algebraic identity (range component is constant across
   ensemble members). The direct pixel-by-pixel check was NEVER explicitly
   asserted before — test_layer.py proved var(range_parts)≈0, which IMPLIES
   the identity, but the direct assertion was absent. First check here.

4. CLASSIFIER HONESTY: zero false positives (no null content → nothing
   INVENTED) and zero false negatives (known null region → all flagged).
   Logic reused from tests/test_layer.py TestClassify.

5. CALIBRATION (R8): well-calibrated oracle passes is_calibrated() (machinery
   works); miscalibrated oracle FAILS is_calibrated() (metric has teeth, R6).
   Full mode additionally runs ResShift + BicubicDownsample(4) calibration on
   16 natural ImageNet images and asserts IS_CALIBRATED = YES.
   Logic reused from tests/test_calibrate.py and eval/calibrate_resshift.py.

KNOWN LIMITATIONS (R6) are always printed regardless of verdict.

Usage
-----
    python falsify.py          # fast mode (default): checks 1–5, no GPU, ~5s
    python falsify.py --fast   # same as above
    python falsify.py --full   # adds ResShift calibration, ~115s

Fast mode
---------
All infrastructure checks + oracle calibration. No GPU required.
ResShift calibration skipped; last known result (commit 2dea9d8) is displayed.

Full mode
---------
Everything in fast mode PLUS ResShift + bicubic SR calibration on 16 images.
Requires GPU/MPS and weights in weights/. Last run: 105s on Apple M-series.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from operators.superres import BoxDownsample
from operators.inpaint import MaskOperator
from operators.deblur import CircularBlur
from operators.bicubic import BicubicDownsample
from engine.oracle import OracleEngine
from layer.decompose import rectify, CONSISTENCY_EPS
from layer.classify import classify, INVENTED
from layer.calibrate import calibrate, is_calibrated
from eval.synthetic import make_null_calibration_case, NullRegion

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (all documented, none are magic numbers)

TOL       = 1e-12          # R7 operator pseudo-inverse tolerance
R3_TOL    = CONSISTENCY_EPS  # = 1e-10  data-consistency tolerance
R2_TOL    = 1e-12          # R2 std identity tolerance

INVENTED_THRESH  = 1e-6    # classifier null_fraction threshold
RECOVERED_THRESH = 1e-12   # classifier ensemble_variance threshold

# is_calibrated() thresholds — FIXED, identical to test_calibrate.py, NOT tuned
MIN_R    = 0.9
MAX_ECE  = 0.3
MIN_SLOP = 0.5
MAX_SLOP = 2.0

ORACLE_N_ENSEMBLE = 200   # large enough for stable sample mean/std on 12×12
ORACLE_N_BINS     = 3     # one bin per oracle region → clean separation

RESSHIFT_N_ENSEMBLE = 6
RESSHIFT_N_BINS     = 10
RESSHIFT_MIN_STD    = 1e-6

_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vendor", "ResShift", "testdata", "Bicubicx4", "gt",
)

# ─────────────────────────────────────────────────────────────────────────────
# Reporting state

_failures: list[str] = []
_section_name: str = ""


def _header(title: str) -> None:
    global _section_name
    _section_name = title
    print(f"\n{'─' * 68}")
    print(f"  {title}")
    print("─" * 68)


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    _failures.append(f"[{_section_name}]  {msg}")


def _note(msg: str) -> None:
    print(f"        {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers

def _rand(shape: tuple, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape)


def _gauss_kernel(size: int = 5, sigma: float = 0.5) -> np.ndarray:
    c = np.arange(size) - size // 2
    rr, cc = np.meshgrid(c, c, indexing="ij")
    k = np.exp(-(rr**2 + cc**2) / (2 * sigma**2))
    return k / k.sum()


def _half_mask(H: int, W: int) -> np.ndarray:
    """Top half observed (1), bottom half hole (0)."""
    m = np.ones((H, W), dtype=np.float64)
    m[H // 2:, :] = 0.0
    return m


def _load_gray(path: str) -> np.ndarray:
    """Load PNG → grayscale float64 [0,1]  (BT.601 luminance)."""
    from PIL import Image
    rgb = np.array(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Operator Correctness (R7)

def check_1_operator_correctness() -> None:
    """A·A⁺·A = A and (A⁺A)² = A⁺A at tol=1e-12, all four operators.

    Logic reused from tests/test_operators.py (same formulas, same tolerance).
    Operators tested: BoxDownsample(2), MaskOperator, CircularBlur(Gauss),
    BicubicDownsample(4).
    """
    _header("1 — Operator Correctness (R7)  tol = 1e-12")

    configs = [
        #  label                          operator                                shape
        ("BoxDownsample(2)",          BoxDownsample(2),                          (24, 24)),
        ("MaskOperator(half 12×12)",  MaskOperator(_half_mask(12, 12)),          (12, 12)),
        ("CircularBlur(Gauss-5σ0.5)", CircularBlur(_gauss_kernel(5, 0.5), 1e-6),(12, 12)),
        ("BicubicDownsample(4)",      BicubicDownsample(4),                      (24, 24)),
    ]

    for label, op, shape in configs:
        x = _rand(shape)

        # A·A⁺·A == A
        err_aap = float(np.max(np.abs(op.forward(op.pinv(op.forward(x))) - op.forward(x))))
        tag = f"{label:<32}"
        if err_aap <= TOL:
            _ok(f"{tag}  A·A⁺·A = A       max_err = {err_aap:.2e}")
        else:
            _fail(f"{tag}  A·A⁺·A ≠ A       max_err = {err_aap:.2e}  (tol = {TOL:.0e})")

        # (A⁺A)² == A⁺A
        Px = op.project(x)
        err_idem = float(np.max(np.abs(op.project(Px) - Px)))
        if err_idem <= TOL:
            _ok(f"{tag}  (A⁺A)² = A⁺A    max_err = {err_idem:.2e}")
        else:
            _fail(f"{tag}  (A⁺A)² ≠ A⁺A    max_err = {err_idem:.2e}  (tol = {TOL:.0e})")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Data Consistency (R3)

def check_2_data_consistency(resshift_engine=None) -> None:
    """‖A·x_out − y‖ ≤ 1e-10 for rectified oracle output on all operators.

    Logic reused from tests/test_layer.py TestRectify.

    NEW ASSERTION: BicubicDownsample rectify was only asserted inside
    eval/calibrate_resshift.py (per-image, eval script) — never in a reusable
    test.  This is the first standalone CI assertion for that path.
    """
    _header("2 — Data Consistency (R3)  tol = 1e-10")
    _note("NEW: BicubicDownsample R3 was only asserted in eval/calibrate_resshift.py,")
    _note("     not in test_layer.py. First reusable CI assertion here (check 2).")

    oracle_configs = [
        ("BoxDownsample(2)",          BoxDownsample(2),                          (24, 24)),
        ("MaskOperator(half 12×12)",  MaskOperator(_half_mask(12, 12)),          (12, 12)),
        ("CircularBlur(Gauss-5σ0.5)", CircularBlur(_gauss_kernel(5, 0.5), 1e-6),(12, 12)),
        ("BicubicDownsample(4)",      BicubicDownsample(4),                      (24, 24)),
    ]

    for label, op, shape in oracle_configs:
        x_gt = _rand(shape, seed=7)
        y = op.forward(x_gt)
        oracle = OracleEngine(op, null_sigma=1.0, seed=0)
        r = rectify(oracle.restore(y), y, op)
        tag = f"{label:<32}"
        if r.residual <= R3_TOL:
            _ok(f"{tag}  residual = {r.residual:.2e}")
        else:
            _fail(f"{tag}  residual = {r.residual:.2e}  (tol = {R3_TOL:.0e})")

    # ResShift path (full mode only)
    if resshift_engine is None:
        return

    _note("Running one ResShift sample for R3 verification (full mode) ...")
    imgs = sorted(glob.glob(os.path.join(_IMAGES_DIR, "*.png")))
    if not imgs:
        _fail("ResShift+Bicubic path: no images in vendor/ResShift/testdata/Bicubicx4/gt/")
        return

    x_gt = _load_gray(imgs[0])
    op = resshift_engine._op
    y = op.forward(x_gt)
    t0 = time.perf_counter()
    x_hat = resshift_engine.restore(y)
    dt = time.perf_counter() - t0
    r = rectify(x_hat, y, op)
    tag = f"ResShift+Bicubic(4) [{dt:.1f}s]"
    if r.residual <= R3_TOL:
        _ok(f"{tag:<32}  residual = {r.residual:.2e}")
    else:
        _fail(f"{tag}  residual = {r.residual:.2e}  (tol = {R3_TOL:.0e})")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — R2 Structural Invariant

def check_3_r2_invariant() -> None:
    """std(x_out_i) == std(null_component_i) pixel-by-pixel, tol = 1e-12.

    Algebraic identity: x_out_i = range + null_i where range = A⁺y is constant
    across all members.  Therefore std(x_out_i) = std(null_i) exactly.

    NEW ASSERTION: this direct pixel-by-pixel equality was NEVER explicitly
    asserted in any prior test.  test_layer.py proved var(range_parts) ≈ 0,
    which IMPLIES the identity by linearity, but the direct form
    "std(x_out) == std(null)" was absent.  First explicit check here.
    """
    _header("3 — R2 Structural Invariant: std(x_out_i) = std(null_i)  tol = 1e-12")
    _note("NEW ASSERTION: was implicit (range_var≈0 ⟹ identity) but never directly tested.")
    _note("Algebraic proof: x_out_i = A⁺y + null_i; A⁺y constant ⟹ std(x_out_i)=std(null_i).")

    configs = [
        ("BoxDownsample(2)",       BoxDownsample(2),                (24, 24)),
        ("MaskOperator(half)",     MaskOperator(_half_mask(12, 12)),(12, 12)),
        ("BicubicDownsample(4)",   BicubicDownsample(4),            (24, 24)),
    ]

    N = 6
    for label, op, shape in configs:
        x_gt = _rand(shape, seed=9)
        y = op.forward(x_gt)
        oracle = OracleEngine(op, null_sigma=1.0, seed=17)

        x_outs:     list[np.ndarray] = []
        null_comps: list[np.ndarray] = []
        for _ in range(N):
            r = rectify(oracle.restore(y), y, op)
            x_outs.append(r.x_out)
            null_comps.append(r.null_component)

        std_xout = np.stack(x_outs).std(axis=0)
        std_null = np.stack(null_comps).std(axis=0)
        err = float(np.max(np.abs(std_xout - std_null)))

        tag = f"{label:<32}"
        if err <= R2_TOL:
            _ok(f"{tag}  std(x_out) = std(null)  max_err = {err:.2e}")
        else:
            _fail(f"{tag}  R2 violated  max_err = {err:.2e}  (tol = {R2_TOL:.0e})")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — Classifier Honesty

def check_4_classifier_honesty() -> None:
    """Zero false positives and zero false negatives on oracle output.

    Logic reused from tests/test_layer.py TestClassify:
      - Zero FP test: zero null content → no pixel labelled INVENTED.
      - Zero FN test: nonzero null exactly in holes → all holes INVENTED,
        no observed pixels INVENTED.
    """
    _header("4 — Classifier Honesty (oracle guarantees)")

    H, W = 12, 12
    mask_arr = _half_mask(H, W)   # rows [0..5] observed, rows [6..11] hole
    op = MaskOperator(mask_arr)
    oracle = OracleEngine(op)
    x_gt = _rand((H, W), seed=3)
    y = op.forward(x_gt)

    hole_mask = (mask_arr == 0)
    obs_mask  = (mask_arr == 1)
    n_hole = int(hole_mask.sum())

    # 4a: Zero false positives — zero null content → nothing INVENTED
    r0 = rectify(oracle.restore(y, np.zeros((H, W))), y, op)
    labels0 = classify(r0.null_component, r0.range_component,
                       invented_threshold=INVENTED_THRESH,
                       recovered_threshold=RECOVERED_THRESH)
    n_fp = int(np.sum(labels0 == INVENTED))
    if n_fp == 0:
        _ok(f"Zero false positives: 0 / {H * W} pixels INVENTED for zero null content")
    else:
        _fail(f"False positives: {n_fp} pixels INVENTED despite zero null content")

    # 4b: Zero false negatives — full hole null → all holes flagged, none observed
    r1 = rectify(oracle.restore(y, (1.0 - mask_arr) * 1.0), y, op)
    labels1 = classify(r1.null_component, r1.range_component,
                       invented_threshold=INVENTED_THRESH,
                       recovered_threshold=RECOVERED_THRESH)
    n_fn  = int(np.sum(labels1[hole_mask] != INVENTED))
    n_fp2 = int(np.sum(labels1[obs_mask]  == INVENTED))

    if n_fn == 0:
        _ok(f"Zero false negatives: all {n_hole} hole pixels flagged INVENTED")
    else:
        _fail(f"False negatives: {n_fn} / {n_hole} hole pixels NOT flagged INVENTED")

    if n_fp2 == 0:
        _ok(f"Zero false positives (observed region): 0 observed pixels flagged INVENTED")
    else:
        _fail(f"False positives (observed): {n_fp2} observed pixels wrongly flagged INVENTED")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — Calibration (R8)

def check_5_calibration(resshift_engine=None) -> None:
    """Oracle calibration (always) + ResShift calibration (full mode).

    5a: Well-calibrated oracle passes is_calibrated() (machinery works).
        Logic reused from tests/test_calibrate.py TestMonotonicRelationship.

    5b: Miscalibrated oracle FAILS is_calibrated() (metric has teeth, R6).
        Logic reused from tests/test_calibrate.py TestMiscalibrationDetected.

    5c [full mode only]: ResShift + BicubicDownsample(4) on 16 ImageNet images.
        Asserts IS_CALIBRATED = YES against FIXED thresholds.
        Prints r, slope, ECE.
        Logic reused from eval/calibrate_resshift.py.

    Calibration thresholds (FIXED — identical to test_calibrate.py):
        min_pearson_r = 0.9,  max_ece = 0.3,  slope ∈ [0.5, 2.0]
    """
    _header("5 — Calibration (R8)")

    # Setup: same mask and region geometry as test_calibrate.py
    H, W = 12, 12
    mask_arr = _half_mask(H, W)
    op_mask  = MaskOperator(mask_arr)
    x_shape  = (H, W)

    base = np.zeros((H, W), dtype=bool)
    ra = base.copy(); ra[6:8,  :] = True   # rows 6–7
    rb = base.copy(); rb[8:10, :] = True   # rows 8–9
    rc = base.copy(); rc[10:12,:] = True   # rows 10–11

    # 5a: well-calibrated oracle (bias == spread → actual_error ≈ predicted_std)
    well_regions = [
        NullRegion(mask=ra, bias=0.5, spread=0.5),
        NullRegion(mask=rb, bias=1.0, spread=1.0),
        NullRegion(mask=rc, bias=1.5, spread=1.5),
    ]
    case_good = make_null_calibration_case(op_mask, x_shape, well_regions, ORACLE_N_ENSEMBLE, seed=0)
    res_good  = calibrate(case_good.x_outs, case_good.x_gt, n_bins=ORACLE_N_BINS, min_predicted_std=1e-6)
    v_good    = is_calibrated(res_good, MIN_R, MAX_ECE, MIN_SLOP, MAX_SLOP)

    if v_good:
        _ok(f"5a Oracle well-calibrated: is_calibrated=True  "
            f"r={res_good.pearson_r:+.3f}  slope={res_good.slope:.3f}  ECE={res_good.ece:.4f}")
    else:
        _fail(f"5a Oracle well-calibrated: is_calibrated=False  "
              f"r={res_good.pearson_r:+.3f}  slope={res_good.slope:.3f}  ECE={res_good.ece:.4f}  "
              f"(calibration machinery broken)")

    # 5b: miscalibrated oracle (bias INVERTED vs spread → metric must reject)
    bad_regions = [
        NullRegion(mask=ra, bias=1.5, spread=0.5),   # high error, low spread
        NullRegion(mask=rb, bias=1.0, spread=1.0),
        NullRegion(mask=rc, bias=0.5, spread=1.5),   # low error, high spread
    ]
    case_bad = make_null_calibration_case(op_mask, x_shape, bad_regions, ORACLE_N_ENSEMBLE, seed=1)
    res_bad  = calibrate(case_bad.x_outs, case_bad.x_gt, n_bins=ORACLE_N_BINS, min_predicted_std=1e-6)
    v_bad    = is_calibrated(res_bad, MIN_R, MAX_ECE, MIN_SLOP, MAX_SLOP)

    if not v_bad:
        _ok(f"5b Miscalibration detected: is_calibrated=False  "
            f"r={res_bad.pearson_r:+.3f}  slope={res_bad.slope:.3f}  ECE={res_bad.ece:.4f}")
    else:
        _fail(f"5b Miscalibration NOT detected: is_calibrated=True  "
              f"r={res_bad.pearson_r:+.3f}  slope={res_bad.slope:.3f}  ECE={res_bad.ece:.4f}  "
              f"(metric has no teeth — R6 violated)")

    # 5c: ResShift calibration
    if resshift_engine is None:
        _note("")
        _note("5c: ResShift calibration SKIPPED (--fast mode; run --full to verify end-to-end)")
        _note("    Last known result (commit 2dea9d8, 16 images × 6 members, 105s):")
        _note("      Pearson r = +0.9667  (threshold ≥ 0.9)")
        _note("      Slope     =  1.5301  (threshold [0.5, 2.0])")
        _note("      ECE       =  0.0282  (threshold ≤ 0.3)")
        _note("      is_calibrated() : YES")
        return

    _note("")
    _note("5c: ResShift + BicubicDownsample(4) calibration on natural images ...")
    imgs = sorted(glob.glob(os.path.join(_IMAGES_DIR, "*.png")))
    if not imgs:
        _fail("5c: no images in vendor/ResShift/testdata/Bicubicx4/gt/")
        return

    op_bicubic   = resshift_engine._op
    per_member:  list[list[np.ndarray]] = [[] for _ in range(RESSHIFT_N_ENSEMBLE)]
    all_x_gt:    list[np.ndarray] = []
    max_residual = 0.0
    t0 = time.perf_counter()

    for img_path in imgs:
        x_gt   = _load_gray(img_path)
        y      = op_bicubic.forward(x_gt)
        x_hats = resshift_engine.ensemble(y, RESSHIFT_N_ENSEMBLE)
        rects  = [rectify(xh, y, op_bicubic) for xh in x_hats]
        for ri in rects:
            if ri.residual > R3_TOL:
                _fail(f"5c R3 violated on {os.path.basename(img_path)}: "
                      f"residual = {ri.residual:.2e}  (tol = {R3_TOL:.0e})")
                return
            max_residual = max(max_residual, ri.residual)
        all_x_gt.append(x_gt)
        for j, ri in enumerate(rects):
            per_member[j].append(ri.x_out)

    t_cal = time.perf_counter() - t0

    x_outs_pooled = [np.vstack(per_member[j]) for j in range(RESSHIFT_N_ENSEMBLE)]
    x_gt_pooled   = np.vstack(all_x_gt)
    pooled  = calibrate(x_outs_pooled, x_gt_pooled, n_bins=RESSHIFT_N_BINS,
                        min_predicted_std=RESSHIFT_MIN_STD)
    verdict = is_calibrated(pooled, MIN_R, MAX_ECE, MIN_SLOP, MAX_SLOP)

    r_v = pooled.pearson_r
    s_v = pooled.slope
    e_v = pooled.ece

    _note(f"    n_images={len(imgs)}  n_ensemble={RESSHIFT_N_ENSEMBLE}  "
          f"runtime={t_cal:.0f}s  max_R3_residual={max_residual:.2e}")
    _note(f"    Pearson r = {r_v:+.4f}  (threshold ≥ {MIN_R})")
    _note(f"    Slope     = {s_v:+.4f}  (threshold [{MIN_SLOP}, {MAX_SLOP}])")
    _note(f"    ECE       = {e_v:.4f}  (threshold ≤ {MAX_ECE})")

    if verdict:
        _ok(f"5c ResShift calibration: IS_CALIBRATED = YES  "
            f"r={r_v:+.4f}  slope={s_v:.4f}  ECE={e_v:.4f}")
    else:
        failing = []
        if r_v < MIN_R:
            failing.append(f"r={r_v:+.4f} < {MIN_R}")
        if e_v > MAX_ECE:
            failing.append(f"ECE={e_v:.4f} > {MAX_ECE}")
        if not (MIN_SLOP <= s_v <= MAX_SLOP):
            failing.append(f"slope={s_v:.4f} not in [{MIN_SLOP}, {MAX_SLOP}]")
        _fail(f"5c ResShift calibration: IS_CALIBRATED = NO  "
              f"failing: {', '.join(failing)}")

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN LIMITATIONS (R6)

def _print_limitations() -> None:
    """R6: always printed regardless of GREEN/RED."""
    print(f"\n{'═' * 68}")
    print("KNOWN LIMITATIONS (R6)  — printed on every run; do not fail the gate")
    print("═" * 68)
    print()
    print("1. Uncertainty magnitude is well-ordered but underconfident")
    print("   slope = 1.53 (not 1.0): the ensemble std correctly identifies")
    print("   which pixels are uncertain but understates the actual error by")
    print("   ~1.53× on average. Reliability curve: actual_error/pred_std")
    print("   ranges from 5.4× at lowest-std bin to 1.6× at highest-std bin.")
    print("   Calibration passes (slope ∈ [0.5, 2.0]) but perfect magnitude")
    print("   calibration (slope = 1.0) has not been demonstrated.")
    print()
    print("2. Scope: ResShift (bicubic SR ×4) only")
    print("   Calibration is proven for ResShift conditioned on a bicubic LR")
    print("   image. The other three operators (box SR, inpainting, deblur)")
    print("   have the proven rectification layer but no calibrated learned")
    print("   model. Extending calibration to those paths requires a matched")
    print("   conditional model and a separate calibration eval.")
    print()
    print("3. Circular vs MATLAB boundary condition")
    print("   BicubicDownsample uses circular/FFT boundary; ResShift was")
    print("   trained with MATLAB imresize (reflect). The mismatch lives in")
    print("   a ~16-pixel border strip. Empirically tested (commit 2dea9d8):")
    print("   r_interior = +0.454 vs r_edge = +0.478 (Δ = −0.024).")
    print("   The circular-boundary effect is negligible and does not affect")
    print("   the calibration verdict.")
    print()
    print("4. Ensemble size N = 6")
    print("   Calibration evaluated with N = 6 members per image. Smaller N")
    print("   increases sampling noise in pred_std. The N = 6 result is stable")
    print("   (15/16 images individually above r = 0.93 per-image; pooled")
    print("   r = +0.97). N has not been swept to characterise the N-dependence.")
    print("═" * 68)

# ─────────────────────────────────────────────────────────────────────────────
# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="falsify.py",
        description="Provenanssi project gate (§1, §9) — GREEN / RED verdict",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fast", action="store_true",
                       help="Fast mode (default): checks 1–5a/b, no GPU. ~5s.")
    modes.add_argument("--full", action="store_true",
                       help="Full mode: adds ResShift calibration. ~115s.")
    args = parser.parse_args()

    full_mode = args.full

    print("=" * 68)
    print("falsify.py — Provenanssi project gate (§1, §9)")
    mode_label = "FULL (ResShift calibration included)" if full_mode else "FAST (ResShift skipped)"
    print(f"mode    : {mode_label}")
    print(f"tol R7  : {TOL:.0e}   tol R3 : {R3_TOL:.0e}   tol R2 : {R2_TOL:.0e}")
    print(f"cal thr : r≥{MIN_R}  ECE≤{MAX_ECE}  slope∈[{MIN_SLOP},{MAX_SLOP}]  (FIXED)")
    print("=" * 68)

    resshift_engine = None
    if full_mode:
        print("\nLoading ResShift engine (full mode) ...")
        from engine.resshift import ResShiftEngine
        resshift_engine = ResShiftEngine(BicubicDownsample(4))

    t_start = time.perf_counter()

    check_1_operator_correctness()
    check_2_data_consistency(resshift_engine)
    check_3_r2_invariant()
    check_4_classifier_honesty()
    check_5_calibration(resshift_engine)

    _print_limitations()

    t_total = time.perf_counter() - t_start
    print(f"\nRuntime : {t_total:.1f}s  ({mode_label})")
    print()

    if _failures:
        print(f"FALSIFY: RED")
        print(f"  First failing check : {_failures[0]}")
        if len(_failures) > 1:
            print(f"  Also failed ({len(_failures) - 1} more):")
            for f in _failures[1:]:
                print(f"    {f}")
        sys.exit(1)
    else:
        print(f"FALSIFY: GREEN  [{mode_label}]")
        sys.exit(0)


if __name__ == "__main__":
    main()
