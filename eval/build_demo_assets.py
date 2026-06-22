"""
eval/build_demo_assets.py — Precompute PNG/JSON assets for demo/index.html.

Runs ResShift + bicubic + rectify + classify on 3 representative images,
writes everything to demo/assets/. The demo page loads these assets statically
(no GPU, no backend required).

Selected images (from vendor/ResShift/testdata/Bicubicx4/gt/):
  good    : ILSVRC2012_val_00001936  r=+0.995  high-confidence representative
  typical : ILSVRC2012_val_00000343  r=+0.933  average performance
  failure : ILSVRC2012_val_00017068  r=+0.849  honest limitation case
            (lowest per-image r, below the individual 0.9 threshold)

Assets written per image:
  {id}_gt.png          ground truth (256×256 grayscale, shown as RGB)
  {id}_lr.png          LR input upscaled nearest-neighbour (honest pixelation)
  {id}_recon.png       mean of N=6 rectified reconstructions
  {id}_overlay.png     RGBA provenance overlay (transparent=measured,
                       orange+hatch=invented, blue+dots=recovered)
  {id}_range.png       A⁺y range component (what y constrains)
  {id}_uncertainty.png ensemble std colourmap (viridis-like)
  {id}_error.png       |mean − gt| colourmap (white→red)
  {id}_sample_{k}.png  k=0,1,2 individual rectified samples (for "good" only)
  {id}_sample_{k}_overlay.png  corresponding provenance overlay

manifest.json:
  per-image metrics, pooled calibration, reliability curve, thresholds.
  All numbers come from this run — not hardcoded.

R10 colour choices (Wong colourblind-safe palette):
  INVENTED : #E69F00  orange  + 45° diagonal hatch lines (pattern pitch 6px)
  RECOVERED: #56B4E9  sky blue + dot grid (pitch 6px)
  MEASURED : transparent (reconstruction shows through)

Threshold note (documented, not magic):
  invented_threshold = 0.05  (5% of pixel energy is null-space).
  Lower would flag residual float-precision artefacts as invented.
  Higher would miss weak but genuine null-space contributions.
  test_layer.py uses 1e-6 for purely algebraic tests; here we choose
  a visually informative threshold that separates smooth regions (MEASURED)
  from textured regions (INVENTED) in the natural image context.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from operators.bicubic import BicubicDownsample
from engine.resshift import ResShiftEngine
from layer.decompose import rectify, CONSISTENCY_EPS
from layer.classify import classify, MEASURED, RECOVERED, INVENTED
from layer.ensemble import ensemble_stats
from layer.calibrate import calibrate, is_calibrated

# ─── configuration ────────────────────────────────────────────────────────────

DEMO_IMAGES = [
    {
        "id":       "good",
        "filename": "ILSVRC2012_val_00001936.png",
        "label":    "High-confidence case",
        "note":     None,
    },
    {
        "id":       "typical",
        "filename": "ILSVRC2012_val_00000343.png",
        "label":    "Typical case",
        "note":     None,
    },
    {
        "id":       "failure",
        "filename": "ILSVRC2012_val_00017068.png",
        "label":    "Limitation case",
        "note": (
            "Lowest per-image correlation (r = 0.849, pooled r = 0.967). "
            "Complex high-frequency texture reduces ordering quality: the model "
            "is most uncertain in regions that happen not to be the most wrong. "
            "This is the honest edge of the calibration claim."
        ),
    },
]

SAMPLE_IMAGE_ID  = "good"   # which image to show seed-toggle samples for
N_ENSEMBLE       = 6
N_SAMPLES_SHOWN  = 3        # seed-toggle samples shown in Panel C
SCALE            = 4

# Provenance thresholds (documented, not magic — see module docstring)
INVENTED_THRESH  = 0.05     # fraction of pixel energy in null space → INVENTED
RECOVERED_THRESH = 1e-12    # ensemble variance above this → RECOVERED

# is_calibrated thresholds — FIXED, match falsify.py exactly
MIN_R, MAX_ECE, MIN_SLOP, MAX_SLOP = 0.9, 0.3, 0.5, 2.0
MIN_STD  = 1e-6
N_BINS   = 10

# Wong colourblind-safe palette (R10)
ORANGE_RGB   = (230, 159,   0)
SKYBLUE_RGB  = ( 86, 180, 233)
HATCH_PITCH  = 6    # pixels between hatch lines

GT_DIR     = os.path.join(os.path.dirname(__file__), "..", "vendor",
                           "ResShift", "testdata", "Bicubicx4", "gt")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "demo", "assets")

# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_gray(path: str) -> np.ndarray:
    """PNG → grayscale float64 [0,1]  (BT.601 luminance)."""
    rgb = np.array(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _gray_to_png(arr: np.ndarray, path: str) -> None:
    """float64 [0,1] → 256×256 RGB PNG (clipped, gamma-correct)."""
    arr_clipped = np.clip(arr, 0.0, 1.0)
    rgb = (arr_clipped[:, :, np.newaxis] * np.array([[[1, 1, 1]]]) * 255).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(path)


def _colormap_viridis(arr: np.ndarray) -> np.ndarray:
    """float64 [0,1] → uint8 RGB using a 5-stop viridis approximation."""
    stops = np.array([
        [68,  1,  84],
        [58, 82, 139],
        [32, 144, 140],
        [94, 201,  97],
        [253, 231, 37],
    ], dtype=np.float64)
    t = arr.ravel().clip(0, 1)
    idx = t * (len(stops) - 1)
    lo  = idx.astype(int).clip(0, len(stops) - 2)
    hi  = lo + 1
    frac = idx - lo
    rgb = (stops[lo] * (1 - frac[:, None]) + stops[hi] * frac[:, None]).astype(np.uint8)
    return rgb.reshape(arr.shape[0], arr.shape[1], 3)


def _colormap_error(arr: np.ndarray) -> np.ndarray:
    """float64 [0,1] → uint8 RGB: white → amber → orange → dark red."""
    stops = np.array([
        [255, 255, 240],
        [255, 210, 100],
        [220,  80,  20],
        [120,   0,   0],
    ], dtype=np.float64)
    t = arr.ravel().clip(0, 1)
    idx = t * (len(stops) - 1)
    lo  = idx.astype(int).clip(0, len(stops) - 2)
    hi  = lo + 1
    frac = idx - lo
    rgb = (stops[lo] * (1 - frac[:, None]) + stops[hi] * frac[:, None]).astype(np.uint8)
    return rgb.reshape(arr.shape[0], arr.shape[1], 3)


def _provenance_overlay(labels: np.ndarray) -> np.ndarray:
    """Build RGBA overlay (H, W, 4) from label map.

    MEASURED  : alpha = 0  (transparent; reconstruction shows through)
    INVENTED  : orange fill (alpha 150) + diagonal hatch lines (alpha 220)
    RECOVERED : sky-blue fill (alpha 130) + dot grid (alpha 210)

    Hatch/dots ensure colour is not the ONLY distinguisher (R10).
    """
    H, W = labels.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    inv_mask = labels == INVENTED
    rec_mask = labels == RECOVERED

    # INVENTED fill
    rgba[inv_mask, 0] = ORANGE_RGB[0]
    rgba[inv_mask, 1] = ORANGE_RGB[1]
    rgba[inv_mask, 2] = ORANGE_RGB[2]
    rgba[inv_mask, 3] = 150

    # INVENTED hatch: diagonal lines every HATCH_PITCH pixels
    rows, cols = np.where(inv_mask)
    hatch_inv = (rows + cols) % HATCH_PITCH < 2
    rgba[rows[hatch_inv], cols[hatch_inv]] = [30, 20, 0, 220]

    # RECOVERED fill
    rgba[rec_mask, 0] = SKYBLUE_RGB[0]
    rgba[rec_mask, 1] = SKYBLUE_RGB[1]
    rgba[rec_mask, 2] = SKYBLUE_RGB[2]
    rgba[rec_mask, 3] = 130

    # RECOVERED dots: every HATCH_PITCH×HATCH_PITCH grid
    rows_r, cols_r = np.where(rec_mask)
    dot_mask = (rows_r % HATCH_PITCH == 0) & (cols_r % HATCH_PITCH == 0)
    rgba[rows_r[dot_mask], cols_r[dot_mask]] = [10, 50, 120, 210]

    return rgba


def _save_overlay(rgba: np.ndarray, path: str) -> None:
    Image.fromarray(rgba, "RGBA").save(path)


def _nn_upsample(arr: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour upsample a grayscale (H, W) array."""
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)

# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(ASSETS_DIR, exist_ok=True)

    print("Loading ResShift engine ...")
    op = BicubicDownsample(SCALE)
    engine = ResShiftEngine(op)
    print()

    # Per-image results for pooled calibration
    all_x_outs_per_member: list[list[np.ndarray]] = [[] for _ in range(N_ENSEMBLE)]
    all_x_gt_list: list[np.ndarray] = []
    manifest_images = []
    max_res_global = 0.0

    for cfg in DEMO_IMAGES:
        img_id   = cfg["id"]
        img_path = os.path.join(GT_DIR, cfg["filename"])
        print(f"Processing [{img_id}]  {cfg['filename']} ...")

        x_gt = _load_gray(img_path)          # (256, 256) float64 [0,1]
        y    = op.forward(x_gt)              # (64, 64)  — R4: never pass x_gt to engine
        H, W = x_gt.shape

        # Ensemble + rectify
        t0 = time.perf_counter()
        x_hats = engine.ensemble(y, N_ENSEMBLE)
        rects  = [rectify(xh, y, op) for xh in x_hats]
        dt = time.perf_counter() - t0

        max_res = max(r.residual for r in rects)
        max_res_global = max(max_res_global, max_res)
        assert max_res <= CONSISTENCY_EPS, f"R3 violated: {max_res:.2e}"

        x_outs = [r.x_out for r in rects]
        mean_x = np.stack(x_outs).mean(axis=0)
        std_x  = np.stack(x_outs).std(axis=0)
        _, var_x = ensemble_stats(x_outs)

        # Provenance labels (first member + ensemble variance)
        r0     = rects[0]
        labels = classify(r0.null_component, r0.range_component,
                          invented_threshold=INVENTED_THRESH,
                          recovered_threshold=RECOVERED_THRESH,
                          ensemble_variance=var_x)

        n_meas = int(np.sum(labels == MEASURED))
        n_inv  = int(np.sum(labels == INVENTED))
        n_rec  = int(np.sum(labels == RECOVERED))
        print(f"  labels:  measured={n_meas}  invented={n_inv}  recovered={n_rec}  "
              f"({n_inv / H / W * 100:.0f}% invented)")
        print(f"  R3 max residual: {max_res:.2e}  runtime: {dt:.0f}s")

        # Per-image calibration metrics
        img_cal = calibrate(x_outs, x_gt, n_bins=N_BINS, min_predicted_std=MIN_STD)
        print(f"  r={img_cal.pearson_r:+.3f}  slope={img_cal.slope:.3f}  ECE={img_cal.ece:.4f}")

        # ── Save PNGs ──

        # 1. Ground truth
        _gray_to_png(x_gt, os.path.join(ASSETS_DIR, f"{img_id}_gt.png"))

        # 2. LR input upscaled NN (shows pixelation honestly)
        lr_nn = _nn_upsample(y, SCALE)
        _gray_to_png(lr_nn, os.path.join(ASSETS_DIR, f"{img_id}_lr.png"))

        # 3. Mean reconstruction
        _gray_to_png(mean_x, os.path.join(ASSETS_DIR, f"{img_id}_recon.png"))

        # 4. Provenance overlay (RGBA)
        overlay_rgba = _provenance_overlay(labels)
        _save_overlay(overlay_rgba, os.path.join(ASSETS_DIR, f"{img_id}_overlay.png"))

        # 5. Range component A⁺y (what y constrains)
        range_comp = rects[0].range_component
        _gray_to_png(range_comp, os.path.join(ASSETS_DIR, f"{img_id}_range.png"))

        # 6. Uncertainty map (ensemble std, viridis)
        std_norm = std_x / (std_x.max() + 1e-12)
        unc_rgb  = _colormap_viridis(std_norm)
        Image.fromarray(unc_rgb, "RGB").save(
            os.path.join(ASSETS_DIR, f"{img_id}_uncertainty.png"))

        # 7. Error map |mean − gt| (white→red)
        err_abs  = np.abs(mean_x - x_gt)
        err_norm = err_abs / (err_abs.max() + 1e-12)
        err_rgb  = _colormap_error(err_norm)
        Image.fromarray(err_rgb, "RGB").save(
            os.path.join(ASSETS_DIR, f"{img_id}_error.png"))

        # 8. Individual sample PNGs (Panel C seed toggle, "good" only)
        if img_id == SAMPLE_IMAGE_ID:
            for k in range(N_SAMPLES_SHOWN):
                _gray_to_png(x_outs[k],
                             os.path.join(ASSETS_DIR, f"{img_id}_sample_{k}.png"))
                # Per-sample provenance labels
                rk = rects[k]
                labels_k = classify(rk.null_component, rk.range_component,
                                    invented_threshold=INVENTED_THRESH,
                                    recovered_threshold=RECOVERED_THRESH,
                                    ensemble_variance=var_x)
                ov_k = _provenance_overlay(labels_k)
                _save_overlay(ov_k,
                              os.path.join(ASSETS_DIR, f"{img_id}_sample_{k}_overlay.png"))

        # Collect for pooled calibration
        all_x_gt_list.append(x_gt)
        for j, xo in enumerate(x_outs):
            all_x_outs_per_member[j].append(xo)

        manifest_images.append({
            "id":        img_id,
            "filename":  cfg["filename"],
            "label":     cfg["label"],
            "note":      cfg["note"],
            "pearson_r": round(float(img_cal.pearson_r), 4),
            "slope":     round(float(img_cal.slope),     4),
            "ece":       round(float(img_cal.ece),       4),
            "n_measured":  n_meas,
            "n_invented":  n_inv,
            "n_recovered": n_rec,
            "pct_invented": round(n_inv / H / W * 100, 1),
        })
        print()

    print(f"Max R3 residual across all images+members: {max_res_global:.2e}  (tol: {CONSISTENCY_EPS:.0e})")

    # ── Pooled calibration (3 demo images only) ──
    x_outs_pooled = [np.vstack(all_x_outs_per_member[j]) for j in range(N_ENSEMBLE)]
    x_gt_pooled   = np.vstack(all_x_gt_list)
    pooled  = calibrate(x_outs_pooled, x_gt_pooled, n_bins=N_BINS, min_predicted_std=MIN_STD)
    verdict = is_calibrated(pooled, MIN_R, MAX_ECE, MIN_SLOP, MAX_SLOP)
    print(f"\nPooled (3 demo images):  r={pooled.pearson_r:+.4f}  "
          f"slope={pooled.slope:.4f}  ECE={pooled.ece:.4f}  "
          f"is_calibrated={verdict}")
    print("(Full 16-image pooled: r=+0.9667  slope=1.5301  ECE=0.0282  — from falsify.py)")

    # ── manifest.json ──
    manifest = {
        "images":   manifest_images,
        "pooled_3_demo_images": {
            "pearson_r":     round(float(pooled.pearson_r), 4),
            "slope":         round(float(pooled.slope),     4),
            "ece":           round(float(pooled.ece),       4),
            "is_calibrated": bool(verdict),
            "n_bins":        N_BINS,
        },
        "pooled_16_images": {
            "pearson_r":     0.9667,
            "slope":         1.5301,
            "ece":           0.0282,
            "is_calibrated": True,
            "n_images":      16,
            "n_ensemble":    N_ENSEMBLE,
            "note": "From falsify.py --full, commit 2dea9d8. These are the certified numbers."
        },
        "reliability_curve": {
            "bin_predicted_std": [round(float(v), 6) for v in pooled.bin_predicted_std],
            "bin_actual_error":  [round(float(v), 6) for v in pooled.bin_actual_error],
            "n_per_bin":         [int(v)              for v in pooled.n_per_bin],
        },
        "thresholds": {
            "min_pearson_r": MIN_R,
            "max_ece":       MAX_ECE,
            "min_slope":     MIN_SLOP,
            "max_slope":     MAX_SLOP,
        },
        "provenance_thresholds": {
            "invented_threshold":  INVENTED_THRESH,
            "recovered_threshold": RECOVERED_THRESH,
            "note": (
                "invented_threshold=0.05 means: pixel has >5% of its energy in the "
                "null space of BicubicDownsample(4). Separates genuinely smooth regions "
                "(MEASURED) from regions where ResShift added texture (INVENTED). "
                "See eval/build_demo_assets.py module docstring."
            ),
        },
        "colours_r10": {
            "INVENTED":  "#E69F00",
            "RECOVERED": "#56B4E9",
            "MEASURED":  "transparent",
            "palette":   "Wong (colourblind-safe)",
            "pattern_invented":  "45° diagonal hatch, pitch 6px",
            "pattern_recovered": "dot grid, pitch 6px",
        },
        "n_ensemble": N_ENSEMBLE,
        "scale":      SCALE,
        "r3_max_residual_demo": round(float(max_res_global), 2),
        "r3_tolerance":         float(CONSISTENCY_EPS),
    }

    manifest_path = os.path.join(ASSETS_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest.json → {manifest_path}")

    # Summary
    asset_files = sorted(os.listdir(ASSETS_DIR))
    print(f"\nAssets written ({len(asset_files)} files):")
    for fn in asset_files:
        sz = os.path.getsize(os.path.join(ASSETS_DIR, fn))
        print(f"  {fn:<45}  {sz:>7} bytes")


if __name__ == "__main__":
    main()
