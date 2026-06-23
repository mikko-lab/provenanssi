#!/usr/bin/env python3
"""
eval/distance_metric.py — Distribution-distance metric + slope correlation.

THREAD: FINDINGS.md §11 item 1.

QUESTION: Does a measured per-image distance from the ResShift training
distribution predict calibration slope?

METRIC (precisely named):
  Cosine distance in ResShift VQ-autoencoder (autoencoder_vq_f4)
  pre-quantization encoder latent space, to the L2-normalized centroid
  of 16 ILSVRC2012 validation image encodings (grayscale pseudo-RGB,
  normalized to [-1, 1]).

  NOT "distance from training distribution" — that would require sampling
  from the full training set. This is distance to a 16-image reference
  centroid in one particular feature space.

LIMITATIONS (must be stated alongside any claimed association):
  · n_ref=16: centroid estimate is noisy; 16 images poorly cover ImageNet
  · Grayscale → pseudo-RGB: color information absent; chromatic ImageNet
    texture features cannot contribute to the distance
  · VQ-AE trained for reconstruction, not as a metric space; pre-quant.
    continuous codes have no guaranteed metric properties
  · n_eval=13: statistically thin — all correlations PRELIMINARY, reported
    with Fisher-z 95% CI and n throughout

STABLE SLOPE SOURCES (from prior results, NOT recomputed here):
  boardwalk, frog_on_log, boy_face, girl_sad, wayuu_woman: N=48 (stability_nscan)
  wood_grain: N=192 (close_findings) — N=48=0.4351 was not converged

7 images lack stable N≥48 slopes; run N=48 calibration here:
  nature_land, dirt_soil, grass_meadow (natural/texture)
  soft_blobs, hard_shapes, linear_grad, radial_grad (synthetic — expected high noise)

CONFOUND: partial correlation controlling for null_frac_gt.
SPLIT: results reported with n=13 (all) and n=9 (non-synthetic only).
"""

import os, sys, time, math
import numpy as np
import torch
from PIL import Image
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from operators.bicubic import BicubicDownsample
from engine.resshift import ResShiftEngine
from layer.decompose import rectify, CONSISTENCY_EPS
from layer.calibrate import calibrate

# ── Config ─────────────────────────────────────────────────────────────────────

SCALE   = 4
H, W    = 256, 256
N_BINS  = 10
MIN_STD = 1e-6
R3_TOL  = CONSISTENCY_EPS
N_NEW   = 48

REPO = os.path.join(os.path.dirname(__file__), "..")
RES  = os.path.join(os.path.dirname(__file__), "research_sources")
REF  = os.path.join(REPO, "vendor", "ResShift", "testdata", "Bicubicx4", "gt")
OUT  = os.path.join(os.path.dirname(__file__), "distance_metric_results.txt")

# ── Eval table ─────────────────────────────────────────────────────────────────
# (label, rel_path, group, prior_slope|None, prior_N|None, known_null_frac|None)
# prior_slope=None → run N=48 calibration in this script
# known_null_frac=None → measure inline

EVAL = [
    ("boardwalk",    "natural/boardwalk_nature.jpg",   "natural",   0.9510,  48,  0.0046),
    ("frog_on_log",  "natural/frog_on_log.jpg",        "natural",   1.4238,  48,  None  ),
    ("nature_land",  "natural/nature_landscape.jpg",   "natural",   None,    None, None ),
    ("boy_face",     "faces/boy_face_venezuela.jpg",   "faces",     3.1829,  48,  0.0049),
    ("girl_sad",     "faces/girl_sad_face.jpg",        "faces",     2.7834,  48,  0.0032),
    ("wayuu_woman",  "faces/wayuu_woman.jpg",          "faces",     1.7483,  48,  0.0162),
    ("wood_grain",   "texture/wood_grain.png",         "texture",   0.5969,  192, 0.0159),
    ("dirt_soil",    "texture/dirt_soil.png",          "texture",   None,    None, None ),
    ("grass_meadow", "texture/grass_meadow.png",       "texture",   None,    None, None ),
    ("soft_blobs",   "synthetic/soft_blobs.png",       "synthetic", None,    None, 0.0014),
    ("hard_shapes",  "synthetic/hard_shapes.png",      "synthetic", None,    None, None ),
    ("linear_grad",  "synthetic/linear_gradient.png",  "synthetic", None,    None, None ),
    ("radial_grad",  "synthetic/radial_gradient.png",  "synthetic", None,    None, None ),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_gray(path: str) -> np.ndarray:
    """Load image → BT.601 grayscale float64 [0,1], resize+center-crop to 256×256."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h:
        new_w, new_h = W, int(round(h * W / w))
    else:
        new_w, new_h = int(round(w * H / h)), H
    img  = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - W) // 2
    top  = (new_h - H) // 2
    img  = img.crop((left, top, left + W, top + H))
    rgb  = np.array(img, dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _vq_features(gray: np.ndarray, engine) -> np.ndarray:
    """Encode grayscale [0,1] (256×256) → 12288-dim pre-quant VQ latent (float32)."""
    x = torch.from_numpy(gray.astype(np.float32))  # (256, 256)
    x = x.unsqueeze(0).expand(3, -1, -1)            # (3, 256, 256) pseudo-RGB
    x = x * 2.0 - 1.0                               # [-1, 1]
    x = x.unsqueeze(0).to(engine._device)           # (1, 3, 256, 256)
    with torch.no_grad():
        h = engine._ae.encode(x)                    # (1, 3, 64, 64)
    return h.squeeze(0).cpu().float().numpy().flatten()  # (12288,)


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def _null_frac_gt(gray: np.ndarray, op) -> float:
    rp   = op.pinv(op.forward(gray))
    null = gray - rp
    return float(np.sum(null**2) / (np.sum(gray**2) + 1e-12))


def _run_calibration(gray: np.ndarray, op, engine, n: int) -> dict:
    """Run N-member ensemble calibration on one image; return slope, r, n."""
    y = op.forward(gray)
    x_hats = engine.ensemble(y, n)
    results = [rectify(xh, y, op) for xh in x_hats]
    for res in results:
        assert res.residual <= R3_TOL, f"R3 violated: {res.residual:.2e}"
    x_outs = [r.x_out for r in results]
    cal = calibrate(x_outs, gray, n_bins=N_BINS, min_predicted_std=MIN_STD)
    return {"slope": cal.slope, "r": cal.pearson_r, "n": n}


def _pearson_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher-z 95% CI for Pearson r. Returns (lo, hi) in r-space."""
    if abs(r) >= 1.0 or n <= 3:
        return (float("nan"), float("nan"))
    z   = math.atanh(r)
    se  = 1.0 / math.sqrt(n - 3)
    z_c = scipy_stats.norm.ppf(1 - alpha / 2)
    return (math.tanh(z - z_c * se), math.tanh(z + z_c * se))


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial correlation r(x, y | z) using closed-form formula."""
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    denom = math.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom < 1e-12:
        return float("nan")
    return float((rxy - rxz * ryz) / denom)


def _corr_block(label: str, distances: np.ndarray, slopes: np.ndarray,
                null_fracs: np.ndarray) -> list[str]:
    """Return formatted lines for one correlation analysis block."""
    n = len(distances)
    r_p, p_p = scipy_stats.pearsonr(distances, slopes)
    lo_p, hi_p = _pearson_ci(r_p, n)
    r_s, p_s = scipy_stats.spearmanr(distances, slopes)
    r_part = _partial_corr(distances, slopes, null_fracs)
    lo_pa, hi_pa = _pearson_ci(r_part, n - 1)  # partial: df = n-1-1 = n-2, conservatively n-1

    lines = [
        f"  [{label}]  n={n}",
        f"    Pearson r(dist, slope)      = {r_p:+.3f}  95% CI [{lo_p:+.3f}, {hi_p:+.3f}]  p={p_p:.3f}",
        f"    Spearman rho(dist, slope)   = {r_s:+.3f}  p={p_s:.3f}",
        f"    Pearson r(null, slope)      = {np.corrcoef(null_fracs, slopes)[0,1]:+.3f}  (confounder)",
        f"    Partial r(dist, slope|null) = {r_part:+.3f}  95% CI [{lo_pa:+.3f}, {hi_pa:+.3f}]  (approx)",
    ]
    return lines


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    t_total = time.perf_counter()
    op = BicubicDownsample(SCALE)

    log("Loading ResShift engine …")
    engine = ResShiftEngine(op)
    log()

    # ── 1. Reference centroid ─────────────────────────────────────────────────
    log("─" * 68)
    log("  STEP 1: VQ encoder features — 16 ILSVRC2012 reference images")
    log("─" * 68)
    import glob
    ref_paths = sorted(glob.glob(os.path.join(REF, "*.png")))
    assert len(ref_paths) == 16, f"Expected 16 reference images, got {len(ref_paths)}"

    ref_feats = []
    for p in ref_paths:
        gray = _load_gray(p)
        feat = _vq_features(gray, engine)
        ref_feats.append(feat)
    ref_feats = np.stack(ref_feats)       # (16, 12288)
    centroid  = ref_feats.mean(axis=0)   # (12288,) raw mean

    # Compute within-reference spread (sanity check)
    ref_dists = [_cosine_dist(f, centroid) for f in ref_feats]
    log(f"  Reference centroid computed from {len(ref_paths)} images")
    log(f"  Within-ref cosine dist: mean={np.mean(ref_dists):.4f}  "
        f"min={np.min(ref_dists):.4f}  max={np.max(ref_dists):.4f}")
    log()

    # ── 2. Eval image features + distances ────────────────────────────────────
    log("─" * 68)
    log("  STEP 2: VQ encoder features + cosine distance — 13 eval images")
    log("─" * 68)
    log(f"  {'Image':<14} {'group':<10} {'cosine_dist':>12}  {'L2_dist':>10}")
    log("  " + "─" * 50)

    distances: dict[str, float] = {}
    l2_dists:  dict[str, float] = {}
    gray_cache: dict[str, np.ndarray] = {}

    for (label, rel, group, prior_slope, prior_n, known_null) in EVAL:
        path = os.path.join(RES, rel)
        gray = _load_gray(path)
        gray_cache[label] = gray
        feat  = _vq_features(gray, engine)
        cdist = _cosine_dist(feat, centroid)
        l2d   = float(np.linalg.norm(feat - centroid))
        distances[label] = cdist
        l2_dists[label]  = l2d
        log(f"  {label:<14} {group:<10} {cdist:>12.4f}  {l2d:>10.2f}")

    log()

    # ── 3. null_frac_gt for all images ────────────────────────────────────────
    log("─" * 68)
    log("  STEP 3: null_frac_gt for all 13 images")
    log("─" * 68)

    null_fracs: dict[str, float] = {}
    for (label, rel, group, prior_slope, prior_n, known_null) in EVAL:
        if known_null is not None:
            null_fracs[label] = known_null
            log(f"  {label:<14}  null_frac_gt = {known_null:.4f}  (from prior measurement)")
        else:
            nf = _null_frac_gt(gray_cache[label], op)
            null_fracs[label] = nf
            log(f"  {label:<14}  null_frac_gt = {nf:.4f}  (measured here)")
    log()

    # ── 4. N=48 calibrations for 7 images ─────────────────────────────────────
    log("─" * 68)
    log(f"  STEP 4: N={N_NEW} calibration for 7 images lacking stable slopes")
    log(f"  Estimated: 7 × {N_NEW} × ~5.5s ≈ {7*N_NEW*5.5:.0f}s  ({7*N_NEW*5.5/60:.0f} min)")
    log("─" * 68)

    new_slopes: dict[str, dict] = {}
    for (label, rel, group, prior_slope, prior_n, known_null) in EVAL:
        if prior_slope is not None:
            continue
        log(f"  [{label}]  running N={N_NEW} ensemble …")
        t0 = time.perf_counter()
        cal = _run_calibration(gray_cache[label], op, engine, N_NEW)
        dt  = time.perf_counter() - t0
        new_slopes[label] = cal
        log(f"    slope={cal['slope']:.4f}  r={cal['r']:+.4f}  ({dt:.0f}s)")
    log()

    # ── 5. Assemble data table ─────────────────────────────────────────────────
    log("─" * 68)
    log("  STEP 5: Data table")
    log("─" * 68)
    log()
    log(f"  {'Image':<14} {'group':<10} {'cosine_dist':>12} {'null_frac':>10} "
        f"{'slope':>8} {'slope_N':>8}  note")
    log("  " + "─" * 72)

    labels_all    = []
    dists_all     = []
    slopes_all    = []
    nulls_all     = []
    slope_Ns_all  = []
    is_synthetic  = []

    for (label, rel, group, prior_slope, prior_n, known_null) in EVAL:
        if prior_slope is not None:
            slope = prior_slope
            sn    = prior_n
            note  = f"N={prior_n} (prior)"
        else:
            cal   = new_slopes[label]
            slope = cal["slope"]
            sn    = N_NEW
            note  = f"N={N_NEW} (this run)  r={cal['r']:+.4f}"
            if group == "synthetic":
                note += "  [high-noise: soft_blobs N12-floor=0.90]" if label == "soft_blobs" else "  [high-noise: synthetic]"

        cdist = distances[label]
        nf    = null_fracs[label]
        log(f"  {label:<14} {group:<10} {cdist:>12.4f} {nf:>10.4f} {slope:>8.4f} {sn:>8}  {note}")

        labels_all.append(label)
        dists_all.append(cdist)
        slopes_all.append(slope)
        nulls_all.append(nf)
        slope_Ns_all.append(sn)
        is_synthetic.append(group == "synthetic")

    dists_all  = np.array(dists_all)
    slopes_all = np.array(slopes_all)
    nulls_all  = np.array(nulls_all)
    is_syn     = np.array(is_synthetic)

    log()

    # ── 6. Statistical analysis ────────────────────────────────────────────────
    log("─" * 68)
    log("  STEP 6: Correlation analysis")
    log("─" * 68)
    log()
    log("  Pearson r with Fisher-z 95% CI. n=13 → SE(z)=1/√10≈0.316,")
    log("  CI width ≈ ±0.6 in r. Treat any finding as PRELIMINARY.")
    log()

    # All 13 images
    for blk in _corr_block("all images, n=13", dists_all, slopes_all, nulls_all):
        log(blk)
    log()

    # Non-synthetic only (n=9)
    mask9 = ~is_syn
    for blk in _corr_block("non-synthetic only, n=9",
                             dists_all[mask9], slopes_all[mask9], nulls_all[mask9]):
        log(blk)
    log()

    # ── 7. Scatter (rank table) ────────────────────────────────────────────────
    log("─" * 68)
    log("  Scatter (sorted by cosine distance)")
    log("─" * 68)
    order = np.argsort(dists_all)
    log(f"  {'rank':>4} {'image':<14} {'group':<10} {'dist':>8} {'slope':>8}")
    log("  " + "─" * 48)
    for rank, idx in enumerate(order, 1):
        syn_flag = " *" if is_syn[idx] else ""
        log(f"  {rank:>4} {labels_all[idx]:<14} {[e[2] for e in EVAL][idx]:<10} "
            f"{dists_all[idx]:>8.4f} {slopes_all[idx]:>8.4f}{syn_flag}")
    log("  (* synthetic group)")
    log()

    # ── 8. Verdict ─────────────────────────────────────────────────────────────
    log("─" * 68)
    log("  VERDICT")
    log("─" * 68)
    log()

    r_all  = scipy_stats.pearsonr(dists_all, slopes_all)[0]
    lo_all, hi_all = _pearson_ci(r_all, len(dists_all))
    r_part_all = _partial_corr(dists_all, slopes_all, nulls_all)
    r_9    = scipy_stats.pearsonr(dists_all[mask9], slopes_all[mask9])[0]
    lo_9, hi_9 = _pearson_ci(r_9, mask9.sum())
    r_part_9 = _partial_corr(dists_all[mask9], slopes_all[mask9], nulls_all[mask9])

    # CI excludes 0?
    ci_excludes_0_all = (lo_all > 0 or hi_all < 0)
    ci_excludes_0_9   = (lo_9  > 0 or hi_9  < 0)

    if ci_excludes_0_all and ci_excludes_0_9 and abs(r_part_all) > 0.3 and abs(r_part_9) > 0.3:
        verdict = "(a) POSITIVE — correlation survives partial-correlation confound check"
    elif ci_excludes_0_all or ci_excludes_0_9:
        verdict = "(b) AMBIGUOUS — CI excludes 0 in one split but not both; confound not resolved"
    else:
        verdict = "(b) NULL / CONFOUNDED — 95% CI includes 0 in both splits"

    log(f"  {verdict}")
    log()
    log(f"  n=13: r={r_all:+.3f}  CI=[{lo_all:+.3f},{hi_all:+.3f}]  partial_r={r_part_all:+.3f}")
    log(f"  n=9:  r={r_9:+.3f}  CI=[{lo_9:+.3f},{hi_9:+.3f}]  partial_r={r_part_9:+.3f}")
    log()
    log("  Interpretation guide:")
    log("    CI includes 0 → cannot rule out sampling noise at n=13")
    log("    partial_r ≈ r → distance adds little beyond null_frac_gt alone")
    log("    partial_r << r → association is largely mediated by null energy")
    log()

    runtime = time.perf_counter() - t_total
    log(f"  Total runtime: {runtime:.0f}s ({runtime/60:.1f} min)")

    # ── 9. Write results ────────────────────────────────────────────────────────
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
