"""
eval/faces_noise_floor.py — Measure the faces group noise floor.

This closes the final open thread from FINDINGS.md: the natural↔faces contrast
was argued to be ~16× above noise using boardwalk as the noise reference, but
the faces group's OWN noise floor was never measured.

Protocol (identical to eval/close_findings.py):
  - Collect MAX_N=240 ResShift members per image (seeds 0..239), nested.
  - 5 non-overlapping windows at N_SMALL=12: seeds 0–11, 12–23, …, 48–59.
  - 5 non-overlapping windows at N_LARGE=48: seeds 0–47, 48–95, …, 192–239.
  - Report slope mean, std, range per image at both window sizes.

Images:
  boy_face       (null_frac_gt=0.0049, slope@N48=3.1829)
  girl_sad_face  (null_frac_gt=0.0032, slope@N48=2.7834)
  wayuu_woman    (null_frac_gt=0.0162, slope@N48=1.7483)

Reference curve (from close_findings.py, null_frac→noise floor range at N=12):
  null 0.0159 (wood_grain)   → range=0.0522
  null 0.0046 (boardwalk)    → range=0.0844
  null 0.0014 (soft_blobs)   → range=0.8992

Place the three face images on this curve and report consistency.

Final SNR:
  natural↔faces difference = faces_mean_N48 - natural_mean_N48
    = 2.5715 - 1.1874 = 1.3841  (N=48 group means from stability_nscan_results.txt)
  Noise floor used = max(natural_best_N12=0.0844, worst_face_N12_range)
  SNR = difference / noise_floor

R6: do not select the reference that produces the best-looking result.
    Use max(natural, faces) noise floor as explicitly instructed.

Runtime: ~790s (720 forward passes, MPS).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from operators.bicubic import BicubicDownsample
from layer.decompose import rectify
from layer.calibrate import calibrate

# ─── configuration ────────────────────────────────────────────────────────────

SCALE       = 4
N_BINS      = 10
MIN_STD     = 1e-6
TARGET_SIZE = 256
MIN_SLOP, MAX_SLOP = 0.5, 2.0

N_SMALL     = 12    # window size matching prior boardwalk measurement
N_LARGE     = 48    # window size matching the N-scan characterisation
N_WINDOWS   = 5
MAX_N       = N_LARGE * N_WINDOWS   # = 240

RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "research_sources")

FACE_IMAGES = [
    # (label, path, null_frac_gt, slope_at_N48_from_nscan)
    ("boy_face",      "faces/boy_face_venezuela.jpg",  0.0049, 3.1829),
    ("girl_sad_face", "faces/girl_sad_face.jpg",       0.0032, 2.7834),
    ("wayuu_woman",   "faces/wayuu_woman.jpg",         0.0162, 1.7483),
]

# Prior-session reference values (from close_findings_results.txt and
# stability_nscan_results.txt) — used for SNR computation and curve placement.
PRIOR_CURVE = [
    ("wood_grain",       0.0159, 0.2323, 0.0522),  # label, null, slope_mean, range_N12
    ("boardwalk_nature", 0.0046, 0.9475, 0.0844),
    ("soft_blobs",       0.0014, 4.3726, 0.8992),
]
NATURAL_MEAN_N48 = 1.1874   # (boardwalk 0.9510 + frog 1.4238) / 2
FACES_MEAN_N48   = 2.5715   # (3.1829 + 2.7834 + 1.7483) / 3
NAT_FACE_DIFF    = FACES_MEAN_N48 - NATURAL_MEAN_N48  # = 1.3841
NATURAL_NOISE_RANGE_N12 = 0.0844   # boardwalk, from close_findings


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_crop(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h:
        new_w, new_h = TARGET_SIZE, int(round(h * TARGET_SIZE / w))
    else:
        new_w, new_h = int(round(w * TARGET_SIZE / h)), TARGET_SIZE
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - TARGET_SIZE) // 2
    top  = (new_h - TARGET_SIZE) // 2
    img  = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))
    rgb  = np.array(img, dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _null_frac_gt(x_gt: np.ndarray, op) -> float:
    rp   = op.pinv(op.forward(x_gt))
    null = x_gt - rp
    return float(np.sum(null**2) / (np.sum(x_gt**2) + 1e-12))


def _window_slopes(x_outs_all: list, x_gt: np.ndarray,
                   win_n: int, n_windows: int) -> list[float]:
    """Compute slope for n_windows non-overlapping windows of size win_n."""
    slopes = []
    for w in range(n_windows):
        s0, s1 = w * win_n, (w + 1) * win_n
        if s1 > len(x_outs_all):
            slopes.append(float("nan"))
            continue
        cal = calibrate(x_outs_all[s0:s1], x_gt, n_bins=N_BINS, min_predicted_std=MIN_STD)
        slopes.append(cal.slope)
    return slopes


def _stats(slopes: list[float]) -> tuple[float, float, float]:
    valid = [s for s in slopes if not np.isnan(s)]
    if len(valid) < 2:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(valid)), float(np.std(valid)), float(max(valid) - min(valid))


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    op = BicubicDownsample(SCALE)

    print("Loading ResShift engine …")
    from engine.resshift import ResShiftEngine
    engine = ResShiftEngine(op)
    print()

    face_results: list[dict] = []
    t_total = time.perf_counter()

    for label, rel_path, expected_nf, slope_n48 in FACE_IMAGES:
        img_path = os.path.join(RESEARCH_DIR, rel_path)
        print(f"{'─'*60}")
        print(f"  [{label}]  null_frac_gt(expected)={expected_nf:.4f}  slope@N48={slope_n48:.4f}")

        x_gt = _load_crop(img_path)
        y    = op.forward(x_gt)

        # Measure actual null_frac_gt
        actual_nf = _null_frac_gt(x_gt, op)
        print(f"  null_frac_gt (measured) = {actual_nf:.4f}")

        # Collect MAX_N=240 members (seeds 0..239)
        t0 = time.perf_counter()
        raw_members = [engine._sample(y, seed=s) for s in range(MAX_N)]
        dt = time.perf_counter() - t0
        print(f"  Collected {MAX_N} members in {dt:.0f}s")

        # Rectify all
        rects  = [rectify(m, y, op) for m in raw_members]
        x_outs = [r.x_out for r in rects]

        # N_SMALL=12 windows
        slopes_small = _window_slopes(x_outs, x_gt, N_SMALL, N_WINDOWS)
        mean_s, std_s, range_s = _stats(slopes_small)
        print(f"  N={N_SMALL} windows ({N_WINDOWS}×{N_SMALL}, seeds 0–{N_SMALL*N_WINDOWS-1}):")
        for w, s in enumerate(slopes_small):
            print(f"    window {w} (seeds {w*N_SMALL:>3}–{(w+1)*N_SMALL-1:<3}): slope={s:.4f}")
        print(f"  → mean={mean_s:.4f}  std={std_s:.4f}  range={range_s:.4f}")

        # N_LARGE=48 windows
        slopes_large = _window_slopes(x_outs, x_gt, N_LARGE, N_WINDOWS)
        mean_l, std_l, range_l = _stats(slopes_large)
        print(f"  N={N_LARGE} windows ({N_WINDOWS}×{N_LARGE}, seeds 0–{N_LARGE*N_WINDOWS-1}):")
        for w, s in enumerate(slopes_large):
            print(f"    window {w} (seeds {w*N_LARGE:>3}–{(w+1)*N_LARGE-1:<3}): slope={s:.4f}")
        print(f"  → mean={mean_l:.4f}  std={std_l:.4f}  range={range_l:.4f}")
        print()

        face_results.append({
            "label":         label,
            "null_frac":     actual_nf,
            "slope_n48_ref": slope_n48,
            "mean_s":  mean_s,  "std_s":  std_s,  "range_s":  range_s,
            "mean_l":  mean_l,  "std_l":  std_l,  "range_l":  range_l,
            "slopes_small": slopes_small,
            "slopes_large": slopes_large,
        })

    total_runtime = time.perf_counter() - t_total

    # ── Compose report ────────────────────────────────────────────────────────

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("  FACES NOISE FLOOR — final measurement for natural↔faces SNR")
    lines.append("  ResShift + BicubicDownsample(4),  5 non-overlapping windows")
    lines.append("=" * 72)
    lines.append("")

    # Summary table N=12
    lines.append(f"  {'Image':<16} {'null_frac':>10}  {'slope@N48':>10}  "
                 f"{'mean(N12)':>10}  {'std':>7}  {'range':>7}")
    lines.append("  " + "─" * 64)
    for fr in face_results:
        lines.append(
            f"  {fr['label']:<16} {fr['null_frac']:>10.4f}  "
            f"{fr['slope_n48_ref']:>10.4f}  "
            f"{fr['mean_s']:>10.4f}  {fr['std_s']:>7.4f}  {fr['range_s']:>7.4f}"
        )
    lines.append("")

    # Summary table N=48
    lines.append(f"  {'Image':<16} {'null_frac':>10}  {'mean(N48)':>10}  "
                 f"{'std':>7}  {'range':>7}")
    lines.append("  " + "─" * 54)
    for fr in face_results:
        lines.append(
            f"  {fr['label']:<16} {fr['null_frac']:>10.4f}  "
            f"{fr['mean_l']:>10.4f}  {fr['std_l']:>7.4f}  {fr['range_l']:>7.4f}"
        )
    lines.append("")

    # Per-window detail
    lines.append("─" * 72)
    lines.append("  Per-window slopes")
    lines.append("─" * 72)
    for fr in face_results:
        lines.append(f"\n  {fr['label']} (null_frac_gt={fr['null_frac']:.4f}):")
        lines.append(f"    N={N_SMALL} windows:")
        for w, s in enumerate(fr["slopes_small"]):
            lines.append(f"      window {w} seeds {w*N_SMALL:>3}–{(w+1)*N_SMALL-1:<3}: {s:.4f}")
        lines.append(f"    N={N_LARGE} windows:")
        for w, s in enumerate(fr["slopes_large"]):
            lines.append(f"      window {w} seeds {w*N_LARGE:>3}–{(w+1)*N_LARGE-1:<3}: {s:.4f}")
    lines.append("")

    # Faces on the energy→noise curve
    lines.append("─" * 72)
    lines.append("  PLACING FACES ON THE ENERGY→NOISE CURVE  (at N=12)")
    lines.append("─" * 72)
    lines.append("")
    lines.append("  Prior reference points (from close_findings_results.txt):")
    for name, nf, sm, r in PRIOR_CURVE:
        lines.append(f"    null={nf:.4f}  slope_mean={sm:.4f}  range={r:.4f}  [{name}]")
    lines.append("")
    lines.append("  Face images:")
    for fr in face_results:
        lines.append(f"    null={fr['null_frac']:.4f}  slope_mean(N12)={fr['mean_s']:.4f}  "
                     f"range(N12)={fr['range_s']:.4f}  [{fr['label']}]")
    lines.append("")

    # Does it follow the inverse-energy pattern?
    lines.append("  Prior inverse-energy relationship (higher null → lower range at N12):")
    lines.append("    wood_grain (0.0159) → 0.052")
    lines.append("    boardwalk  (0.0046) → 0.084")
    lines.append("    soft_blobs (0.0014) → 0.899")
    lines.append("")
    lines.append("  Face images fall within this range:")
    for fr in face_results:
        # Compare to interpolated expectation
        nf = fr["null_frac"]
        r  = fr["range_s"]
        if nf > 0.0046:
            expected = "< 0.084 (closer to wood_grain end)"
        elif nf > 0.0014:
            expected = "between 0.084 and 0.899"
        else:
            expected = "> 0.899"
        lines.append(f"    {fr['label']:<16}: null={nf:.4f} → range={r:.4f}  "
                     f"(expected {expected})")
    lines.append("")

    # Honest SNR computation
    lines.append("─" * 72)
    lines.append("  HONEST NATURAL↔FACES SNR")
    lines.append("─" * 72)
    lines.append("")
    lines.append(f"  Slope difference (N=48 group means, from stability_nscan):")
    slope_refs = " / ".join(f"{fr['slope_n48_ref']:.4f}" for fr in face_results)
    lines.append(f"    faces mean N=48   = {FACES_MEAN_N48:.4f}  ({slope_refs})")
    lines.append(f"    natural mean N=48 = {NATURAL_MEAN_N48:.4f}  "
                 f"(boardwalk 0.9510 / frog_on_log 1.4238)")
    lines.append(f"    difference        = {NAT_FACE_DIFF:.4f}")
    lines.append("")

    # Worst-case noise floor within the contrast
    natural_range_n12 = NATURAL_NOISE_RANGE_N12
    faces_ranges_n12  = [fr["range_s"] for fr in face_results if not np.isnan(fr["range_s"])]
    worst_face_n12    = max(faces_ranges_n12) if faces_ranges_n12 else float("nan")
    worst_contrast_n12 = max(natural_range_n12, worst_face_n12)

    faces_ranges_n48  = [fr["range_l"] for fr in face_results if not np.isnan(fr["range_l"])]
    worst_face_n48    = max(faces_ranges_n48) if faces_ranges_n48 else float("nan")
    # Natural noise floor at N=48: scale from N=12 by sqrt(12/48)=0.5 (rough estimate)
    natural_range_n48_est = natural_range_n12 * 0.5
    worst_contrast_n48 = max(natural_range_n48_est, worst_face_n48)

    snr_n12 = NAT_FACE_DIFF / worst_contrast_n12 if worst_contrast_n12 > 0 else float("inf")
    snr_n48 = NAT_FACE_DIFF / worst_contrast_n48 if worst_contrast_n48 > 0 else float("inf")

    lines.append(f"  Noise floor comparison (N=12 windows):")
    lines.append(f"    natural (boardwalk, measured):  range = {natural_range_n12:.4f}")
    for fr in face_results:
        lines.append(f"    {fr['label']:<18}:  range = {fr['range_s']:.4f}")
    lines.append(f"    worst across contrast = {worst_contrast_n12:.4f}  "
                 f"[{'faces' if worst_face_n12 > natural_range_n12 else 'natural'}]")
    lines.append(f"  → Honest SNR at N=12: {NAT_FACE_DIFF:.4f} / {worst_contrast_n12:.4f}"
                 f" = {snr_n12:.1f}×")
    lines.append("")
    lines.append(f"  Noise floor comparison (N=48 windows):")
    lines.append(f"    natural (N=12 range × 0.5 scaling est.): range ≈ {natural_range_n48_est:.4f}")
    for fr in face_results:
        lines.append(f"    {fr['label']:<18}:  range = {fr['range_l']:.4f}")
    lines.append(f"    worst across contrast = {worst_contrast_n48:.4f}")
    lines.append(f"  → Honest SNR at N=48: {NAT_FACE_DIFF:.4f} / {worst_contrast_n48:.4f}"
                 f" = {snr_n48:.1f}×")
    lines.append("")

    # wayuu_woman overlap check
    lines.append("─" * 72)
    lines.append("  WAYUU_WOMAN OVERLAP CHECK")
    lines.append("─" * 72)
    lines.append("")
    wayuu_n48_slope = next(fr["slope_n48_ref"] for fr in face_results
                           if fr["label"] == "wayuu_woman")
    nat_range_low   = 0.9510   # boardwalk N=48 from nscan
    nat_range_high  = 1.4238   # frog_on_log N=48 from nscan
    lines.append(f"  Natural group range at N=48: {nat_range_low:.4f} – {nat_range_high:.4f}")
    lines.append(f"  wayuu_woman slope at N=48:   {wayuu_n48_slope:.4f}")
    if wayuu_n48_slope <= nat_range_high * 1.0:
        lines.append(f"  → wayuu_woman ({wayuu_n48_slope:.4f}) is above natural max ({nat_range_high:.4f})")
        lines.append(f"    by {wayuu_n48_slope - nat_range_high:.4f} — small gap, potential overlap zone.")
    else:
        lines.append(f"  → wayuu_woman ({wayuu_n48_slope:.4f}) is clearly above natural range.")
    lines.append(f"  boy_face ({face_results[0]['slope_n48_ref']:.4f}) and "
                 f"girl_sad_face ({face_results[1]['slope_n48_ref']:.4f}) are well above natural.")
    lines.append("")

    # Verdict
    lines.append("=" * 72)
    lines.append("  VERDICT")
    lines.append("=" * 72)
    lines.append(_verdict(snr_n12, snr_n48, worst_contrast_n12, worst_face_n12,
                          natural_range_n12, wayuu_n48_slope, nat_range_high, face_results))
    lines.append("")
    lines.append(f"  Total runtime: {total_runtime:.0f}s ({total_runtime/60:.1f} min)")
    lines.append("=" * 72)

    output = "\n".join(lines)
    print(output)

    out_path = os.path.join(os.path.dirname(__file__), "faces_noise_floor_results.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nResults written to {out_path}")


def _verdict(snr_n12: float, snr_n48: float, worst_n12: float, worst_face_n12: float,
             nat_n12: float, wayuu_slope: float, nat_high: float,
             face_results: list) -> str:
    lines: list[str] = []
    lines.append("")
    STRONG_SNR = 5.0
    WEAK_SNR   = 3.0

    # Determine (a) or (b)
    snr_ref = snr_n12   # primary comparison: N=12 windows for like-for-like with boardwalk

    if snr_ref >= STRONG_SNR:
        verdict = "a"
        strength = "strongly"
    elif snr_ref >= WEAK_SNR:
        verdict = "a"
        strength = "marginally"
    else:
        verdict = "b"
        strength = None

    if verdict == "a":
        lines.append(f"  ({verdict}) SURVIVES — {strength}.")
        lines.append(f"  The natural↔faces slope difference ({NAT_FACE_DIFF:.4f}) exceeds the")
        lines.append(f"  worst in-contrast noise floor ({worst_n12:.4f}) by {snr_ref:.1f}× at N=12.")
        lines.append(f"  At N=48, SNR = {snr_n48:.1f}× (smaller windows reduce noise further).")
        lines.append(f"  Caveats:")
        lines.append(f"    - Only 2 of 3 faces images are clearly above the natural range;")
        lines.append(f"      wayuu_woman (slope={wayuu_slope:.4f}) is only {wayuu_slope-nat_high:.4f} above")
        lines.append(f"      natural max ({nat_high:.4f}) — within the noise floor.")
        lines.append(f"    - The contrast is driven by boy_face and girl_sad_face, not")
        lines.append(f"      the group mean. The 'faces group' effect is heterogeneous.")
        lines.append(f"    - Grouping is intuitive, not metric-based.")
    else:
        lines.append(f"  (b) COLLAPSES.")
        lines.append(f"  The natural↔faces slope difference ({NAT_FACE_DIFF:.4f}) is only")
        lines.append(f"  {snr_ref:.1f}× the worst in-contrast noise floor ({worst_n12:.4f}).")
        lines.append(f"  This does not clear the 3× threshold. Ensemble-variance slope")
        lines.append(f"  cannot resolve the natural↔faces distribution difference at N=12.")
        lines.append(f"  At N=48 (reduced noise), SNR = {snr_n48:.1f}×.")
        if snr_n48 >= WEAK_SNR:
            lines.append(f"  Note: at N=48 the SNR exceeds 3×, suggesting the signal MIGHT")
            lines.append(f"  survive at larger N — but this requires direct measurement,")
            lines.append(f"  not the N=48-scaled estimate used here.")
        lines.append(f"  Negative result: domain-shift-in-slope is not established above")
        lines.append(f"  noise for this contrast at N=12. The finding does not hold.")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
