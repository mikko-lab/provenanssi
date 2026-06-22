"""
eval/close_findings.py — Close the two open threads from FINDINGS.md §5.

Thread 1 — wood_grain convergence:
  The N-scan showed wood_grain slope still rising at N=48 (0.1564→0.2645→0.3554→0.4351).
  Extend to N=192 (nested seeds) to determine the converged value and whether it crosses
  the IS_CALIBRATED threshold (slope ≥ 0.5).

Thread 2 — noise floor generalisation:
  The prior noise floor (σ=0.029, range=0.084) was measured on ONE image (boardwalk_nature),
  a mid-null-energy natural scene.  Here we measure the noise floor on three images spanning
  null-space energy:
    high  — wood_grain      (null_frac_gt = 0.0159)
    mid   — boardwalk_nature (null_frac_gt = 0.0046, prior measurement replicated)
    low   — soft_blobs       (null_frac_gt = 0.0014, smooth Gaussian blobs)
  Protocol: 5 non-overlapping N=12 windows (seeds 0–11, 12–23, …, 48–59).
  Reuses wood_grain members 0–59 from the convergence run (no extra GPU time).

Null-space energy proxy (GT projection):
  null_frac_gt = ||(I − A⁺A)x||² / ||x||²
  = fraction of GT image energy that is unrecoverable from the LR measurement.
  Smooth images → low; textured images → high.

R6: report what the numbers show. If the noise floor is much higher for soft_blobs
(slope variance >> boardwalk), the 25× SNR from the prior session was optimistic;
recompute using the worst-case noise floor.

Usage:
    python eval/close_findings.py

Output: printed report + eval/close_findings_results.txt
Runtime estimate: ~330 forward passes × ~1.1s ≈ 360s (6 min) on MPS.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from operators.bicubic import BicubicDownsample
from layer.decompose import rectify
from layer.calibrate import calibrate

# ─── configuration ─────────────────────────────────────────────────────────

SCALE       = 4
N_BINS      = 10
MIN_STD     = 1e-6
TARGET_SIZE = 256

# Fixed thresholds — identical to falsify.py
MIN_R, MAX_ECE, MIN_SLOP, MAX_SLOP = 0.9, 0.3, 0.5, 2.0

RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "research_sources")

# Thread 1: wood_grain convergence
CONVERGENCE_N_VALUES = [48, 96, 144, 192]   # N=48 re-runs as consistency check
CONVERGENCE_MAX_N    = 192

# Thread 2: noise floor generalisation
NOISE_IMAGES = [
    # (label, path, null_frac_gt, note)
    ("wood_grain",
     "texture/wood_grain.png",
     0.0159,
     "high null-space energy; members reused from convergence run"),
    ("boardwalk_nature",
     "natural/boardwalk_nature.jpg",
     0.0046,
     "mid null-space energy; replicates prior session noise floor"),
    ("soft_blobs",
     "synthetic/soft_blobs.png",
     0.0014,
     "low null-space energy; smooth Gaussian blobs — hardest case"),
]
NOISE_N         = 12   # each window size
NOISE_WINDOWS   = 5    # non-overlapping windows → max seed = NOISE_N * NOISE_WINDOWS - 1 = 59
NOISE_MAX_N     = NOISE_N * NOISE_WINDOWS  # = 60


# ─── helpers ─────────────────────────────────────────────────────────────────

def _load_crop(path: str, size: int = TARGET_SIZE) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h:
        new_w, new_h = size, int(round(h * size / w))
    else:
        new_w, new_h = int(round(w * size / h)), size
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    img  = img.crop((left, top, left + size, top + size))
    rgb  = np.array(img, dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _collect(engine, y: np.ndarray, n: int) -> list[np.ndarray]:
    """Collect n rectified x_out members (seeds 0..n-1)."""
    return [engine._sample(y, seed=s) for s in range(n)]


def _calibrate_subset(raw_members, x_gt, n, op):
    """Rectify first n members and calibrate."""
    rects  = [rectify(m, x_gt_approx, op) for m, x_gt_approx in
              zip(raw_members[:n], [None]*n)]
    # Note: rectify signature is rectify(x_hat, y, op) not rectify(x_hat, x_gt, op)
    # We need y not x_gt here — see below
    raise RuntimeError("use _calibrate_from_rects instead")


def _run_image(engine, op, img_path: str, n_collect: int):
    """
    Load image, collect n_collect ResShift members (seeds 0..n-1),
    rectify all, return (x_gt, y, raw_members, rects, x_outs).
    """
    x_gt = _load_crop(img_path)
    y    = op.forward(x_gt)
    raw  = [engine._sample(y, seed=s) for s in range(n_collect)]
    rects = [rectify(m, y, op) for m in raw]
    x_outs = [r.x_out for r in rects]
    return x_gt, y, x_outs


def _null_frac_gt(x_gt: np.ndarray, op) -> float:
    """Fraction of GT energy in the null space: ||(I-A⁺A)x||²/||x||²."""
    rp = op.pinv(op.forward(x_gt))
    null = x_gt - rp
    return float(np.sum(null**2) / (np.sum(x_gt**2) + 1e-12))


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    op = BicubicDownsample(SCALE)

    print("Loading ResShift engine …")
    from engine.resshift import ResShiftEngine
    engine = ResShiftEngine(op)
    print()

    lines: list[str] = []
    t_total = time.perf_counter()

    # ══════════════════════════════════════════════════════════════════════
    # THREAD 1: wood_grain convergence (N=48→96→144→192)
    # ══════════════════════════════════════════════════════════════════════

    wg_path = os.path.join(RESEARCH_DIR, "texture/wood_grain.png")
    print(f"{'═'*60}")
    print(f"  THREAD 1: wood_grain convergence  (collecting {CONVERGENCE_MAX_N} members)")
    print(f"{'═'*60}")

    t0 = time.perf_counter()
    wg_x_gt, wg_y, wg_x_outs = _run_image(engine, op, wg_path, CONVERGENCE_MAX_N)
    dt_wg = time.perf_counter() - t0
    print(f"  Collected {CONVERGENCE_MAX_N} members in {dt_wg:.0f}s")

    wg_null_frac = _null_frac_gt(wg_x_gt, op)
    print(f"  null_frac_gt = {wg_null_frac:.4f}")

    convergence_rows: list[tuple[int, float, float]] = []  # (N, slope, r)
    prev_slope = None
    for N in CONVERGENCE_N_VALUES:
        cal = calibrate(wg_x_outs[:N], wg_x_gt, n_bins=N_BINS, min_predicted_std=MIN_STD)
        delta = abs(cal.slope - prev_slope) if prev_slope is not None else float("nan")
        convergence_rows.append((N, cal.slope, cal.pearson_r))
        delta_str = f"  |Δ|={delta:.4f}" if not np.isnan(delta) else ""
        cal_str   = "✓ IS_CAL" if MIN_SLOP <= cal.slope <= MAX_SLOP else "  —     "
        print(f"  N={N:>3}: slope={cal.slope:.4f}  r={cal.pearson_r:+.4f}"
              f"  {cal_str}{delta_str}")
        prev_slope = cal.slope

    print()

    # ══════════════════════════════════════════════════════════════════════
    # THREAD 2: noise floor across three images
    # ══════════════════════════════════════════════════════════════════════

    print(f"{'═'*60}")
    print(f"  THREAD 2: noise floor generalisation  (5×N={NOISE_N} windows each)")
    print(f"{'═'*60}")

    noise_results: list[dict] = []   # per-image results

    for label, rel_path, expected_null_frac, note in NOISE_IMAGES:
        img_path = os.path.join(RESEARCH_DIR, rel_path)
        print(f"\n  [{label}]  ({note})")

        if label == "wood_grain":
            # Reuse members 0–59 already collected
            x_gt  = wg_x_gt
            x_outs_60 = wg_x_outs[:NOISE_MAX_N]
            null_frac = wg_null_frac
            print(f"  Reusing members 0–{NOISE_MAX_N-1} from convergence run.")
        else:
            t0 = time.perf_counter()
            x_gt, _, x_outs_60 = _run_image(engine, op, img_path, NOISE_MAX_N)
            dt = time.perf_counter() - t0
            null_frac = _null_frac_gt(x_gt, op)
            print(f"  Collected {NOISE_MAX_N} members in {dt:.0f}s")

        print(f"  null_frac_gt = {null_frac:.4f}")

        window_slopes: list[float] = []
        window_rs: list[float] = []
        for rep in range(NOISE_WINDOWS):
            s0, s1 = rep * NOISE_N, (rep + 1) * NOISE_N
            cal = calibrate(x_outs_60[s0:s1], x_gt, n_bins=N_BINS, min_predicted_std=MIN_STD)
            window_slopes.append(cal.slope)
            window_rs.append(cal.pearson_r)
            nan_str = " [NaN]" if np.isnan(cal.slope) else ""
            print(f"    window {rep} (seeds {s0:>2}–{s1-1:<2}): "
                  f"slope={cal.slope:.4f}  r={cal.pearson_r:+.4f}{nan_str}")

        valid = [s for s in window_slopes if not np.isnan(s)]
        if len(valid) >= 2:
            s_mean  = float(np.mean(valid))
            s_std   = float(np.std(valid))
            s_range = float(max(valid) - min(valid))
        else:
            s_mean = s_std = s_range = float("nan")

        print(f"  → mean={s_mean:.4f}  std={s_std:.4f}  range={s_range:.4f}")

        noise_results.append({
            "label":      label,
            "null_frac":  null_frac,
            "window_slopes": window_slopes,
            "mean":  s_mean,
            "std":   s_std,
            "range": s_range,
        })

    total_runtime = time.perf_counter() - t_total

    # ══════════════════════════════════════════════════════════════════════
    # Report
    # ══════════════════════════════════════════════════════════════════════

    lines.append("")
    lines.append("=" * 72)
    lines.append("  CLOSE_FINDINGS — closing open threads from FINDINGS.md §5")
    lines.append("  ResShift + BicubicDownsample(4), nested seeds, fixed thresholds")
    lines.append("=" * 72)
    lines.append("")

    # Thread 1 table
    lines.append("─" * 72)
    lines.append("  THREAD 1: wood_grain slope convergence  (seeds nested, N=48→192)")
    lines.append("─" * 72)
    lines.append("")
    lines.append(f"  Previous N-scan (FINDINGS.md §3c): N=6→48 trajectory")
    lines.append(f"    0.1564 → 0.2645 → 0.3554 → 0.4351  (still rising at N=48)")
    lines.append("")
    lines.append(f"  {'N':>4}  {'slope':>8}  {'r':>8}  {'|Δ(N,N/2)|':>12}  IS_CAL")
    lines.append("  " + "─" * 48)
    prev_s = None
    for N, slope, r in convergence_rows:
        if prev_s is not None and not np.isnan(prev_s):
            delta_s = f"{abs(slope - prev_s):>12.4f}"
        else:
            delta_s = f"{'':>12}"
        in_cal = "YES" if MIN_SLOP <= slope <= MAX_SLOP else " NO"
        lines.append(f"  {N:>4}  {slope:>8.4f}  {r:>+8.4f}  {delta_s}  {in_cal}")
        prev_s = slope
    lines.append("")

    # Convergence verdict
    n48_slope = next(s for (N,s,r) in convergence_rows if N==48)
    n192_slope = next(s for (N,s,r) in convergence_rows if N==192)
    n96_slope  = next(s for (N,s,r) in convergence_rows if N==96)
    n144_slope = next(s for (N,s,r) in convergence_rows if N==144)
    still_rising = (n144_slope < n192_slope) and (n96_slope < n144_slope)
    crossed_threshold = n192_slope >= MIN_SLOP

    lines.append(f"  Consistency check with prior N-scan N=48 value (0.4351):")
    lines.append(f"    This run N=48 slope = {n48_slope:.4f}  "
                 f"→ {'matches' if abs(n48_slope-0.4351)<0.001 else 'MISMATCH — check seeds'}")
    lines.append("")
    if still_rising:
        lines.append(f"  Convergence: STILL RISING at N=192 (0.4351→{n192_slope:.4f}).")
        lines.append(f"  Threshold cross: slope {'≥' if crossed_threshold else '<'} 0.5 at N=192.")
    else:
        lines.append(f"  Convergence: PLATEAU reached (last delta = {abs(n192_slope-n144_slope):.4f}).")
        lines.append(f"  Threshold cross: slope {'≥' if crossed_threshold else '<'} 0.5 at N=192.")
    lines.append("")

    # Thread 2 table
    lines.append("─" * 72)
    lines.append(f"  THREAD 2: noise floor across images spanning null-space energy")
    lines.append(f"  Protocol: 5 × N={NOISE_N} non-overlapping seed windows (seeds 0–{NOISE_MAX_N-1})")
    lines.append("─" * 72)
    lines.append("")
    lines.append(f"  {'Image':<20} {'null_frac_gt':>12}  "
                 f"{'slope mean':>10}  {'std':>7}  {'range':>7}")
    lines.append("  " + "─" * 60)
    for nr in noise_results:
        lines.append(
            f"  {nr['label']:<20} {nr['null_frac']:>12.4f}  "
            f"{nr['mean']:>10.4f}  {nr['std']:>7.4f}  {nr['range']:>7.4f}"
        )
    lines.append("")

    # Per-window detail
    lines.append("  Per-window breakdown:")
    for nr in noise_results:
        lines.append(f"\n  {nr['label']} (null_frac_gt={nr['null_frac']:.4f}):")
        for rep, s in enumerate(nr["window_slopes"]):
            s0, s1 = rep*NOISE_N, (rep+1)*NOISE_N-1
            nan_tag = " [NaN]" if np.isnan(s) else ""
            lines.append(f"    window {rep} seeds {s0:>2}–{s1:<2}: slope={s:.4f}{nan_tag}")
    lines.append("")

    # Does noise floor scale with null-space energy?
    lines.append("─" * 72)
    lines.append("  NOISE FLOOR SCALING WITH NULL-SPACE ENERGY")
    lines.append("─" * 72)
    lines.append("")
    ranges = {nr["label"]: nr["range"] for nr in noise_results}
    prior_boardwalk_range = 0.0844   # from prior session, seeds 0–59
    lines.append(f"  Noise floor (range across 5×N={NOISE_N} windows):")
    for nr in noise_results:
        lines.append(f"    {nr['label']:<20}: range={nr['range']:.4f}  "
                     f"(null_frac_gt={nr['null_frac']:.4f})")
    lines.append(f"\n  Prior session boardwalk range (reference): {prior_boardwalk_range:.4f}")
    lines.append(f"  Consistency check: this session boardwalk range = "
                 f"{ranges.get('boardwalk_nature', float('nan')):.4f}")
    lines.append("")

    worst_range = max(nr["range"] for nr in noise_results if not np.isnan(nr["range"]))
    worst_label = next(nr["label"] for nr in noise_results
                       if abs(nr["range"] - worst_range) < 1e-9)
    lines.append(f"  Worst-case noise floor: range={worst_range:.4f}  [{worst_label}]")

    # ── Revised SNR and verdict ──
    lines.append("")
    lines.append("─" * 72)
    lines.append("  REVISED SNR AND HONEST SIGNAL CHARACTERISATION")
    lines.append("─" * 72)
    lines.append("")

    # Between-group spread at N=48 from prior session (still valid)
    # natural mean 1.1874, faces mean 2.5715, texture N=48 slope
    nat_mean   = 1.1874   # from stability_nscan_results.txt
    face_mean  = 2.5715
    tex_n48    = n48_slope   # = 0.4351 from this run (or whatever it is)

    bg_spread = max(nat_mean, face_mean, tex_n48) - min(nat_mean, face_mean, tex_n48)

    snr_prior   = bg_spread / prior_boardwalk_range
    snr_revised = bg_spread / worst_range if worst_range > 0 else float("inf")

    # Natural↔Faces contrast specifically
    nat_face_diff = face_mean - nat_mean
    snr_nat_face_prior   = nat_face_diff / prior_boardwalk_range
    snr_nat_face_revised = nat_face_diff / worst_range if worst_range > 0 else float("inf")

    lines.append(f"  Between-group spread at N=48 (from prior session, unchanged):")
    lines.append(f"    natural mean  : {nat_mean:.4f}")
    lines.append(f"    faces mean    : {face_mean:.4f}")
    lines.append(f"    texture (wood): {tex_n48:.4f}")
    lines.append(f"    spread        : {bg_spread:.4f}")
    lines.append("")
    lines.append(f"  Prior SNR (best-case, boardwalk noise floor):")
    lines.append(f"    {bg_spread:.4f} / {prior_boardwalk_range:.4f} = {snr_prior:.1f}×")
    lines.append("")
    lines.append(f"  Revised SNR (worst-case, {worst_label} noise floor):")
    lines.append(f"    {bg_spread:.4f} / {worst_range:.4f} = {snr_revised:.1f}×")
    lines.append("")
    lines.append(f"  Natural↔Faces contrast specifically:")
    lines.append(f"    difference    : {nat_face_diff:.4f}")
    lines.append(f"    prior SNR     : {snr_nat_face_prior:.1f}×")
    lines.append(f"    revised SNR   : {snr_nat_face_revised:.1f}×")
    lines.append("")

    # Verdict per contrast
    lines.append("  Verdict per group contrast:")
    contrasts = [
        ("natural↔faces",
         nat_face_diff,
         "faces slope ~2.57 vs natural ~1.19"),
        ("natural↔texture(N=192)",
         abs(nat_mean - n192_slope),
         f"texture(N=192)={n192_slope:.4f} vs natural ~1.19"),
        ("faces↔texture(N=192)",
         abs(face_mean - n192_slope),
         f"faces ~2.57 vs texture(N=192)={n192_slope:.4f}"),
    ]
    for name, diff, desc in contrasts:
        snr_c = diff / worst_range if worst_range > 0 else float("inf")
        hold = "HOLDS above noise" if snr_c > 3 else "WEAK (<3× noise floor)"
        lines.append(f"    {name:<30}: diff={diff:.4f}  SNR={snr_c:.1f}×  → {hold}")
        lines.append(f"      ({desc})")
    lines.append("")

    lines.append("=" * 72)
    lines.append(f"  Total runtime: {total_runtime:.0f}s ({total_runtime/60:.1f} min)")
    lines.append("=" * 72)

    output = "\n".join(lines)
    print(output)

    out_path = os.path.join(os.path.dirname(__file__), "close_findings_results.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
