"""
eval/stability_nscan.py — Slope stability vs ensemble size (N-scan).

Tests whether calibration slope is a stable measurement at feasible N,
or primarily sampling noise.  Answers three questions:

  1. N-SCAN: does slope converge as N grows from 6 → 12 → 24 → 48?
  2. FACES CONVERGENCE: does within-group spread in slope shrink with N,
     or do images stay separated? (Test whether N=6 spread was noise.)
  3. NOISE FLOOR: how much does slope wobble across 5 independent N=12
     draws from the same image, at fixed N?

EXCLUDED: synthetic group.  Reason: linear_gradient / radial_gradient /
soft_blobs / hard_shapes have near-zero null-space energy after rectify on
smooth inputs.  The ensemble std is dominated by high-frequency sampling
artifacts rather than genuine null-space variation, making slope either
degenerate or dominated by the artefact.  Characterising this is a
separate problem; it does not bear on the natural/faces/texture question.

Seed strategy (nested):
  N=6  ← seeds 0..5
  N=12 ← seeds 0..11  (strictly includes the N=6 members)
  N=24 ← seeds 0..23
  N=48 ← seeds 0..47
  All seeds are deterministic (torch.manual_seed + torch.mps.manual_seed).

Noise floor strategy:
  One image (boardwalk_nature.jpg).  Run N=60 (seeds 0..59).
  Five non-overlapping N=12 windows: seeds 0-11, 12-23, 24-35, 36-47, 48-59.
  Slope spread across the 5 windows = sampling noise at N=12.

Images used:
  natural  : boardwalk_nature.jpg  (slope=0.953 at N=6)
  natural  : frog_on_log.jpg       (slope=1.309 at N=6)
  faces    : boy_face_venezuela.jpg (slope=3.124 at N=6)
  faces    : girl_sad_face.jpg      (slope=3.045 at N=6)
  faces    : wayuu_woman.jpg        (slope=1.472 at N=6)  ← within-group outlier
  texture  : wood_grain.png         (slope=0.156 at N=6)

R6 (research version): report what the numbers show.
A null result ("slope does not stabilise") is the correct answer if true.
Do not select seeds or subsets to manufacture stability.

Usage
-----
    python eval/stability_nscan.py

Output: printed report + eval/stability_nscan_results.txt
Runtime on MPS: ~7–10 min (300 total forward passes × ~1.4s each)
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
from layer.calibrate import calibrate, CalibrationResult

# ─── configuration (fixed, not tuned) ────────────────────────────────────────

SCALE        = 4
N_BINS       = 10
MIN_STD      = 1e-6
TARGET_SIZE  = 256

N_SCAN_VALUES   = [6, 12, 24, 48]
MAX_N           = 48              # seeds 0..MAX_N-1 for main N-scan images
NOISE_IMAGE     = "boardwalk_nature.jpg"
NOISE_MAX_N     = 60              # seeds 0..59 for the noise floor image
NOISE_N         = 12              # each noise-floor window size
NOISE_REPEATS   = 5               # number of non-overlapping windows

RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "research_sources")

# Images in run order.  boardwalk listed first — it gets extra seeds for
# the noise floor measurement and is processed once.
IMAGES = [
    # (label,            group,    relative path under RESEARCH_DIR)
    ("boardwalk_nature", "natural",  "natural/boardwalk_nature.jpg"),
    ("frog_on_log",      "natural",  "natural/frog_on_log.jpg"),
    ("boy_face",         "faces",    "faces/boy_face_venezuela.jpg"),
    ("girl_sad_face",    "faces",    "faces/girl_sad_face.jpg"),
    ("wayuu_woman",      "faces",    "faces/wayuu_woman.jpg"),
    ("wood_grain",       "texture",  "texture/wood_grain.png"),
]


# ─── image loading ─────────────────────────────────────────────────────────────

def _load_and_crop(path: str, size: int = TARGET_SIZE) -> np.ndarray:
    """Load image → centre-crop → size×size → BT.601 grayscale float64 [0,1]."""
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


# ─── data structures ──────────────────────────────────────────────────────────

@dataclass
class NScanRow:
    label: str
    group: str
    n: int
    r: float
    slope: float
    ece: float


@dataclass
class NoiseRow:
    repeat: int         # 0-based window index
    seed_start: int
    seed_end: int
    slope: float
    r: float


# ─── sampling helper ──────────────────────────────────────────────────────────

def _sample_member(engine, y: np.ndarray, seed: int) -> np.ndarray:
    """One engine output reproducibly for a given seed.

    ResShiftEngine: calls engine._sample(y, seed=seed) which sets torch seed.
    OracleEngine (fast/smoke-test): uses numpy seed before engine.restore(y).
    Both are deterministic and reproducible at the same seed.
    """
    if hasattr(engine, "_sample"):
        return engine._sample(y, seed=seed)
    # OracleEngine (fast mode smoke-test only)
    np.random.seed(seed)
    return engine.restore(y)


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Oracle engine (smoke-test only; numbers meaningless)")
    args = parser.parse_args()

    op = BicubicDownsample(SCALE)

    if args.fast:
        from engine.oracle import OracleEngine
        engine = OracleEngine(op)
        print("Fast mode: OracleEngine (numbers not meaningful)")
    else:
        from engine.resshift import ResShiftEngine
        engine = ResShiftEngine(op)

    nscan_rows: list[NScanRow] = []
    noise_rows: list[NoiseRow] = []

    t_total = time.perf_counter()

    for label, group, rel_path in IMAGES:
        img_path = os.path.join(RESEARCH_DIR, rel_path)
        is_noise_image = os.path.basename(img_path) == NOISE_IMAGE
        n_max = NOISE_MAX_N if is_noise_image else MAX_N

        print(f"\n{'─'*60}")
        print(f"  {label}  ({group})  — collecting {n_max} members (seeds 0..{n_max-1})")

        x_gt = _load_and_crop(img_path)
        y    = op.forward(x_gt)

        # Collect all members at once — no GPU reload between N values
        t0 = time.perf_counter()
        raw_members = [_sample_member(engine, y, seed=s) for s in range(n_max)]
        dt_collect = time.perf_counter() - t0
        print(f"  collected {n_max} members in {dt_collect:.0f}s")

        # Rectify all members
        rects  = [rectify(m, y, op) for m in raw_members]
        x_outs = [r.x_out for r in rects]

        # N-scan: calibrate at each N using the first N members (nested)
        for N in N_SCAN_VALUES:
            if N > n_max:
                continue
            cal = calibrate(x_outs[:N], x_gt, n_bins=N_BINS, min_predicted_std=MIN_STD)
            nscan_rows.append(NScanRow(
                label=label, group=group, n=N,
                r=cal.pearson_r, slope=cal.slope, ece=cal.ece,
            ))
            print(f"  N={N:>2}: r={cal.pearson_r:+.4f}  slope={cal.slope:.4f}  ECE={cal.ece:.4f}")

        # Noise floor: 5 non-overlapping N=12 windows (boardwalk only)
        if is_noise_image:
            print(f"\n  Noise floor ({NOISE_REPEATS} × N={NOISE_N} windows, non-overlapping seeds):")
            for rep in range(NOISE_REPEATS):
                s0 = rep * NOISE_N
                s1 = s0 + NOISE_N
                if s1 > n_max:
                    print(f"  repeat {rep}: insufficient seeds (need {s1}, have {n_max}); skipping")
                    continue
                members_rep = x_outs[s0:s1]
                cal_rep = calibrate(members_rep, x_gt, n_bins=N_BINS, min_predicted_std=MIN_STD)
                noise_rows.append(NoiseRow(
                    repeat=rep, seed_start=s0, seed_end=s1 - 1,
                    slope=cal_rep.slope, r=cal_rep.pearson_r,
                ))
                print(f"    repeat {rep} (seeds {s0}–{s1-1}): "
                      f"slope={cal_rep.slope:.4f}  r={cal_rep.pearson_r:+.4f}")

    total_runtime = time.perf_counter() - t_total

    # ── Compose report ──────────────────────────────────────────────────────────
    lines: list[str] = []

    lines.append("")
    lines.append("=" * 72)
    lines.append("  SLOPE STABILITY vs ENSEMBLE SIZE — N-SCAN RESULTS")
    lines.append("  ResShift + BicubicDownsample(4), seeds nested: N=6⊂12⊂24⊂48")
    lines.append("=" * 72)
    lines.append("")

    # Table 1: N-scan
    lines.append(f"  {'Image':<20} {'Group':<9} {'N':>4}  {'r':>8}  {'slope':>8}  {'|Δslope|':>10}")
    lines.append("  " + "─" * 66)

    prev_slope: dict[str, float] = {}
    for row in nscan_rows:
        delta_str = ""
        if row.label in prev_slope and row.n > N_SCAN_VALUES[0]:
            delta = abs(row.slope - prev_slope[row.label])
            delta_str = f"{delta:>10.4f}"
        prev_slope[row.label] = row.slope
        lines.append(
            f"  {row.label:<20} {row.group:<9} {row.n:>4}  "
            f"{row.r:>+8.4f}  {row.slope:>8.4f}  {delta_str}"
        )

    lines.append("")

    # Table 2: faces convergence
    lines.append("─" * 72)
    lines.append("  FACES CONVERGENCE (within-group spread at N=6 vs N=48)")
    lines.append("─" * 72)
    lines.append("")
    face_rows_by_n: dict[int, list[NScanRow]] = {}
    for row in nscan_rows:
        if row.group == "faces":
            face_rows_by_n.setdefault(row.n, []).append(row)

    lines.append(f"  {'Image':<20} {'slope@N=6':>10}  {'slope@N=48':>10}  {'|Δ|':>8}")
    lines.append("  " + "─" * 52)
    face_n6  = {r.label: r.slope for r in face_rows_by_n.get(6, [])}
    face_n48 = {r.label: r.slope for r in face_rows_by_n.get(48, [])}
    for lbl in ["boy_face", "girl_sad_face", "wayuu_woman"]:
        s6  = face_n6.get(lbl, float("nan"))
        s48 = face_n48.get(lbl, float("nan"))
        delta = abs(s48 - s6) if not (np.isnan(s6) or np.isnan(s48)) else float("nan")
        lines.append(f"  {lbl:<20} {s6:>10.4f}  {s48:>10.4f}  {delta:>8.4f}")

    # Within-group range at N=6 and N=48
    slopes_6  = [r.slope for r in face_rows_by_n.get(6,  [])]
    slopes_48 = [r.slope for r in face_rows_by_n.get(48, [])]
    if slopes_6:
        lines.append(f"\n  Faces slope range at N=6:  "
                     f"{min(slopes_6):.4f} – {max(slopes_6):.4f}  "
                     f"(span {max(slopes_6)-min(slopes_6):.4f})")
    if slopes_48:
        lines.append(f"  Faces slope range at N=48: "
                     f"{min(slopes_48):.4f} – {max(slopes_48):.4f}  "
                     f"(span {max(slopes_48)-min(slopes_48):.4f})")
    lines.append("")

    # Table 3: noise floor
    lines.append("─" * 72)
    lines.append(f"  NOISE FLOOR — slope spread across {NOISE_REPEATS}×N={NOISE_N} "
                 f"independent seed windows  [{NOISE_IMAGE}]")
    lines.append("─" * 72)
    lines.append("")
    if noise_rows:
        lines.append(f"  {'Repeat':<8} {'Seeds':>12}  {'slope':>8}  {'r':>8}")
        lines.append("  " + "─" * 40)
        for nr in noise_rows:
            lines.append(f"  {nr.repeat:<8} {nr.seed_start:>4}–{nr.seed_end:<6}  "
                         f"{nr.slope:>8.4f}  {nr.r:>+8.4f}")
        slopes_noise = [nr.slope for nr in noise_rows]
        lines.append("")
        lines.append(f"  Mean slope    : {np.mean(slopes_noise):.4f}")
        lines.append(f"  Std  slope    : {np.std(slopes_noise):.4f}")
        lines.append(f"  Range (max−min): {max(slopes_noise)-min(slopes_noise):.4f}")
        lines.append("")
        lines.append(f"  ← This is the noise floor: how much slope wobbles from")
        lines.append(f"    sampling alone at fixed N={NOISE_N} on one image.")
    lines.append("")

    # Verdict section
    lines.append("=" * 72)
    lines.append("  VERDICT")
    lines.append("=" * 72)
    lines.append(_verdict(nscan_rows, noise_rows, total_runtime))
    lines.append("")
    lines.append(f"  Total runtime: {total_runtime:.0f}s  "
                 f"({total_runtime/60:.1f} min)")

    output = "\n".join(lines)
    print(output)

    out_path = os.path.join(os.path.dirname(__file__), "stability_nscan_results.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nResults written to {out_path}")


def _verdict(nscan_rows: list[NScanRow], noise_rows: list[NoiseRow],
             runtime: float) -> str:
    """Generate the honest (a)/(b) verdict from the data."""

    # Convergence test: |slope(48) - slope(24)| per image
    slope_by_n: dict[tuple[str, int], float] = {(r.label, r.n): r.slope for r in nscan_rows}
    images = list(dict.fromkeys(r.label for r in nscan_rows))

    convergence_deltas = []
    for lbl in images:
        s24 = slope_by_n.get((lbl, 24), float("nan"))
        s48 = slope_by_n.get((lbl, 48), float("nan"))
        if not (np.isnan(s24) or np.isnan(s48)):
            convergence_deltas.append((lbl, abs(s48 - s24)))

    # Noise floor
    noise_std  = np.std([nr.slope for nr in noise_rows]) if noise_rows else float("nan")
    noise_range = (max(nr.slope for nr in noise_rows) - min(nr.slope for nr in noise_rows)) if noise_rows else float("nan")

    # Between-group spread at N=48
    natural_slopes = [slope_by_n.get((lbl, 48), float("nan"))
                      for lbl, grp, _ in [("boardwalk_nature","natural",""),
                                          ("frog_on_log","natural","")]
                      if not np.isnan(slope_by_n.get((lbl, 48), float("nan")))]
    face_slopes    = [slope_by_n.get((lbl, 48), float("nan"))
                      for lbl in ["boy_face", "girl_sad_face", "wayuu_woman"]
                      if not np.isnan(slope_by_n.get((lbl, 48), float("nan")))]
    texture_slopes = [slope_by_n.get(("wood_grain", 48), float("nan"))]

    lines = [""]

    # Convergence
    lines.append("  Convergence (|slope(N=48) − slope(N=24)|) per image:")
    max_delta = 0.0
    for lbl, delta in convergence_deltas:
        lines.append(f"    {lbl:<20}: {delta:.4f}")
        max_delta = max(max_delta, delta)
    lines.append("")

    # Noise floor vs between-group spread
    if natural_slopes and face_slopes:
        nat_mean   = np.mean(natural_slopes)
        face_mean  = np.mean(face_slopes)
        tex_slope  = texture_slopes[0] if not np.isnan(texture_slopes[0]) else float("nan")
        bg_spread  = max(face_mean, tex_slope if not np.isnan(tex_slope) else 0) - \
                     min(nat_mean, tex_slope if not np.isnan(tex_slope) else 0)
        lines.append(f"  Between-group slope spread (natural vs faces vs texture) at N=48:")
        lines.append(f"    natural mean  : {nat_mean:.4f}")
        lines.append(f"    faces mean    : {face_mean:.4f}")
        lines.append(f"    texture (wood): {tex_slope:.4f}")
        lines.append(f"    spread        : {bg_spread:.4f}")
        lines.append(f"  Noise floor (std at N=12, fixed image): {noise_std:.4f}  "
                     f"range={noise_range:.4f}")
        lines.append("")

        # Determine verdict
        CONVERGENCE_THRESHOLD = 0.5
        converged = max_delta < CONVERGENCE_THRESHOLD

        noise_floor_estimate = noise_range   # use range as conservative estimate

        # Is between-group spread substantially above noise?
        if not np.isnan(noise_floor_estimate) and not np.isnan(bg_spread):
            signal_to_noise = bg_spread / noise_floor_estimate if noise_floor_estimate > 0 else float("inf")
        else:
            signal_to_noise = float("nan")

        lines.append(f"  Convergence threshold used: |Δslope(48,24)| < {CONVERGENCE_THRESHOLD}")
        lines.append(f"  Max delta observed: {max_delta:.4f}  → "
                     f"{'CONVERGED' if converged else 'NOT CONVERGED'}")
        lines.append(f"  Between-group/noise ratio: {signal_to_noise:.1f}×")
        lines.append("")

        # Plain-language verdict
        if converged and not np.isnan(signal_to_noise) and signal_to_noise > 3.0:
            verdict_letter = "(a)"
            verdict_text = (
                "Slope appears to converge at feasible N, and between-group differences "
                "exceed the noise floor. Domain-shift in slope is a measurable effect, "
                "not yet definitively established but worth investigating further with "
                "larger N and more images per group."
            )
        else:
            verdict_letter = "(b)"
            verdict_text = (
                "Slope does NOT reliably stabilise at N=48 and/or between-group "
                "differences do not clearly exceed the noise floor. "
                "The domain-shift-in-slope finding from the prior table is not yet "
                "established above sampling noise. Building a detector on the N=6 "
                "slope values would be premature."
            )

        lines.append(f"  VERDICT {verdict_letter}: {verdict_text}")

    lines.append("")
    lines.append("  NOTE: 'distance from training distribution' is an intuitive proxy,")
    lines.append("  not a rigorous metric. No FID or embedding-space distance computed.")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
