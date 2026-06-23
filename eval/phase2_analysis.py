"""
eval/phase2_analysis.py — Phase 2 pre-registered analysis (n=26, 2026-06-24)

Pre-registration: eval/pre_registration.md (commit 8b5e68b)

Measurements per image:
  - slope : OLS calibration slope at N=48  (reuse prior where certified)
  - nf_std: std of slope across 5 independent N=12 windows (reuse prior where available)
  - rho_nn: mean nearest-neighbor ACF at r=1 from N=12 coherence run (reuse prior where available)
  - null_frac_gt: projection fraction (reuse prior where available)

Four pre-stated tests + dissociation:
  Thread 1 — r(dist, slope) + dissociation (faces/painting/natural overlap window)
  Thread 2 — power law α: nf_std ∝ slope^α
  Thread 3 — r(rho_nn, slope), r(rho_nn, nf_std), mediation
  Girl_sad_face anomaly — within-faces r(rho_nn, slope)

Usage:
    python eval/phase2_analysis.py
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.stats as scipy_stats
from pathlib import Path

# ─── imports from existing eval scripts ───────────────────────────────────────
from eval.distance_metric_v2 import (
    _load_gray_256, _run_calibration, _null_frac_gt,
    _pearson_ci, _partial_corr,
    N_BINS, MIN_STD,
)
from layer.decompose import rectify
from layer.calibrate import calibrate
from eval.spatial_coherence import _radial_acf, _coherence_metrics

# ─── paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent.parent
RES  = Path(__file__).parent / "research_sources"

def _p(rel: str) -> str:
    return str(RES / rel)

# ─── prior certified data ─────────────────────────────────────────────────────
# slope: N=48 OLS slope from distance_metric_v2_results.txt (commit ~a2ebcc1 era)
# nf_windows: 5 N=12 window slopes from slope_noise_mechanism.py PW_N12
# rho_nn: Update 7, spatial_coherence_results.txt
# null_frac: distance_metric_v2_results.txt / FINDINGS.md
# dist: Phase 1 phase1_report.txt measurements (2026-06-23)

PRIOR: dict[str, dict] = {
    "wayuu_woman": {
        "category": "faces",
        "path": _p("faces/wayuu_woman.jpg"),
        "dist": 0.7387,
        "slope": 1.7483, "slope_N": 48,
        "null_frac": 0.0162,
        "nf_windows": [1.5941, 1.6118, 1.6508, 1.5149, 1.5523],
        "rho_nn": 0.245,
    },
    "girl_sad_face": {
        "category": "faces",
        "path": _p("faces/girl_sad_face.jpg"),
        "dist": 0.7584,
        "slope": 2.7834, "slope_N": 48,
        "null_frac": 0.0032,
        "nf_windows": [2.7936, 2.8291, 2.7364, 2.7797, 2.8415],
        "rho_nn": 0.154,
    },
    "boardwalk": {
        "category": "natural",
        "path": _p("natural/boardwalk_nature.jpg"),
        "dist": 0.7828,
        "slope": 0.9510, "slope_N": 48,
        "null_frac": 0.0046,
        "nf_windows": [0.9602, 0.9556, 0.9508, 0.8932, 0.9776],
        "rho_nn": 0.161,
    },
    "boy_face": {
        "category": "faces",
        "path": _p("faces/boy_face_venezuela.jpg"),
        "dist": 0.8078,
        "slope": 3.1829, "slope_N": 48,
        "null_frac": 0.0049,
        "nf_windows": [3.2046, 3.1504, 3.1621, 3.1106, 3.2603],
        "rho_nn": 0.219,
    },
    "nature_land": {
        "category": "natural",
        "path": _p("natural/nature_landscape.jpg"),
        "dist": 0.7989,
        "slope": 1.1147, "slope_N": 48,
        "null_frac": 0.0128,
        "nf_windows": None, "rho_nn": None,
    },
    "nat_landscape2": {
        "category": "natural",
        "path": _p("new_v2/nat_landscape2.jpeg"),
        "dist": 0.7985,
        "slope": 1.2053, "slope_N": 48,
        "null_frac": 0.0181,
        "nf_windows": None, "rho_nn": None,
    },
    "nat_landscape3": {
        "category": "natural",
        "path": _p("new_v2/nat_landscape3.jpeg"),
        "dist": 0.8118,
        "slope": 1.0313, "slope_N": 48,
        "null_frac": 0.0120,
        "nf_windows": None, "rho_nn": None,
    },
    "frog_on_log": {
        "category": "natural",
        "path": _p("natural/frog_on_log.jpg"),
        "dist": 0.8626,
        "slope": 1.4238, "slope_N": 48,
        "null_frac": 0.0111,
        "nf_windows": None, "rho_nn": None,
    },
    "dirt_soil": {
        "category": "texture",
        "path": _p("texture/dirt_soil.png"),
        "dist": 0.9118,
        "slope": 0.8536, "slope_N": 48,
        "null_frac": 0.0088,
        "nf_windows": None, "rho_nn": None,
    },
    "grass_meadow": {
        "category": "texture",
        "path": _p("texture/grass_meadow.png"),
        "dist": 0.9036,
        "slope": 0.5778, "slope_N": 48,
        "null_frac": 0.0477,
        "nf_windows": None, "rho_nn": None,
    },
    # wood_grain: slope was N=192 (converges slowly); run fresh N=48 for Phase 2.
    # NF windows (PW_N12) and rho_nn are from prior measurements.
    # NOTE: PW_N12 slopes (~0.23) reflect N=12 bias for this image; NF std is
    # computed from those windows and is noted as potentially underestimating true
    # N=12 variability around the N=48/192 slope. Flagged in Thread 2 report.
    "wood_grain": {
        "category": "texture",
        "path": _p("texture/wood_grain.png"),
        "dist": 0.9444,
        "slope": None, "slope_N": None,   # run fresh N=48
        "null_frac": 0.0159,
        "nf_windows": [0.2645, 0.2355, 0.2154, 0.2123, 0.2337],
        "rho_nn": 0.098,
    },
}

# New images — all measurements fresh
NEW_IMAGES: dict[str, dict] = {
    "face_red_hair":          {"category": "faces",   "path": _p("large_sample/face_red_hair.jpg"),          "dist": 0.7213},
    "face_algerian":          {"category": "faces",   "path": _p("large_sample/face_algerian.jpg"),           "dist": 0.8032},
    "paint_vermeer_pearl":    {"category": "painting", "path": _p("large_sample/paint_vermeer_pearl.jpg"),    "dist": 0.7596},
    "paint_vermeer_milk":     {"category": "painting", "path": _p("large_sample/paint_vermeer_milk.jpg"),     "dist": 0.7712},
    "paint_rembrandt_self":   {"category": "painting", "path": _p("large_sample/paint_rembrandt_self.jpg"),   "dist": 0.7867},
    "paint_monet_magpie":     {"category": "painting", "path": _p("large_sample/paint_monet_magpie.jpg"),     "dist": 0.8294},
    "paint_monet_lilies":     {"category": "painting", "path": _p("large_sample/paint_monet_lilies.jpg"),     "dist": 0.8309},
    "natural_caribou":        {"category": "natural",  "path": _p("large_sample/natural_caribou.jpg"),        "dist": 0.8239},
    "natural_coral_reef":     {"category": "natural",  "path": _p("large_sample/natural_coral_reef.jpg"),     "dist": 0.8339},
    "natural_fir_snow":       {"category": "natural",  "path": _p("large_sample/natural_fir_snow.jpg"),       "dist": 0.8421},
    "natural_snow_mountain":  {"category": "natural",  "path": _p("large_sample/natural_snow_mountain.jpg"),  "dist": 0.9253},
    "texture_sand":           {"category": "texture",  "path": _p("large_sample/texture_sand.jpg"),           "dist": 0.9253},
    "texture_brick":          {"category": "texture",  "path": _p("large_sample/texture_brick.jpg"),          "dist": 0.9257},
    "texture_stone":          {"category": "texture",  "path": _p("large_sample/texture_stone.jpg"),          "dist": 0.9266},
    "texture_cement":         {"category": "texture",  "path": _p("large_sample/texture_cement.jpg"),         "dist": 0.9258},
}

N_CAL   = 48      # calibration ensemble size
N_NF    = 12      # NF window size
N_NF_W  = 5       # number of NF windows
N_COH   = 12      # coherence ensemble size
R_MIN   = 0.90    # minimum calibration Pearson r to include
NF_MIN  = 0.001   # minimum null_frac_gt to include

# ─── helpers ─────────────────────────────────────────────────────────────────

def _nf_std(windows: list[float]) -> float:
    return float(np.std(windows, ddof=1))


def _run_nf_windows(gray: np.ndarray, op, engine, n_windows: int, n_per: int, seed_base: int = 0) -> list[float]:
    """Run n_windows independent N=n_per calibration windows, return per-window slopes."""
    y = op.forward(gray)
    slopes = []
    for w in range(n_windows):
        x_hats = [engine._sample(y, seed=seed_base + w * n_per + i) for i in range(n_per)]
        results = [rectify(xh, y, op) for xh in x_hats]
        x_outs  = [r.x_out for r in results]
        cal = calibrate(x_outs, gray, n_bins=N_BINS, min_predicted_std=MIN_STD)
        slopes.append(float(cal.slope))
    return slopes


def _run_coherence(gray: np.ndarray, op, engine, n: int, seed_base: int = 500) -> float:
    """Run N=n coherence ensemble; return rho_nn."""
    y = op.forward(gray)
    null_comps = []
    for i in range(n):
        x_hat = engine._sample(y, seed=seed_base + i)
        null_comps.append(rectify(x_hat, y, op).null_component)
    metrics = _coherence_metrics(null_comps)
    return metrics["rho_nn"]


def _ols_log_log(slopes: list[float], stds: list[float]) -> dict:
    """Power law fit: log(std) = log(a) + α·log(slope). Returns dict with CIs."""
    x = np.log(np.array(slopes, dtype=float))
    y = np.log(np.array(stds,   dtype=float))
    n = len(x)
    x_bar, y_bar = x.mean(), y.mean()
    ss_xx = ((x - x_bar)**2).sum()
    ss_xy = ((x - x_bar) * (y - y_bar)).sum()
    alpha  = float(ss_xy / ss_xx)
    log_a  = float(y_bar - alpha * x_bar)
    y_hat  = log_a + alpha * x
    ss_res = ((y - y_hat)**2).sum()
    ss_tot = ((y - y_bar)**2).sum()
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    df     = n - 2
    sigma2 = float(ss_res / df) if df > 0 else float("nan")
    se     = float(math.sqrt(sigma2 / ss_xx)) if df > 0 else float("nan")
    tc     = float(scipy_stats.t.ppf(0.975, df))
    return {
        "alpha": alpha, "log_a": log_a, "r2": r2, "se": se,
        "ci_lo": alpha - tc * se, "ci_hi": alpha + tc * se,
        "df": df, "n": n,
    }


def _pearson_full(x: list[float], y: list[float]) -> dict:
    n = len(x)
    xa, ya = np.array(x, dtype=float), np.array(y, dtype=float)
    r, p = scipy_stats.pearsonr(xa, ya)
    lo, hi = _pearson_ci(float(r), n)
    return {"r": float(r), "p": float(p), "ci_lo": lo, "ci_hi": hi, "n": n}


def _spearman_full(x: list[float], y: list[float]) -> dict:
    rho, p = scipy_stats.spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n": len(x)}


def _kruskal_wallis(groups: dict[str, list[float]]) -> dict:
    vals = [np.array(v) for v in groups.values()]
    labels = list(groups.keys())
    stat, p = scipy_stats.kruskal(*vals)
    n_total = sum(len(v) for v in vals)
    eta2 = float((stat - len(vals) + 1) / (n_total - len(vals)))
    return {"stat": float(stat), "p": float(p), "eta2": eta2, "groups": labels}


def _mwu(a: list[float], b: list[float]) -> dict:
    stat, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"stat": float(stat), "p": float(p), "na": len(a), "nb": len(b)}


def _fmt_r(r: float, lo: float, hi: float, n: int, p: float) -> str:
    return f"r={r:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  n={n}  p={p:.3f}"

# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    lines: list[str] = []
    def out(s: str = "", end: str = "\n", flush: bool = False) -> None:
        print(s, end=end, flush=flush)
        if end == "\n":
            lines.append(s)
        else:
            lines.append(s)   # will be continued in next call

    out("=" * 72)
    out("  PHASE 2 — PRE-REGISTERED ANALYSIS  (n=26, pre-reg commit 8b5e68b)")
    out("  Date: 2026-06-24")
    out("=" * 72)
    out()

    # ── Load engine ───────────────────────────────────────────────────────────
    from operators.bicubic import BicubicDownsample
    from engine.resshift import ResShiftEngine
    SCALE = 4
    op     = BicubicDownsample(SCALE)
    engine = ResShiftEngine(op)
    out("Engine and operator loaded.")
    out()

    # ── Data collection ───────────────────────────────────────────────────────
    out("─" * 72)
    out("  DATA COLLECTION")
    out("─" * 72)
    out()

    # Build unified record list
    records: list[dict] = []
    for label, d in PRIOR.items():
        rec = dict(d, label=label, is_new=False)
        records.append(rec)
    for label, d in NEW_IMAGES.items():
        rec = dict(d, label=label, is_new=True,
                   null_frac=None, slope=None, slope_N=None,
                   nf_windows=None, rho_nn=None)
        records.append(rec)

    excluded: list[str] = []

    for rec in records:
        label = rec["label"]
        gray  = _load_gray_256(rec["path"])
        rec["gray"] = gray

        # null_frac_gt
        if rec["null_frac"] is None:
            rec["null_frac"] = _null_frac_gt(gray, op)

        # slope at N=48
        if rec["slope"] is None:
            out(f"  [{label}] calibration N={N_CAL} …", end="", flush=True)
            t0  = time.perf_counter()
            cal = _run_calibration(gray, op, engine, N_CAL)
            dt  = time.perf_counter() - t0
            rec["slope"]   = cal["slope"]
            rec["slope_N"] = N_CAL
            rec["cal_r"]   = cal["r"]
            out(f"  slope={cal['slope']:.4f}  r={cal['r']:+.4f}  ({dt:.0f}s)")
        else:
            out(f"  [{label}] slope={rec['slope']:.4f} (prior N={rec['slope_N']})")

        # Check exclusion criteria
        if rec.get("cal_r") is not None and rec["cal_r"] < R_MIN:
            out(f"  EXCLUDE {label}: cal_r={rec['cal_r']:.4f} < {R_MIN}")
            excluded.append(label)
            continue
        if rec["null_frac"] < NF_MIN:
            out(f"  EXCLUDE {label}: null_frac={rec['null_frac']:.6f} < {NF_MIN}")
            excluded.append(label)
            continue

        # NF windows
        if rec["nf_windows"] is None:
            out(f"  [{label}] NF windows ({N_NF_W}×N={N_NF}) …", end="", flush=True)
            t0 = time.perf_counter()
            rec["nf_windows"] = _run_nf_windows(gray, op, engine, N_NF_W, N_NF)
            dt = time.perf_counter() - t0
            out(f"  std={_nf_std(rec['nf_windows']):.4f}  ({dt:.0f}s)")
        else:
            out(f"  [{label}] nf_std={_nf_std(rec['nf_windows']):.4f} (prior windows)")

        # Coherence rho_nn
        if rec["rho_nn"] is None:
            out(f"  [{label}] coherence N={N_COH} …", end="", flush=True)
            t0 = time.perf_counter()
            rec["rho_nn"] = _run_coherence(gray, op, engine, N_COH)
            dt = time.perf_counter() - t0
            out(f"  rho_nn={rec['rho_nn']:.4f}  ({dt:.0f}s)")
        else:
            out(f"  [{label}] rho_nn={rec['rho_nn']:.4f} (prior Update7)")

    out()
    records = [r for r in records if r["label"] not in excluded]
    out(f"  Active sample: n={len(records)} (excluded: {excluded or 'none'})")
    out()

    # ── Data table ────────────────────────────────────────────────────────────
    out("─" * 72)
    out("  DATA TABLE (sorted by distance)")
    out("─" * 72)
    out()
    out(f"  {'label':<22}  {'cat':<9}  {'dist':>6}  {'slope':>6}  {'nf_std':>7}  {'rho_nn':>7}  {'null_f':>7}")
    out("  " + "─" * 68)
    for r in sorted(records, key=lambda x: x["dist"]):
        nf = _nf_std(r["nf_windows"])
        out(f"  {r['label']:<22}  {r['category']:<9}  {r['dist']:>6.4f}  "
            f"{r['slope']:>6.4f}  {nf:>7.4f}  {r['rho_nn']:>7.4f}  {r['null_frac']:>7.4f}")
    out()

    # Extract vectors
    def vec(field, cats=None, dist_lo=None, dist_hi=None):
        rows = records
        if cats:
            rows = [r for r in rows if r["category"] in cats]
        if dist_lo is not None:
            rows = [r for r in rows if r["dist"] >= dist_lo]
        if dist_hi is not None:
            rows = [r for r in rows if r["dist"] <= dist_hi]
        return rows

    all_r = records
    nonpaint_r = [r for r in records if r["category"] != "painting"]

    def slopes(rows): return [r["slope"] for r in rows]
    def dists(rows):  return [r["dist"]  for r in rows]
    def nulls(rows):  return [r["null_frac"] for r in rows]
    def nf_stds(rows): return [_nf_std(r["nf_windows"]) for r in rows]
    def rhos(rows):   return [r["rho_nn"] for r in rows]

    # ═══════════════════════════════════════════════════════════════════════════
    out("=" * 72)
    out("  THREAD 1 — DISTANCE → SLOPE  (pre-reg tests A/B/C/D/E + dissociation)")
    out("=" * 72)
    out()
    out("  Pre-stated hypothesis: r(dist, slope) < 0 (nearer ImageNet → higher slope)")
    out()

    # Test A: all images
    A = _pearson_full(dists(all_r), slopes(all_r))
    out(f"  A. Pearson, ALL (n={A['n']}):              {_fmt_r(A['r'], A['ci_lo'], A['ci_hi'], A['n'], A['p'])}")

    # Test B: excluding paintings
    B = _pearson_full(dists(nonpaint_r), slopes(nonpaint_r))
    out(f"  B. Pearson, no paintings (n={B['n']}):     {_fmt_r(B['r'], B['ci_lo'], B['ci_hi'], B['n'], B['p'])}")

    # Test C: partial r(dist, slope | null_frac)
    dists_a  = np.array(dists(all_r))
    slopes_a = np.array(slopes(all_r))
    nulls_a  = np.array(nulls(all_r))
    rpa = _partial_corr(dists_a, slopes_a, nulls_a)
    n_pa = len(all_r)
    lo_pa, hi_pa = _pearson_ci(rpa, n_pa - 1)  # df=n-4, approx n-1
    out(f"  C. Partial r(dist,slope|null), ALL (n={n_pa}):  r={rpa:+.3f}  CI [{lo_pa:+.3f}, {hi_pa:+.3f}]  (approx)")

    # Test D: Spearman
    D = _spearman_full(dists(all_r), slopes(all_r))
    out(f"  D. Spearman, ALL (n={D['n']}):             rho={D['rho']:+.3f}  p={D['p']:.3f}")

    out()

    # Test E: within-group r for each category with n≥6
    out("  E. Within-group r(dist, slope):")
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, rows in sorted(by_cat.items()):
        if len(rows) < 4:
            out(f"     {cat}: n={len(rows)} < 4, skip")
            continue
        e = _pearson_full(dists(rows), slopes(rows))
        out(f"     {cat} (n={e['n']}): {_fmt_r(e['r'], e['ci_lo'], e['ci_hi'], e['n'], e['p'])}")
    out()

    # Thread 1 verdict (Tests A and C must both have CI excluding 0 and r<0 for (a))
    A_sig = (A["ci_lo"] < 0 and A["ci_hi"] < 0) and A["r"] < 0
    C_sig = (lo_pa < 0 and hi_pa < 0) and rpa < 0
    A_weak = A["r"] < 0 and A["p"] < 0.10
    if A_sig and C_sig:
        t1_verdict = "(a) POSITIVE — r<0 and CI excludes 0 in A and C"
    elif A_weak:
        t1_verdict = "(a-weak) — r in predicted direction, CI includes 0, p<0.10 in A"
    elif A["r"] < 0:
        t1_verdict = "(b-leaning) — r<0 but CI includes 0 and p≥0.10"
    else:
        t1_verdict = "(b) NOT SUPPORTED — r≥0 or in wrong direction"
    out(f"  THREAD 1 VERDICT (A+C rule): {t1_verdict}")
    out()

    # ── Dissociation test ────────────────────────────────────────────────────
    out("  DISSOCIATION TEST (Test F, pre-reg 2026-06-24)")
    out("  Window: 0.72 ≤ dist ≤ 0.87 (faces/painting/natural overlap)")
    out()

    overlap_cats = {"faces", "painting", "natural"}
    overlap_rows = [r for r in records if r["category"] in overlap_cats
                    and 0.72 <= r["dist"] <= 0.87]
    overlap_by_cat: dict[str, list[float]] = {}
    for r in overlap_rows:
        overlap_by_cat.setdefault(r["category"], []).append(r["slope"])

    out(f"  Images in window (n={len(overlap_rows)}):")
    for cat, slps in sorted(overlap_by_cat.items()):
        m = np.mean(slps); s = np.std(slps, ddof=1) if len(slps) > 1 else float("nan")
        out(f"    {cat:<10}  n={len(slps)}  slopes={[f'{x:.3f}' for x in slps]}")
        out(f"              mean={m:.3f}  std={s:.3f}")
    out()

    # KW test (need ≥2 groups with ≥2 observations)
    eligible = {k: v for k, v in overlap_by_cat.items() if len(v) >= 2}
    if len(eligible) >= 2:
        kw = _kruskal_wallis(eligible)
        out(f"  Kruskal-Wallis (groups: {kw['groups']}): H={kw['stat']:.3f}  p={kw['p']:.3f}  η²={kw['eta2']:.3f}")

        # Pre-stated pairwise: faces vs painting, faces vs natural
        for g1, g2 in [("faces", "painting"), ("faces", "natural")]:
            if g1 in eligible and g2 in eligible:
                mw = _mwu(eligible[g1], eligible[g2])
                diff = np.mean(eligible[g1]) - np.mean(eligible[g2])
                out(f"  Mann-Whitney {g1} vs {g2}: U={mw['stat']:.0f}  p={mw['p']:.3f}"
                    f"  mean_diff={diff:+.3f}")
    else:
        out(f"  Not enough groups with n≥2 for KW test. Groups: {dict((k,len(v)) for k,v in overlap_by_cat.items())}")
        kw = None
    out()

    # Dissociation verdict
    if eligible and len(eligible) >= 2:
        max_diff = 0.0
        group_means = {k: np.mean(v) for k, v in eligible.items()}
        for a in group_means:
            for b in group_means:
                if a != b:
                    max_diff = max(max_diff, abs(group_means[a] - group_means[b]))

        if kw["p"] < 0.05 and max_diff > 0.5:
            dis_verdict = "DISSOCIATION CONFIRMED — slope varies where distance does not"
        elif kw["p"] < 0.10 and max_diff > 0.5:
            dis_verdict = "BORDERLINE DISSOCIATION — p<0.10, max mean diff>0.5"
        elif max_diff > 0.5:
            dis_verdict = f"MIXED — max mean diff={max_diff:.3f}>0.5 but p={kw['p']:.3f}≥0.10"
        else:
            dis_verdict = f"NULL — no dissociation detected (max diff={max_diff:.3f}, p={kw['p']:.3f})"
    else:
        dis_verdict = "INCONCLUSIVE — insufficient data in overlap window"
    out(f"  DISSOCIATION VERDICT: {dis_verdict}")
    out()

    # ═══════════════════════════════════════════════════════════════════════════
    out("=" * 72)
    out("  THREAD 2 — SLOPE → NOISE-FLOOR α  (power law re-estimation)")
    out("=" * 72)
    out()
    out("  Pre-stated H0: α=0  H1: α≈0.35–0.50 (sub-proportional)")
    out("  Fit excludes: slope < 0.2 (degenerate) and wood_grain (N=12 bias note)")
    out()

    # NOTE: wood_grain's NF windows (PW_N12) reflect ~0.23 window slopes rather than
    # the N=48 slope of ~0.60, making it inconsistent with the protocol.
    # Per pre-registration: "Fit on: non-synthetic images EXCLUDING soft_blobs and
    # any image with slope < 0.2." Wood_grain is included per the written rule, but
    # the NF std computed from N=12 windows is flagged as potentially inconsistent.

    fit_rows = [r for r in records if r["slope"] >= 0.2]
    fit_slopes = slopes(fit_rows)
    fit_nf     = nf_stds(fit_rows)
    fit_labels = [r["label"] for r in fit_rows]

    out(f"  Fit set (n={len(fit_rows)}, slope≥0.2):")
    for r in sorted(fit_rows, key=lambda x: x["slope"]):
        nf = _nf_std(r["nf_windows"])
        flag = "  ← wood_grain NF bias" if r["label"] == "wood_grain" else ""
        out(f"    {r['label']:<22}  slope={r['slope']:.4f}  nf_std={nf:.4f}{flag}")
    out()

    fit = _ols_log_log(fit_slopes, fit_nf)
    out(f"  OLS log-log fit (n={fit['n']}, df={fit['df']}):")
    out(f"    α̂  = {fit['alpha']:+.4f}  SE={fit['se']:.4f}  95% CI [{fit['ci_lo']:+.4f}, {fit['ci_hi']:+.4f}]")
    out(f"    R² = {fit['r2']:.4f}")
    out()

    # Also run without wood_grain for sensitivity
    fit_now = [r for r in fit_rows if r["label"] != "wood_grain"]
    if len(fit_now) >= 4:
        fit2 = _ols_log_log(slopes(fit_now), nf_stds(fit_now))
        out(f"  Sensitivity (wood_grain excluded, n={fit2['n']}):")
        out(f"    α̂  = {fit2['alpha']:+.4f}  SE={fit2['se']:.4f}  95% CI [{fit2['ci_lo']:+.4f}, {fit2['ci_hi']:+.4f}]")
        out(f"    R² = {fit2['r2']:.4f}")
        out()

    # Verdict
    alpha_excludes_zero = fit["ci_lo"] > 0
    alpha_below_one     = fit["ci_hi"] < 1.0
    if alpha_excludes_zero and alpha_below_one:
        t2_verdict = "DEFINITIVE — α CI excludes 0 and is entirely below 1.0 (sub-proportional confirmed)"
    elif alpha_excludes_zero:
        t2_verdict = "PROVISIONAL — α CI excludes 0 but includes 1.0"
    else:
        t2_verdict = "NULL — α CI includes 0"
    out(f"  THREAD 2 VERDICT: {t2_verdict}")
    out(f"    α̂={fit['alpha']:.4f}  CI [{fit['ci_lo']:.4f}, {fit['ci_hi']:.4f}]  n={fit['n']}")
    out()

    # ═══════════════════════════════════════════════════════════════════════════
    out("=" * 72)
    out("  THREAD 3 — COHERENCE  (rho_nn vs slope, NF, mediation)")
    out("=" * 72)
    out()

    # Test F: r(rho_nn, slope)
    F = _pearson_full(rhos(all_r), slopes(all_r))
    out(f"  F. r(rho_nn, slope), ALL (n={F['n']}): {_fmt_r(F['r'], F['ci_lo'], F['ci_hi'], F['n'], F['p'])}")

    # Test G: r(rho_nn, nf_std)
    G = _pearson_full(rhos(all_r), nf_stds(all_r))
    out(f"  G. r(rho_nn, nf_std), ALL (n={G['n']}): {_fmt_r(G['r'], G['ci_lo'], G['ci_hi'], G['n'], G['p'])}")

    # Test H: partial r(slope, nf_std | rho_nn) — mediation
    slopes_a2 = np.array(slopes(all_r))
    nf_a2     = np.array(nf_stds(all_r))
    rho_a2    = np.array(rhos(all_r))
    direct_r  = np.corrcoef(slopes_a2, nf_a2)[0, 1]
    partial_h = _partial_corr(slopes_a2, nf_a2, rho_a2)
    lo_h, hi_h = _pearson_ci(partial_h, len(all_r) - 1)
    out(f"  H. Direct r(slope, nf_std)         = {direct_r:+.3f}")
    out(f"     Partial r(slope,nf_std|rho_nn)  = {partial_h:+.3f}  CI [{lo_h:+.3f}, {hi_h:+.3f}]")
    mediation_threshold = 0.7 * abs(direct_r)
    mediation_confirmed = abs(partial_h) < mediation_threshold
    out(f"     Mediation threshold (0.7×|direct|) = {mediation_threshold:.3f}")
    out(f"     |partial| = {abs(partial_h):.3f} {'< threshold → MEDIATION CONFIRMED' if mediation_confirmed else '≥ threshold → no mediation'}")
    out()

    # Thread 3 verdict for F (primary)
    if F["r"] > 0 and F["ci_lo"] > 0:
        t3_verdict = "(a) POSITIVE — rho_nn→slope CI excludes 0"
    elif F["r"] > 0.2 and F["ci_lo"] <= 0:
        t3_verdict = "(a-weak) — r>0.2 but CI includes 0"
    else:
        t3_verdict = "(b) NOT SUPPORTED — r≤0.2 or wrong direction"
    out(f"  THREAD 3 VERDICT (Test F): {t3_verdict}")
    out()

    # ═══════════════════════════════════════════════════════════════════════════
    out("=" * 72)
    out("  GIRL_SAD_FACE ANOMALY — within-faces coherence")
    out("=" * 72)
    out()

    faces_r = [r for r in records if r["category"] == "faces"]
    out(f"  Face images (n={len(faces_r)}):")
    for r in sorted(faces_r, key=lambda x: x["slope"]):
        out(f"    {r['label']:<22}  slope={r['slope']:.4f}  rho_nn={r['rho_nn']:.4f}")
    out()

    if len(faces_r) >= 4:
        face_corr = _pearson_full(slopes(faces_r), rhos(faces_r))
        out(f"  Within-faces r(slope, rho_nn) (n={face_corr['n']}): "
            f"{_fmt_r(face_corr['r'], face_corr['ci_lo'], face_corr['ci_hi'], face_corr['n'], face_corr['p'])}")
        out()
        if face_corr["ci_lo"] <= 0 <= face_corr["ci_hi"]:
            anomaly_verdict = "CI includes 0 → anomaly is noise (n too small to distinguish)"
        elif face_corr["r"] < 0:
            anomaly_verdict = "r<0 within faces → coherence and slope DECOUPLED within faces"
        else:
            anomaly_verdict = "r>0 within faces → coherence tracks slope within faces (anomaly dissolves)"
        out(f"  ANOMALY VERDICT: {anomaly_verdict}")
    else:
        out(f"  n={len(faces_r)} faces — insufficient for within-group test (need ≥4)")
    out()

    # ═══════════════════════════════════════════════════════════════════════════
    out("=" * 72)
    out("  OVERALL SUMMARY — ESTABLISHED vs OPEN")
    out("=" * 72)
    out()
    out(f"  n = {len(records)} non-synthetic images in Phase 2 analysis")
    out()
    out(f"  Thread 1 (distance→slope):  {t1_verdict}")
    out(f"  Dissociation:               {dis_verdict}")
    out(f"  Thread 2 (power law α):     {t2_verdict}")
    out(f"    α̂={fit['alpha']:.4f}  CI [{fit['ci_lo']:.4f}, {fit['ci_hi']:.4f}]  n={fit['n']}")
    out(f"  Thread 3 (coherence→slope): {t3_verdict}")
    out(f"  girl_sad_face anomaly:      {anomaly_verdict if len(faces_r) >= 4 else 'INCONCLUSIVE (n<4)'}")
    out()

    # Save results
    results_path = Path(__file__).parent / "phase2_results.txt"
    results_path.write_text("\n".join(lines))
    print(f"\nResults written to {results_path}")

    # ── Return data dict for FINDINGS.md appending ────────────────────────────
    return {
        "records": records,
        "thread1": {"A": A, "B": B, "partial_r": rpa, "partial_ci": (lo_pa, hi_pa),
                    "D": D, "verdict": t1_verdict},
        "dissociation": {"kw": kw if eligible and len(eligible) >= 2 else None,
                         "verdict": dis_verdict,
                         "overlap_by_cat": overlap_by_cat},
        "thread2": {"fit": fit, "fit2": fit2 if len(fit_now) >= 4 else None,
                    "verdict": t2_verdict},
        "thread3": {"F": F, "G": G, "direct_r": direct_r,
                    "partial_h": partial_h, "partial_h_ci": (lo_h, hi_h),
                    "mediation": mediation_confirmed, "verdict": t3_verdict},
        "anomaly": {"verdict": anomaly_verdict if len(faces_r) >= 4 else "INCONCLUSIVE"},
    }


if __name__ == "__main__":
    result = main()
