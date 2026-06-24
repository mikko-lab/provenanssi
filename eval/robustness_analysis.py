"""
eval/robustness_analysis.py — Content-driver robustness checks (pre-registered)

Pre-registration: eval/pre_registration_robustness.md (commit 5850951)
BEFORE any LOO computation.

Checks:
  1. Leave-one-out on P1 (partial r(A_dom | −β_spec, rho_gt))
  2. Re-run partial structure with A_dom_broad; LOO on P1'
     + caribou consistency under A_dom_broad
  + annotation-reliability caveat recorded

Uses GT features computed in content_driver_analysis.py (same images, same loading).

Usage:
    python eval/robustness_analysis.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.stats as scipy_stats
from PIL import Image

REPO = Path(__file__).parent.parent
RES  = Path(__file__).parent / "research_sources"

def _p(rel: str) -> str:
    return str(RES / rel)

ALPHA = 0.05

# ─── Dataset (same as content_driver_analysis.py) ────────────────────────────
DATASET: list[dict] = [
    {"name": "face_red_hair",         "path": _p("large_sample/face_red_hair.jpg"),         "slope": 1.7027, "A_dom": 1, "A_dom_broad": 1},
    {"name": "wayuu_woman",           "path": _p("faces/wayuu_woman.jpg"),                   "slope": 1.7483, "A_dom": 1, "A_dom_broad": 1},
    {"name": "girl_sad_face",         "path": _p("faces/girl_sad_face.jpg"),                 "slope": 2.7834, "A_dom": 1, "A_dom_broad": 1},
    {"name": "paint_vermeer_pearl",   "path": _p("large_sample/paint_vermeer_pearl.jpg"),   "slope": 2.7135, "A_dom": 1, "A_dom_broad": 1},
    {"name": "paint_vermeer_milk",    "path": _p("large_sample/paint_vermeer_milk.jpg"),    "slope": 2.3033, "A_dom": 1, "A_dom_broad": 1},
    {"name": "boardwalk",             "path": _p("natural/boardwalk_nature.jpg"),            "slope": 0.9510, "A_dom": 0, "A_dom_broad": 0},
    {"name": "paint_rembrandt_self",  "path": _p("large_sample/paint_rembrandt_self.jpg"),  "slope": 2.6592, "A_dom": 1, "A_dom_broad": 1},
    {"name": "nat_landscape2",        "path": _p("new_v2/nat_landscape2.jpeg"),              "slope": 1.2053, "A_dom": 0, "A_dom_broad": 0},
    {"name": "nature_land",           "path": _p("natural/nature_landscape.jpg"),            "slope": 1.1147, "A_dom": 0, "A_dom_broad": 0},
    {"name": "face_algerian",         "path": _p("large_sample/face_algerian.jpg"),          "slope": 1.6042, "A_dom": 1, "A_dom_broad": 1},
    {"name": "boy_face",              "path": _p("faces/boy_face_venezuela.jpg"),             "slope": 3.1829, "A_dom": 1, "A_dom_broad": 1},
    {"name": "nat_landscape3",        "path": _p("new_v2/nat_landscape3.jpeg"),              "slope": 1.0313, "A_dom": 0, "A_dom_broad": 0},
    {"name": "natural_caribou",       "path": _p("large_sample/natural_caribou.jpg"),       "slope": 2.1056, "A_dom": 0, "A_dom_broad": 1},
    {"name": "paint_monet_magpie",    "path": _p("large_sample/paint_monet_magpie.jpg"),    "slope": 1.2735, "A_dom": 0, "A_dom_broad": 0},
    {"name": "paint_monet_lilies",    "path": _p("large_sample/paint_monet_lilies.jpg"),    "slope": 1.4173, "A_dom": 0, "A_dom_broad": 0},
    {"name": "natural_coral_reef",    "path": _p("large_sample/natural_coral_reef.jpg"),    "slope": 1.2619, "A_dom": 0, "A_dom_broad": 0},
    {"name": "natural_fir_snow",      "path": _p("large_sample/natural_fir_snow.jpg"),      "slope": 1.5055, "A_dom": 0, "A_dom_broad": 0},
    {"name": "frog_on_log",           "path": _p("natural/frog_on_log.jpg"),                 "slope": 1.4238, "A_dom": 0, "A_dom_broad": 1},
    {"name": "grass_meadow",          "path": _p("texture/grass_meadow.png"),                "slope": 0.5778, "A_dom": 0, "A_dom_broad": 0},
    {"name": "dirt_soil",             "path": _p("texture/dirt_soil.png"),                   "slope": 0.8536, "A_dom": 0, "A_dom_broad": 0},
    {"name": "natural_snow_mountain", "path": _p("large_sample/natural_snow_mountain.jpg"), "slope": 1.1403, "A_dom": 0, "A_dom_broad": 0},
    {"name": "texture_sand",          "path": _p("large_sample/texture_sand.jpg"),          "slope": 0.3434, "A_dom": 0, "A_dom_broad": 0},
    {"name": "texture_cement",        "path": _p("large_sample/texture_cement.jpg"),        "slope": 0.3416, "A_dom": 0, "A_dom_broad": 0},
    {"name": "wood_grain",            "path": _p("texture/wood_grain.png"),                  "slope": 0.4351, "A_dom": 0, "A_dom_broad": 0},
]

W = H = 256


# ─── Image loading ─────────────────────────────────────────────────────────────
def _load_gray_256(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h:
        img = img.resize((W, int(round(h * W / w))), Image.LANCZOS)
    else:
        img = img.resize((int(round(w * H / h)), H), Image.LANCZOS)
    w2, h2 = img.size
    l = (w2 - W) // 2; t = (h2 - H) // 2
    img = img.crop((l, t, l + W, t + H))
    rgb = np.array(img, dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


# ─── Feature computations (same as content_driver_analysis.py) ────────────────
def compute_beta_spec(gray: np.ndarray) -> float:
    N = gray.shape[0]
    F_shifted = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(F_shifted) ** 2
    freqs_shifted = np.fft.fftshift(np.fft.fftfreq(N))
    fx, fy = np.meshgrid(freqs_shifted, freqs_shifted)
    r_map = np.sqrt(fx ** 2 + fy ** 2)
    f_min, f_max = 2.0 / N, 64.0 / N
    f_edges = np.logspace(np.log10(f_min), np.log10(f_max), 31)
    f_centers = np.sqrt(f_edges[:-1] * f_edges[1:])
    P_list, f_list = [], []
    for i in range(30):
        mask = (r_map >= f_edges[i]) & (r_map < f_edges[i + 1])
        if mask.sum() >= 3:
            P_list.append(float(power[mask].mean()))
            f_list.append(float(f_centers[i]))
    if len(P_list) < 5:
        return float("nan")
    log_f = np.log(np.array(f_list))
    log_P = np.log(np.array(P_list))
    A_mat = np.column_stack([log_f, np.ones(len(log_f))])
    beta_vec, _, _, _ = np.linalg.lstsq(A_mat, log_P, rcond=None)
    return float(beta_vec[0])


def compute_rho_gt(gray: np.ndarray) -> float:
    mu, std = gray.mean(), gray.std()
    if std < 1e-10:
        return 0.0
    z = (gray - mu) / std
    return float((np.mean(z[:, :-1] * z[:, 1:]) + np.mean(z[:-1, :] * z[1:, :])) / 2.0)


# ─── Statistics ───────────────────────────────────────────────────────────────
def partial_corr_2(x: np.ndarray, y: np.ndarray, z1: np.ndarray, z2: np.ndarray) -> float:
    Z = np.column_stack([z1, z2, np.ones(len(x))])
    def resid(v: np.ndarray) -> np.ndarray:
        b, _, _, _ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ b
    rx, ry = resid(x), resid(y)
    if rx.std() < 1e-10 or ry.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_ci_t(r: float, n: int, k: int = 2, alpha: float = ALPHA) -> tuple[float, float]:
    dof = n - k - 2
    se_denom = n - k - 3
    if abs(r) >= 1.0 or dof <= 1 or se_denom <= 0:
        return float("nan"), float("nan")
    z  = math.atanh(r)
    se = 1.0 / math.sqrt(se_denom)
    tc = scipy_stats.t.ppf(1.0 - alpha / 2.0, df=dof)
    return math.tanh(z - tc * se), math.tanh(z + tc * se)


def pearson_ci_t(r: float, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    if abs(r) >= 1.0 or n <= 3:
        return float("nan"), float("nan")
    z  = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    tc = scipy_stats.t.ppf(1.0 - alpha / 2.0, df=n - 2)
    return math.tanh(z - tc * se), math.tanh(z + tc * se)


def hat_diagonal(Z: np.ndarray) -> np.ndarray:
    """Diagonal of hat matrix H = Z(Z'Z)^-1 Z'."""
    Q, _ = np.linalg.qr(Z)
    return np.sum(Q ** 2, axis=1)


def ci_excl_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def ci_str(lo: float, hi: float) -> str:
    return f"[{lo:+.3f}, {hi:+.3f}]"


# ─── Logging ──────────────────────────────────────────────────────────────────
_log: list[str] = []


def out(s: str = "", end: str = "\n", flush: bool = False) -> None:
    print(s, end=end, flush=flush)
    _log.append(s + end)


# ─── LOO helper ───────────────────────────────────────────────────────────────
def run_loo(semantic: np.ndarray, slopes: np.ndarray, h1: np.ndarray, h2: np.ndarray,
            names: list[str], label: str) -> dict:
    """Leave-one-out on partial r(semantic | h1, h2).
    Returns dict with all LOO results.
    """
    n = len(slopes)
    Z_full = np.column_stack([h1, h2, np.ones(n)])
    h_diag = hat_diagonal(Z_full)  # leverage of each point in the regression space

    p_loo = []
    lo_loo, hi_loo = [], []
    excl_zero = []

    for i in range(n):
        idx = [j for j in range(n) if j != i]
        s_i   = semantic[idx]
        sl_i  = slopes[idx]
        h1_i  = h1[idx]
        h2_i  = h2[idx]
        r_i   = partial_corr_2(s_i, sl_i, h1_i, h2_i)
        lo_i, hi_i = partial_ci_t(r_i, n - 1, k=2, alpha=ALPHA)
        p_loo.append(r_i)
        lo_loo.append(lo_i)
        hi_loo.append(hi_i)
        excl_zero.append(ci_excl_zero(lo_i, hi_i))

    p_arr   = np.array(p_loo)
    excl_arr = np.array(excl_zero)
    exclude_0_count = int(excl_arr.sum())
    pivotal_idx = [i for i in range(n) if not excl_arr[i]]

    out(f"\n  LOO on P({label} | H1, H2):")
    out(f"    P1 range: [{p_arr.min():+.4f}, {p_arr.max():+.4f}]")
    out(f"    Exclude-0 count: {exclude_0_count}/24")

    if pivotal_idx:
        out(f"    PIVOTAL IMAGES (drop causes CI to include 0):")
        for i in pivotal_idx:
            flag = " ← HIGH LEVERAGE" if h_diag[i] > 0.25 else ""
            out(f"      drop {names[i]:<28} P1={p_loo[i]:+.4f}  CI {ci_str(lo_loo[i], hi_loo[i])}"
                f"  slope={slopes[i]:.4f}  {label}={int(semantic[i])}"
                f"  h_ii={h_diag[i]:.3f}{flag}")
    else:
        out(f"    No pivotal images — all 24 drops exclude 0.")

    verdict = "ROBUST" if exclude_0_count == 24 else "FRAGILE"
    out(f"    LOO verdict: {verdict}")

    return {
        "p_arr": p_arr, "lo_arr": np.array(lo_loo), "hi_arr": np.array(hi_loo),
        "excl_arr": excl_arr, "exclude_0_count": exclude_0_count,
        "pivotal_idx": pivotal_idx, "h_diag": h_diag, "verdict": verdict,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    out("=" * 72)
    out("CONTENT-DRIVER ROBUSTNESS CHECKS (pre-registered)")
    out("Pre-registration: eval/pre_registration_robustness.md (commit 5850951)")
    out("=" * 72)

    # ── Load features ─────────────────────────────────────────────────────────
    out("\n[0] Loading GT images and computing features …", flush=True)
    n = len(DATASET)
    names       = [d["name"]         for d in DATASET]
    slopes      = np.array([d["slope"]       for d in DATASET], dtype=float)
    A_dom       = np.array([d["A_dom"]       for d in DATASET], dtype=float)
    A_dom_broad = np.array([d["A_dom_broad"] for d in DATASET], dtype=float)

    beta_specs, rho_gts = [], []
    for i, d in enumerate(DATASET):
        out(f"  [{i+1:2d}/24] {d['name']}", end=" ", flush=True)
        gray = _load_gray_256(d["path"])
        beta = compute_beta_spec(gray)
        rho  = compute_rho_gt(gray)
        beta_specs.append(beta)
        rho_gts.append(rho)
        out(f"β={beta:+.4f}  rho={rho:.4f}")

    neg_beta = -np.array(beta_specs, dtype=float)
    rho_arr  = np.array(rho_gts, dtype=float)

    # ── Full-sample P1 recap ──────────────────────────────────────────────────
    out("\n[RECAP] Full-sample P1 from Update 10 (n=24):")
    p1_full = partial_corr_2(A_dom, slopes, neg_beta, rho_arr)
    lo1f, hi1f = partial_ci_t(p1_full, n, k=2)
    out(f"  P1 A_dom | H1,H2: r={p1_full:+.4f}  95%CI {ci_str(lo1f,hi1f)}"
        f"  {'excl 0' if ci_excl_zero(lo1f,hi1f) else 'INCL 0'}")

    p1b_full = partial_corr_2(A_dom_broad, slopes, neg_beta, rho_arr)
    lo1bf, hi1bf = partial_ci_t(p1b_full, n, k=2)
    out(f"  P1' A_dom_broad | H1,H2: r={p1b_full:+.4f}  95%CI {ci_str(lo1bf,hi1bf)}"
        f"  {'excl 0' if ci_excl_zero(lo1bf,hi1bf) else 'INCL 0'}")

    # ══════════════════════════════════════════════════════════════════════════
    out("\n" + "=" * 72)
    out("CHECK 1 — Leave-one-out on P1 (A_dom | H1, H2)")
    out("=" * 72)
    out("  Procedure: n=23 at each drop, df=19, se=1/sqrt(18), t-critical df=19")
    out(f"  High-leverage threshold h_ii > 2×3/24 = {2*3/24:.3f}")

    loo1 = run_loo(A_dom, slopes, neg_beta, rho_arr, names, "A_dom")

    # ══════════════════════════════════════════════════════════════════════════
    out("\n" + "=" * 72)
    out("CHECK 2 — Re-run with A_dom_broad; LOO on P1'; caribou consistency")
    out("=" * 72)

    # 2a. Marginal r(A_dom_broad, slope) Bonferroni CI
    r_broad = float(np.corrcoef(A_dom_broad, slopes)[0, 1])
    lo_broad_bonf, hi_broad_bonf = pearson_ci_t(r_broad, n, alpha=0.05/3)
    lo_broad_unc,  hi_broad_unc  = pearson_ci_t(r_broad, n, alpha=0.05)
    out(f"\n  Marginal r(A_dom_broad, slope) = {r_broad:+.4f}")
    out(f"    98.3%CI (Bonferroni) {ci_str(lo_broad_bonf, hi_broad_bonf)}"
        f"  {'CONFIRMED' if ci_excl_zero(lo_broad_bonf, hi_broad_bonf) else 'not confirmed'}")
    out(f"    95%CI               {ci_str(lo_broad_unc, hi_broad_unc)}")

    # 2b. P1' full sample
    out(f"\n  P1' (A_dom_broad | H1, H2) = {p1b_full:+.4f}  95%CI {ci_str(lo1bf, hi1bf)}"
        f"  {'excl 0' if ci_excl_zero(lo1bf, hi1bf) else 'INCL 0'}")

    # 2c. LOO on P1'
    loo1b = run_loo(A_dom_broad, slopes, neg_beta, rho_arr, names, "A_dom_broad")

    # 2d. Caribou consistency under A_dom_broad
    out("\n  Caribou consistency check under A_dom_broad:")
    caribou_i = next(i for i, d in enumerate(DATASET) if d["name"] == "natural_caribou")
    broad_1_idx = [i for i in range(n) if A_dom_broad[i] == 1 and i != caribou_i]
    broad_1_slopes = slopes[broad_1_idx]
    caribou_slope = slopes[caribou_i]
    mu_broad = broad_1_slopes.mean()
    sd_broad = broad_1_slopes.std(ddof=1)
    z_caribou = (caribou_slope - mu_broad) / sd_broad if sd_broad > 0 else float("nan")
    out(f"    A_dom_broad=1 (excl caribou, n={len(broad_1_idx)}): mean={mu_broad:.4f}  std={sd_broad:.4f}")
    out(f"    natural_caribou slope: {caribou_slope:.4f}  z-score vs group: {z_caribou:+.2f}")
    consistent = abs(z_caribou) < 2.0
    out(f"    Caribou within ±2 SD of A_dom_broad=1 group: {'YES — consistent' if consistent else 'NO — outlier'}")

    # ══════════════════════════════════════════════════════════════════════════
    out("\n" + "=" * 72)
    out("ROBUSTNESS COMPARISON: A_dom vs A_dom_broad")
    out("=" * 72)

    out(f"\n  A_dom:       exclude_0_count = {loo1['exclude_0_count']}/24   ({loo1['verdict']})")
    out(f"  A_dom_broad: exclude_0_count = {loo1b['exclude_0_count']}/24  ({loo1b['verdict']})")

    c1 = loo1['exclude_0_count']
    c2 = loo1b['exclude_0_count']

    if c2 > c1 and ci_excl_zero(lo1bf, hi1bf) and consistent:
        robustness_verdict = "A_dom_BROAD_PREFERRED"
    elif c1 == 24:
        robustness_verdict = "A_dom_ROBUST"
    elif c1 == 0 and c2 == 0:
        robustness_verdict = "NEITHER_ROBUST"
    elif c2 > c1:
        robustness_verdict = "A_dom_BROAD_MORE_ROBUST"
    else:
        robustness_verdict = "A_dom_MORE_ROBUST_OR_EQUAL"
    out(f"\n  Pre-registered verdict: {robustness_verdict}")

    # ══════════════════════════════════════════════════════════════════════════
    out("\n" + "=" * 72)
    out("HONEST SYNTHESIS")
    out("=" * 72)

    out("\n  LOO RESULTS:")
    out(f"    P1  (A_dom | H1,H2):       exclude_0 = {c1}/24  {loo1['verdict']}")
    out(f"    P1' (A_dom_broad | H1,H2): exclude_0 = {c2}/24  {loo1b['verdict']}")

    out(f"\n  PIVOTAL IMAGES (if any):")
    if loo1['pivotal_idx']:
        out(f"    A_dom pivotals: {[names[i] for i in loo1['pivotal_idx']]}")
    else:
        out(f"    A_dom pivotals: none")
    if loo1b['pivotal_idx']:
        out(f"    A_dom_broad pivotals: {[names[i] for i in loo1b['pivotal_idx']]}")
    else:
        out(f"    A_dom_broad pivotals: none")

    out(f"\n  CARIBOU: {'consistent with A_dom_broad=1 group' if consistent else 'outlier within A_dom_broad=1 group'}"
        f"  (z={z_caribou:+.2f}, group mean={mu_broad:.4f}±{sd_broad:.4f})")

    if robustness_verdict == "A_dom_BROAD_PREFERRED":
        out("\n  DRIVER: 'Dominant foreground subject (animal OR human)' (A_dom_broad)")
        out("  is the more robust independent predictor AND explains the caribou case.")
        out("  Re-description: slope elevation is associated with a prominent foreground")
        out("  subject, not human faces specifically. 'Face/portrait' framing overstates")
        out("  the specificity.")
    elif robustness_verdict == "A_dom_ROBUST":
        out("\n  DRIVER: A_dom (human face/figure) is ROBUST across all 24 LOO drops.")
        out("  Independence from spectral/coherence features is not point-driven.")
    elif robustness_verdict == "NEITHER_ROBUST":
        out("\n  NEITHER A_dom NOR A_dom_broad is robust.")
        out("  Honest verdict: slope tracks a foreground-subject content split,")
        out("  but no single semantic feature is established as independent driver at n=24.")
    else:
        out(f"\n  DRIVER verdict ({robustness_verdict}): A_dom_broad shows more LOO stability.")
        out("  Interpret as: the 'prominent foreground subject' description is more")
        out("  consistent than 'human face specifically,' but neither is fully robust.")

    out(f"\n  ANNOTATION-RELIABILITY CAVEAT (standing limitation, pre-stated):")
    out("    A_dom and A_dom_broad are single-annotator binary labels (Claude Code,")
    out("    2026-06-24). No inter-rater agreement was measured. The caribou case shows")
    out("    the boundary is judgment-dependent. All content-driver results are conditional")
    out("    on this annotation scheme. This must be stated in any citation of the result.")

    out(f"\n  SCOPE: ResShift + BicubicDownsample(4) only.")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "robustness_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(_log)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
