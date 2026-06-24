"""
eval/content_driver_analysis.py — Content-driver investigation (pre-registered Phase 2)

Pre-registration: eval/pre_registration_content_driver.md
Commits: 4f19d64 (initial), 75b2d46 (additions A + B) — both BEFORE this run.

Tests H1 (−β_spec), H2 (rho_gt), H3 (A_dom) against calibration slope.
No model runs: features computed purely from GT pixel values.

Usage:
    python eval/content_driver_analysis.py
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

# ─── Constants ────────────────────────────────────────────────────────────────
W = H = 256
ALPHA      = 0.05
ALPHA_BONF = ALPHA / 3   # 3 primary tests → 0.01667

# ─── Dataset ──────────────────────────────────────────────────────────────────
# slope: Phase 2 pre-registered values (commit dd987b1)
# A_dom, A_dom_broad: pre-stated annotations (commit 75b2d46, Addition B)
# painting: "portrait" | "landscape" | None

DATASET: list[dict] = [
    {"name": "face_red_hair",         "path": _p("large_sample/face_red_hair.jpg"),         "slope": 1.7027, "A_dom": 1, "A_dom_broad": 1, "category": "faces",   "painting": None},
    {"name": "wayuu_woman",           "path": _p("faces/wayuu_woman.jpg"),                   "slope": 1.7483, "A_dom": 1, "A_dom_broad": 1, "category": "faces",   "painting": None},
    {"name": "girl_sad_face",         "path": _p("faces/girl_sad_face.jpg"),                 "slope": 2.7834, "A_dom": 1, "A_dom_broad": 1, "category": "faces",   "painting": None},
    {"name": "paint_vermeer_pearl",   "path": _p("large_sample/paint_vermeer_pearl.jpg"),   "slope": 2.7135, "A_dom": 1, "A_dom_broad": 1, "category": "painting", "painting": "portrait"},
    {"name": "paint_vermeer_milk",    "path": _p("large_sample/paint_vermeer_milk.jpg"),    "slope": 2.3033, "A_dom": 1, "A_dom_broad": 1, "category": "painting", "painting": "portrait"},
    {"name": "boardwalk",             "path": _p("natural/boardwalk_nature.jpg"),            "slope": 0.9510, "A_dom": 0, "A_dom_broad": 0, "category": "natural", "painting": None},
    {"name": "paint_rembrandt_self",  "path": _p("large_sample/paint_rembrandt_self.jpg"),  "slope": 2.6592, "A_dom": 1, "A_dom_broad": 1, "category": "painting", "painting": "portrait"},
    {"name": "nat_landscape2",        "path": _p("new_v2/nat_landscape2.jpeg"),              "slope": 1.2053, "A_dom": 0, "A_dom_broad": 0, "category": "natural", "painting": None},
    {"name": "nature_land",           "path": _p("natural/nature_landscape.jpg"),            "slope": 1.1147, "A_dom": 0, "A_dom_broad": 0, "category": "natural", "painting": None},
    {"name": "face_algerian",         "path": _p("large_sample/face_algerian.jpg"),          "slope": 1.6042, "A_dom": 1, "A_dom_broad": 1, "category": "faces",   "painting": None},
    {"name": "boy_face",              "path": _p("faces/boy_face_venezuela.jpg"),             "slope": 3.1829, "A_dom": 1, "A_dom_broad": 1, "category": "faces",   "painting": None},
    {"name": "nat_landscape3",        "path": _p("new_v2/nat_landscape3.jpeg"),              "slope": 1.0313, "A_dom": 0, "A_dom_broad": 0, "category": "natural", "painting": None},
    {"name": "natural_caribou",       "path": _p("large_sample/natural_caribou.jpg"),       "slope": 2.1056, "A_dom": 0, "A_dom_broad": 1, "category": "natural", "painting": None},
    {"name": "paint_monet_magpie",    "path": _p("large_sample/paint_monet_magpie.jpg"),    "slope": 1.2735, "A_dom": 0, "A_dom_broad": 0, "category": "painting", "painting": "landscape"},
    {"name": "paint_monet_lilies",    "path": _p("large_sample/paint_monet_lilies.jpg"),    "slope": 1.4173, "A_dom": 0, "A_dom_broad": 0, "category": "painting", "painting": "landscape"},
    {"name": "natural_coral_reef",    "path": _p("large_sample/natural_coral_reef.jpg"),    "slope": 1.2619, "A_dom": 0, "A_dom_broad": 0, "category": "natural", "painting": None},
    {"name": "natural_fir_snow",      "path": _p("large_sample/natural_fir_snow.jpg"),      "slope": 1.5055, "A_dom": 0, "A_dom_broad": 0, "category": "natural", "painting": None},
    {"name": "frog_on_log",           "path": _p("natural/frog_on_log.jpg"),                 "slope": 1.4238, "A_dom": 0, "A_dom_broad": 1, "category": "natural", "painting": None},
    {"name": "grass_meadow",          "path": _p("texture/grass_meadow.png"),                "slope": 0.5778, "A_dom": 0, "A_dom_broad": 0, "category": "texture", "painting": None},
    {"name": "dirt_soil",             "path": _p("texture/dirt_soil.png"),                   "slope": 0.8536, "A_dom": 0, "A_dom_broad": 0, "category": "texture", "painting": None},
    {"name": "natural_snow_mountain", "path": _p("large_sample/natural_snow_mountain.jpg"), "slope": 1.1403, "A_dom": 0, "A_dom_broad": 0, "category": "natural", "painting": None},
    {"name": "texture_sand",          "path": _p("large_sample/texture_sand.jpg"),          "slope": 0.3434, "A_dom": 0, "A_dom_broad": 0, "category": "texture", "painting": None},
    {"name": "texture_cement",        "path": _p("large_sample/texture_cement.jpg"),        "slope": 0.3416, "A_dom": 0, "A_dom_broad": 0, "category": "texture", "painting": None},
    {"name": "wood_grain",            "path": _p("texture/wood_grain.png"),                  "slope": 0.4351, "A_dom": 0, "A_dom_broad": 0, "category": "texture", "painting": None},
]


# ─── Image loading ─────────────────────────────────────────────────────────────

def _load_gray_256(path: str) -> np.ndarray:
    """Load image → BT.601 grayscale float64 [0,1] at 256×256 (center-crop)."""
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


# ─── Feature computations ──────────────────────────────────────────────────────

def compute_beta_spec(gray: np.ndarray) -> float:
    """Radially averaged power spectrum log-log slope (β_spec).
    Fit range: f in [2/256, 64/256] normalized. Returns β_spec < 0 for natural images.
    Predictor used in analysis: −β_spec (positive for smooth/structured images).
    """
    N = gray.shape[0]  # 256
    F_shifted = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(F_shifted) ** 2

    freqs_shifted = np.fft.fftshift(np.fft.fftfreq(N))
    fx, fy = np.meshgrid(freqs_shifted, freqs_shifted)
    r_map = np.sqrt(fx ** 2 + fy ** 2)

    f_min, f_max = 2.0 / N, 64.0 / N
    n_bins = 30
    f_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)
    f_centers = np.sqrt(f_edges[:-1] * f_edges[1:])

    P_list, f_list = [], []
    for i in range(n_bins):
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
    return float(beta_vec[0])  # β_spec < 0 for natural images


def compute_rho_gt(gray: np.ndarray) -> float:
    """Nearest-neighbor ACF of GT image at Δ=1 px (horizontal + vertical mean).
    Same computation as rho_nn in Phase 2 but applied to GT pixel values.
    """
    mu, std = gray.mean(), gray.std()
    if std < 1e-10:
        return 0.0
    z = (gray - mu) / std
    rho_h = float(np.mean(z[:, :-1] * z[:, 1:]))
    rho_v = float(np.mean(z[:-1, :] * z[1:, :]))
    return (rho_h + rho_v) / 2.0


# ─── Statistics ───────────────────────────────────────────────────────────────

def pearson_ci_t(r: float, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Fisher-z CI with t-critical at df=n-2 (pre-registered method)."""
    if abs(r) >= 1.0 or n <= 3:
        return float("nan"), float("nan")
    z  = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    tc = scipy_stats.t.ppf(1.0 - alpha / 2.0, df=n - 2)
    return math.tanh(z - tc * se), math.tanh(z + tc * se)


def partial_corr_2(x: np.ndarray, y: np.ndarray, z1: np.ndarray, z2: np.ndarray) -> float:
    """Partial r(x, y | z1, z2) via OLS residuals."""
    Z = np.column_stack([z1, z2, np.ones(len(x))])
    def resid(v: np.ndarray) -> np.ndarray:
        b, _, _, _ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ b
    rx, ry = resid(x), resid(y)
    if rx.std() < 1e-10 or ry.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_ci_t(r: float, n: int, k: int = 2, alpha: float = ALPHA) -> tuple[float, float]:
    """Fisher-z CI for partial r with k control variables. df = n - k - 2."""
    dof = n - k - 2
    se_denom = n - k - 3
    if abs(r) >= 1.0 or dof <= 1 or se_denom <= 0:
        return float("nan"), float("nan")
    z  = math.atanh(r)
    se = 1.0 / math.sqrt(se_denom)
    tc = scipy_stats.t.ppf(1.0 - alpha / 2.0, df=dof)
    return math.tanh(z - tc * se), math.tanh(z + tc * se)


def compute_vif(X: np.ndarray) -> np.ndarray:
    """VIF_i = 1/(1-R²_i), where R²_i is from OLS regression of column i on others."""
    n, k = X.shape
    vif = np.zeros(k)
    for i in range(k):
        xi = X[:, i]
        others = np.column_stack([np.delete(X, i, axis=1), np.ones(n)])
        b, _, _, _ = np.linalg.lstsq(others, xi, rcond=None)
        xi_hat = others @ b
        ss_res = float(np.sum((xi - xi_hat) ** 2))
        ss_tot = float(np.sum((xi - xi.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        vif[i] = 1.0 / (1.0 - r2) if r2 < 0.9999 else float("inf")
    return vif


def ci_excl_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def ci_str(lo: float, hi: float) -> str:
    return f"[{lo:+.3f}, {hi:+.3f}]"


def marginal_verdict(r: float, lo_bonf: float, hi_bonf: float,
                     lo_unc: float, hi_unc: float, dir_ok: bool) -> str:
    if r < 0:
        return "REFUTED (wrong direction)"
    if ci_excl_zero(lo_bonf, hi_bonf) and dir_ok:
        return "CONFIRMED"
    if ci_excl_zero(lo_unc, hi_unc) and dir_ok:
        return "EXPLORATORY-SUPPORTED"
    return "EXPLORATORY-INDETERMINATE"


# ─── Logging ──────────────────────────────────────────────────────────────────

_log: list[str] = []


def out(s: str = "", end: str = "\n", flush: bool = False) -> None:
    print(s, end=end, flush=flush)
    _log.append(s + end)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    out("=" * 72)
    out("CONTENT-DRIVER INVESTIGATION — Phase 2 (pre-registered)")
    out("Pre-registration commits: 4f19d64, 75b2d46")
    out("=" * 72)

    n_total = len(DATASET)
    names       = [d["name"]         for d in DATASET]
    slopes      = np.array([d["slope"]      for d in DATASET], dtype=float)
    A_dom       = np.array([d["A_dom"]      for d in DATASET], dtype=float)
    A_dom_broad = np.array([d["A_dom_broad"] for d in DATASET], dtype=float)

    # ── 1. Load images and compute H1, H2 ─────────────────────────────────────
    out("\n[1] Feature computation from GT pixel values")
    out("-" * 40)

    beta_specs: list[float] = []
    rho_gts:    list[float] = []

    for i, d in enumerate(DATASET):
        out(f"  [{i+1:2d}/24] {d['name']:<28}", end="", flush=True)
        try:
            gray = _load_gray_256(d["path"])
            beta = compute_beta_spec(gray)
            rho  = compute_rho_gt(gray)
            out(f"β_spec={beta:+.4f}  −β_spec={-beta:.4f}  rho_gt={rho:.4f}")
        except Exception as e:
            out(f"ERROR: {e}")
            beta, rho = float("nan"), float("nan")
        beta_specs.append(beta)
        rho_gts.append(rho)

    beta_arr  = np.array(beta_specs, dtype=float)
    rho_arr   = np.array(rho_gts,   dtype=float)
    neg_beta  = -beta_arr  # H1 predictor

    # ── 2. Feature table ──────────────────────────────────────────────────────
    out("\n[2] Feature table (sorted by slope descending)")
    out("-" * 78)
    out(f"{'Image':<26} {'slope':>7} {'β_spec':>8} {'−β_spec':>8} {'rho_gt':>8} {'Adom':>5} {'Abroad':>7}")
    for i in np.argsort(slopes)[::-1]:
        out(f"  {names[i]:<24} {slopes[i]:7.4f} {beta_arr[i]:8.4f} {neg_beta[i]:8.4f}"
            f" {rho_arr[i]:8.4f} {int(A_dom[i]):5d} {int(A_dom_broad[i]):7d}")

    # ── 3. Marginal correlations ───────────────────────────────────────────────
    out("\n[3] Marginal correlations with calibration slope (n=24)")
    out(f"    Bonferroni α = {ALPHA_BONF:.4f} (3 primary tests)")
    out("-" * 72)

    feat_names = ["H1 (−β_spec)", "H2 (rho_gt)", "H3 (A_dom)"]
    feat_vecs  = [neg_beta, rho_arr, A_dom]
    margs: list[dict] = []

    for fname, fvec in zip(feat_names, feat_vecs):
        r = float(np.corrcoef(fvec, slopes)[0, 1])
        lo_u, hi_u = pearson_ci_t(r, n_total, alpha=ALPHA)
        lo_b, hi_b = pearson_ci_t(r, n_total, alpha=ALPHA_BONF)
        margs.append({"fname": fname, "r": r, "lo_u": lo_u, "hi_u": hi_u,
                      "lo_b": lo_b, "hi_b": hi_b})
        out(f"  {fname:<20}: r={r:+.4f}  95%CI {ci_str(lo_u,hi_u)}  98.3%CI {ci_str(lo_b,hi_b)}")

    # ── 4. Directional painting check (Addition A) ────────────────────────────
    out("\n[4] Directional check: portrait vs landscape paintings")
    out("    n=3 vs n=2 — directional only; p<0.05 is structurally impossible (Addition A)")
    out("    CANNOT confirm content-vs-style (confounded in this subset)")
    out("-" * 72)

    portrait_idx  = [i for i, d in enumerate(DATASET) if d["painting"] == "portrait"]
    landscape_idx = [i for i, d in enumerate(DATASET) if d["painting"] == "landscape"]

    out(f"  Portraits ({len(portrait_idx)}):  {[names[i] for i in portrait_idx]}")
    out(f"    slopes: {[f'{slopes[i]:.4f}' for i in portrait_idx]}  mean={slopes[portrait_idx].mean():.4f}")
    out(f"  Landscapes ({len(landscape_idx)}): {[names[i] for i in landscape_idx]}")
    out(f"    slopes: {[f'{slopes[i]:.4f}' for i in landscape_idx]}  mean={slopes[landscape_idx].mean():.4f}")

    dir_ok: dict[str, bool] = {}
    for fname, fvec in [("H1 (−β_spec)", neg_beta), ("H2 (rho_gt)", rho_arr)]:
        p_mean = float(fvec[portrait_idx].mean())
        l_mean = float(fvec[landscape_idx].mean())
        ok = p_mean > l_mean
        dir_ok[fname] = ok
        out(f"  {fname}: portrait={p_mean:.4f}  landscape={l_mean:.4f}  "
            f"→ {'PASS' if ok else 'FAIL'} (portrait>landscape?)")
    dir_ok["H3 (A_dom)"] = True
    out("  H3 (A_dom): PASS trivially (portrait A_dom=1, landscape A_dom=0 by annotation)")

    # ── 5. Verdicts ───────────────────────────────────────────────────────────
    out("\n[5] Pre-registered verdicts (marginal + directional combined)")
    out("-" * 72)
    for m in margs:
        v = marginal_verdict(m["r"], m["lo_b"], m["hi_b"], m["lo_u"], m["hi_u"],
                             dir_ok[m["fname"]])
        out(f"  {m['fname']:<20}: {v}")

    # ── 6. H3 falsification: natural_caribou ──────────────────────────────────
    out("\n[6] H3 internal falsification: natural_caribou")
    out("    Pre-registered: A_dom=0 (not human), slope=2.1056 (above natural mean)")
    out("-" * 72)

    caribou_i = next(i for i, d in enumerate(DATASET) if d["name"] == "natural_caribou")
    nat_A0_idx = [i for i, d in enumerate(DATASET)
                  if d["category"] == "natural" and d["A_dom"] == 0 and i != caribou_i]
    nat_A0_slopes = slopes[nat_A0_idx]

    out(f"  natural_caribou: slope={slopes[caribou_i]:.4f}  −β_spec={neg_beta[caribou_i]:.4f}  rho_gt={rho_arr[caribou_i]:.4f}")
    out(f"  Other A_dom=0 naturals (n={len(nat_A0_idx)}): {[names[i] for i in nat_A0_idx]}")
    out(f"    slope mean={nat_A0_slopes.mean():.4f}  range=[{nat_A0_slopes.min():.4f}, {nat_A0_slopes.max():.4f}]")
    pct = float(np.mean(nat_A0_slopes <= slopes[caribou_i]))
    out(f"  Caribou slope percentile among other A_dom=0 naturals: {pct:.0%}")
    out(f"  H3 challenge: caribou has A_dom=0 but elevated slope.")
    out(f"  H1/H2 accommodation: if caribou's −β_spec or rho_gt is comparably high,")
    out(f"  spectral/coherence properties can explain its elevation without A_dom.")

    # ── 7. A_dom_broad secondary ──────────────────────────────────────────────
    out("\n[7] Secondary analysis: A_dom_broad (human OR prominent animal)")
    out("    Not in Bonferroni correction.")
    out("-" * 72)

    r_ab = float(np.corrcoef(A_dom_broad, slopes)[0, 1])
    lo_ab, hi_ab = pearson_ci_t(r_ab, n_total)
    r_a = float(np.corrcoef(A_dom, slopes)[0, 1])

    out(f"  r(A_dom,       slope) = {r_a:+.4f}  (n_1={int(A_dom.sum())}, mean_slope_1={slopes[A_dom==1].mean():.4f}, mean_slope_0={slopes[A_dom==0].mean():.4f})")
    out(f"  r(A_dom_broad, slope) = {r_ab:+.4f}  95%CI {ci_str(lo_ab, hi_ab)}  (n_1={int(A_dom_broad.sum())}, mean_slope_1={slopes[A_dom_broad==1].mean():.4f}, mean_slope_0={slopes[A_dom_broad==0].mean():.4f})")
    if r_ab > r_a:
        out(f"  r_broad ({r_ab:+.4f}) > r_A_dom ({r_a:+.4f}): broadening to animals improves fit.")
        out(f"  Interpretation: 'prominent foreground subject' may matter more than 'human face'.")
    else:
        out(f"  r_broad ({r_ab:+.4f}) ≤ r_A_dom ({r_a:+.4f}): broadening does not improve fit.")
        out(f"  Interpretation: broadening to animals does not add explanatory power.")

    # ── 8. Collinearity check ─────────────────────────────────────────────────
    out("\n[8] Collinearity check (Addition B, pre-registered)")
    out("-" * 72)

    X_pred    = np.column_stack([neg_beta, rho_arr, A_dom])
    pred_labels = ["−β_spec", "rho_gt", "A_dom"]

    out("  Pairwise Pearson r among three predictors:")
    pairwise_rs: list[float] = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        r_p = float(np.corrcoef(X_pred[:, i], X_pred[:, j])[0, 1])
        pairwise_rs.append(r_p)
        out(f"    r({pred_labels[i]}, {pred_labels[j]}) = {r_p:+.4f}")

    vifs = compute_vif(X_pred)
    out("  VIF per predictor (>5: high, >10: severe):")
    for pn, v in zip(pred_labels, vifs):
        flag = "  ← HIGH" if 5 < v <= 10 else ("  ← SEVERE" if v > 10 else "")
        out(f"    VIF({pn}) = {v:.2f}{flag}")

    n_high_pair = sum(1 for r_p in pairwise_rs if abs(r_p) > 0.70)
    high_collin = n_high_pair >= 2
    out(f"  Summary: {n_high_pair}/3 pairwise r > 0.70  max_VIF={vifs.max():.2f}"
        f"  → {'HIGH COLLINEARITY' if high_collin else 'LOW/MODERATE COLLINEARITY'}")

    # ── 9. Partial correlations ───────────────────────────────────────────────
    out("\n[9] Partial correlations (Addition B, pre-registered)")
    out(f"    df = n - 2 - 2 = {n_total - 4}  (2 controls, n=24)")
    out("-" * 72)

    p1 = partial_corr_2(A_dom,    slopes, neg_beta, rho_arr)
    p2 = partial_corr_2(neg_beta, slopes, rho_arr,  A_dom)
    p3 = partial_corr_2(rho_arr,  slopes, neg_beta,  A_dom)

    p1_lo, p1_hi = partial_ci_t(p1, n_total)
    p2_lo, p2_hi = partial_ci_t(p2, n_total)
    p3_lo, p3_hi = partial_ci_t(p3, n_total)

    def pverdict(lo: float, hi: float) -> str:
        return "CI excl 0 → INDEPENDENT" if ci_excl_zero(lo, hi) else "CI incl 0 → not independent"

    out(f"  P1 partial r(A_dom | −β_spec, rho_gt)  = {p1:+.4f}  95%CI {ci_str(p1_lo,p1_hi)}  {pverdict(p1_lo,p1_hi)}")
    out(f"  P2 partial r(−β_spec | rho_gt, A_dom)  = {p2:+.4f}  95%CI {ci_str(p2_lo,p2_hi)}  {pverdict(p2_lo,p2_hi)}")
    out(f"  P3 partial r(rho_gt | −β_spec, A_dom)  = {p3:+.4f}  95%CI {ci_str(p3_lo,p3_hi)}  {pverdict(p3_lo,p3_hi)}")

    n_indep = sum([ci_excl_zero(p1_lo, p1_hi),
                   ci_excl_zero(p2_lo, p2_hi),
                   ci_excl_zero(p3_lo, p3_hi)])
    all_incl = n_indep == 0

    if all_incl and high_collin:
        partial_verdict = "ENTANGLED"
    elif n_indep == 1:
        partial_verdict = "ONE_INDEPENDENT"
    elif n_indep >= 2:
        partial_verdict = "MULTI_INDEPENDENT"
    else:
        partial_verdict = "MIXED"
    out(f"\n  Partial verdict: {partial_verdict} ({n_indep}/3 partials exclude 0)")

    # ── 10. Honest synthesis ──────────────────────────────────────────────────
    out("\n" + "=" * 72)
    out("[10] HONEST SYNTHESIS")
    out("=" * 72)

    out("\n  MARGINAL CORRELATIONS (n=24, Bonferroni α=0.0167):")
    for m in margs:
        v = marginal_verdict(m["r"], m["lo_b"], m["hi_b"], m["lo_u"], m["hi_u"],
                             dir_ok[m["fname"]])
        out(f"    {m['fname']:<20}: r={m['r']:+.4f}  {v}")

    out(f"\n  PAINTING CONTRAST (DIRECTIONAL ONLY — cannot confirm content-vs-style, Addition A):")
    for fname in feat_names:
        out(f"    {fname:<20}: {'PASS' if dir_ok[fname] else 'FAIL'}")

    out(f"\n  COLLINEARITY:")
    out(f"    {n_high_pair}/3 pairwise r > 0.70  max VIF = {vifs.max():.1f}")

    out(f"\n  PARTIAL CORRELATIONS (df=20, wide CIs):")
    out(f"    P1 A_dom    | H1,H2: {p1:+.4f}  {ci_str(p1_lo,p1_hi)}")
    out(f"    P2 −β_spec  | H2,A_dom: {p2:+.4f}  {ci_str(p2_lo,p2_hi)}")
    out(f"    P3 rho_gt   | H1,A_dom: {p3:+.4f}  {ci_str(p3_lo,p3_hi)}")
    out(f"    Partial verdict: {partial_verdict}")

    out(f"\n  DRIVER VERDICT:")
    if partial_verdict == "ENTANGLED":
        out("    ENTANGLED — predictors collinear; no partial excludes 0.")
        out("    What is established: H1, H2, H3 each correlate with slope (marginals).")
        out("    The face/portrait vs landscape/texture split is tracked by all three features.")
        out("    What is NOT established: which specific property — spectral smoothness,")
        out("    local coherence, or semantic human-figure presence — is the active driver.")
        out("    The three predictors co-vary so tightly (both measure the same face/non-face")
        out("    split) that partial correlations cannot separate them at n=24.")
        out("    Resolution design: add smooth non-face images (high −β_spec, A_dom=0)")
        out("    to separate H1/H2 from H3; or face images with disrupted spectral structure")
        out("    to separate H3 from H1/H2.")
    elif partial_verdict == "ONE_INDEPENDENT":
        if ci_excl_zero(p1_lo, p1_hi):
            out("    INDEPENDENT DRIVER: A_dom (semantic face/figure presence)")
            out("    survives controlling −β_spec and rho_gt.")
        elif ci_excl_zero(p2_lo, p2_hi):
            out("    INDEPENDENT DRIVER: −β_spec (global spectral slope)")
            out("    survives controlling rho_gt and A_dom.")
        elif ci_excl_zero(p3_lo, p3_hi):
            out("    INDEPENDENT DRIVER: rho_gt (local GT coherence)")
            out("    survives controlling −β_spec and A_dom.")
        out("    Caveats: n=24; partial df=20 → CIs are wide; one result at this n")
        out("    is weak evidence. No content-vs-style claim (Addition A).")
    else:
        out(f"    {partial_verdict}: see partial values. n=24 partials are underpowered;")
        out("    interpret with caution.")

    out(f"\n  CARIBOU INTERNAL TEST:")
    out(f"    natural_caribou: A_dom=0, slope={slopes[caribou_i]:.4f}")
    out(f"    −β_spec={neg_beta[caribou_i]:.4f}, rho_gt={rho_arr[caribou_i]:.4f}")
    out(f"    If its H1/H2 features are also elevated relative to other naturals,")
    out(f"    spectral/coherence properties can account for its slope without H3.")

    out(f"\n  A_dom_broad SECONDARY: r={r_ab:+.4f}  95%CI {ci_str(lo_ab,hi_ab)}")
    if r_ab > r_a:
        out(f"    Broadening to animals ({int(A_dom_broad.sum())} images) improves r vs H3 alone ({int(A_dom.sum())} images).")
        out(f"    Suggests 'prominent foreground subject' may be the relevant property, not human-specific.")
    else:
        out(f"    Broadening to animals does not improve fit → animal presence not the mechanism.")

    out(f"\n  SCOPE: ResShift + BicubicDownsample(4) specific.")
    out(f"  NO content-vs-style claim: painting contrast (n=3 vs n=2) cannot")
    out(f"  dissociate content from Vermeer/Rembrandt vs Monet stylistic conventions.")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "content_driver_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(_log)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
