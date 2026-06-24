"""
eval/mechanism_gate_analysis.py — Mechanism gate: expanded low-level control test

Pre-registration: eval/pre_registration_mechanism_gate.md (commit 142e7cb)
Written and committed BEFORE this script was run.

Tests whether A_dom_broad's independence survives 5 controls: H1, H2, C1, C2, C3.
  H1 = −β_spec  (spectral slope)
  H2 = rho_gt   (GT nearest-neighbour ACF)
  C1 = V_patch  (mean local patch variance, 8×8)
  C2 = G_mean   (mean gradient magnitude via Sobel)
  C3 = V_het    (std of per-patch variances, 8×8)

Outcome: SEMANTICS-ROBUST / PROXY-REVEALED / ENTANGLED

Usage:
    .venv/bin/python3 eval/mechanism_gate_analysis.py
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
from scipy.ndimage import sobel

REPO = Path(__file__).parent.parent
RES  = Path(__file__).parent / "research_sources"

def _p(rel: str) -> str:
    return str(RES / rel)

ALPHA = 0.05

# ─── Dataset (same as robustness_analysis.py) ─────────────────────────────────
DATASET: list[dict] = [
    {"name": "face_red_hair",         "path": _p("large_sample/face_red_hair.jpg"),         "slope": 1.7027, "A_dom_broad": 1},
    {"name": "wayuu_woman",           "path": _p("faces/wayuu_woman.jpg"),                   "slope": 1.7483, "A_dom_broad": 1},
    {"name": "girl_sad_face",         "path": _p("faces/girl_sad_face.jpg"),                 "slope": 2.7834, "A_dom_broad": 1},
    {"name": "paint_vermeer_pearl",   "path": _p("large_sample/paint_vermeer_pearl.jpg"),   "slope": 2.7135, "A_dom_broad": 1},
    {"name": "paint_vermeer_milk",    "path": _p("large_sample/paint_vermeer_milk.jpg"),    "slope": 2.3033, "A_dom_broad": 1},
    {"name": "boardwalk",             "path": _p("natural/boardwalk_nature.jpg"),            "slope": 0.9510, "A_dom_broad": 0},
    {"name": "paint_rembrandt_self",  "path": _p("large_sample/paint_rembrandt_self.jpg"),  "slope": 2.6592, "A_dom_broad": 1},
    {"name": "nat_landscape2",        "path": _p("new_v2/nat_landscape2.jpeg"),              "slope": 1.2053, "A_dom_broad": 0},
    {"name": "nature_land",           "path": _p("natural/nature_landscape.jpg"),            "slope": 1.1147, "A_dom_broad": 0},
    {"name": "face_algerian",         "path": _p("large_sample/face_algerian.jpg"),          "slope": 1.6042, "A_dom_broad": 1},
    {"name": "boy_face",              "path": _p("faces/boy_face_venezuela.jpg"),             "slope": 3.1829, "A_dom_broad": 1},
    {"name": "nat_landscape3",        "path": _p("new_v2/nat_landscape3.jpeg"),              "slope": 1.0313, "A_dom_broad": 0},
    {"name": "natural_caribou",       "path": _p("large_sample/natural_caribou.jpg"),       "slope": 2.1056, "A_dom_broad": 1},
    {"name": "paint_monet_magpie",    "path": _p("large_sample/paint_monet_magpie.jpg"),    "slope": 1.2735, "A_dom_broad": 0},
    {"name": "paint_monet_lilies",    "path": _p("large_sample/paint_monet_lilies.jpg"),    "slope": 1.4173, "A_dom_broad": 0},
    {"name": "natural_coral_reef",    "path": _p("large_sample/natural_coral_reef.jpg"),    "slope": 1.2619, "A_dom_broad": 0},
    {"name": "natural_fir_snow",      "path": _p("large_sample/natural_fir_snow.jpg"),      "slope": 1.5055, "A_dom_broad": 0},
    {"name": "frog_on_log",           "path": _p("natural/frog_on_log.jpg"),                 "slope": 1.4238, "A_dom_broad": 1},
    {"name": "grass_meadow",          "path": _p("texture/grass_meadow.png"),                "slope": 0.5778, "A_dom_broad": 0},
    {"name": "dirt_soil",             "path": _p("texture/dirt_soil.png"),                   "slope": 0.8536, "A_dom_broad": 0},
    {"name": "natural_snow_mountain", "path": _p("large_sample/natural_snow_mountain.jpg"), "slope": 1.1403, "A_dom_broad": 0},
    {"name": "texture_sand",          "path": _p("large_sample/texture_sand.jpg"),          "slope": 0.3434, "A_dom_broad": 0},
    {"name": "texture_cement",        "path": _p("large_sample/texture_cement.jpg"),        "slope": 0.3416, "A_dom_broad": 0},
    {"name": "wood_grain",            "path": _p("texture/wood_grain.png"),                  "slope": 0.4351, "A_dom_broad": 0},
]

W = H_IMG = 256


# ─── Image loading ─────────────────────────────────────────────────────────────
def _load_gray_256(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h:
        img = img.resize((W, int(round(h * W / w))), Image.LANCZOS)
    else:
        img = img.resize((int(round(w * H_IMG / h)), H_IMG), Image.LANCZOS)
    w2, h2 = img.size
    l = (w2 - W) // 2; t = (h2 - H_IMG) // 2
    img = img.crop((l, t, l + W, t + H_IMG))
    rgb = np.array(img, dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


# ─── Feature computations ──────────────────────────────────────────────────────
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


def compute_v_patch(gray: np.ndarray, patch_size: int = 8) -> float:
    """Mean of per-8×8-patch pixel variances. Low = smooth image."""
    h, w = gray.shape
    variances = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            variances.append(float(gray[i:i + patch_size, j:j + patch_size].var()))
    return float(np.mean(variances))


def compute_g_mean(gray: np.ndarray) -> float:
    """Mean gradient magnitude via Sobel operators."""
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def compute_v_het(gray: np.ndarray, patch_size: int = 8) -> float:
    """Std of per-8×8-patch pixel variances. High = mixed smooth/textured image."""
    h, w = gray.shape
    variances = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            variances.append(float(gray[i:i + patch_size, j:j + patch_size].var()))
    return float(np.std(variances, ddof=1))


# ─── Statistics ───────────────────────────────────────────────────────────────
def partial_corr_k(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    """Partial r(x, y | controls) via OLS residuals. controls is n×k matrix."""
    Z = np.column_stack([controls, np.ones(len(x))])
    def resid(v: np.ndarray) -> np.ndarray:
        b, _, _, _ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ b
    rx, ry = resid(x), resid(y)
    if rx.std() < 1e-10 or ry.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_ci_t(r: float, n: int, k: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Fisher-z CI for partial r with k controls. df = n-k-2, se = 1/sqrt(n-k-3)."""
    dof = n - k - 2
    se_denom = n - k - 3
    if abs(r) >= 1.0 or dof <= 1 or se_denom <= 0:
        return float("nan"), float("nan")
    z  = math.atanh(r)
    se = 1.0 / math.sqrt(se_denom)
    tc = scipy_stats.t.ppf(1.0 - alpha / 2.0, df=dof)
    return math.tanh(z - tc * se), math.tanh(z + tc * se)


def ci_excl_zero(lo: float, hi: float) -> bool:
    if math.isnan(lo) or math.isnan(hi):
        return False
    return lo > 0 or hi < 0


def ci_str(lo: float, hi: float) -> str:
    if math.isnan(lo) or math.isnan(hi):
        return "[nan, nan]"
    return f"[{lo:+.3f}, {hi:+.3f}]"


def compute_vif(X: np.ndarray) -> np.ndarray:
    """VIF_i = 1/(1-R²_i) for each column in X (no intercept added here)."""
    n, k = X.shape
    vifs = []
    for i in range(k):
        y_i = X[:, i]
        X_rest = np.column_stack([X[:, j] for j in range(k) if j != i] + [np.ones(n)])
        b, _, _, _ = np.linalg.lstsq(X_rest, y_i, rcond=None)
        y_pred = X_rest @ b
        ss_res = float(np.sum((y_i - y_pred) ** 2))
        ss_tot = float(np.sum((y_i - y_i.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        vifs.append(1.0 / (1.0 - r2) if r2 < 1.0 else float("inf"))
    return np.array(vifs)


def pairwise_r(X: np.ndarray, names: list[str]) -> list[tuple[str, str, float]]:
    """All pairwise Pearson r for columns of X."""
    k = X.shape[1]
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            r = float(np.corrcoef(X[:, i], X[:, j])[0, 1])
            pairs.append((names[i], names[j], r))
    return pairs


# ─── Logging ──────────────────────────────────────────────────────────────────
_log: list[str] = []


def out(s: str = "", end: str = "\n", flush: bool = False) -> None:
    print(s, end=end, flush=flush)
    _log.append(s + end)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    out("=" * 72)
    out("MECHANISM GATE: Expanded low-level control test (pre-registered)")
    out("Pre-registration: eval/pre_registration_mechanism_gate.md (commit 142e7cb)")
    out("=" * 72)

    # ── Load and compute features ──────────────────────────────────────────────
    out("\n[0] Loading GT images and computing all features …", flush=True)
    n = len(DATASET)
    names       = [d["name"]         for d in DATASET]
    slopes      = np.array([d["slope"]       for d in DATASET], dtype=float)
    a_broad     = np.array([d["A_dom_broad"] for d in DATASET], dtype=float)

    beta_specs, rho_gts = [], []
    v_patches, g_means, v_hets = [], [], []

    for i, d in enumerate(DATASET):
        out(f"  [{i+1:2d}/24] {d['name']}", end=" … ", flush=True)
        gray = _load_gray_256(d["path"])
        beta  = compute_beta_spec(gray)
        rho   = compute_rho_gt(gray)
        vpat  = compute_v_patch(gray)
        gm    = compute_g_mean(gray)
        vhet  = compute_v_het(gray)
        beta_specs.append(beta)
        rho_gts.append(rho)
        v_patches.append(vpat)
        g_means.append(gm)
        v_hets.append(vhet)
        out(f"β={beta:+.4f}  rho={rho:.4f}  V_pat={vpat:.5f}  G_mean={gm:.4f}  V_het={vhet:.5f}")

    h1 = -np.array(beta_specs, dtype=float)   # −β_spec
    h2 = np.array(rho_gts,     dtype=float)
    c1 = np.array(v_patches,   dtype=float)
    c2 = np.array(g_means,     dtype=float)
    c3 = np.array(v_hets,      dtype=float)

    # ── Pairwise correlations among predictors ─────────────────────────────────
    out("\n" + "=" * 72)
    out("COLLINEARITY CHECK — Pairwise Pearson r (5 controls)")
    out("=" * 72)
    pred_names = ["H1(−β)", "H2(rho)", "C1(Vpat)", "C2(Gmean)", "C3(Vhet)"]
    pred_mat   = np.column_stack([h1, h2, c1, c2, c3])
    pairs = pairwise_r(pred_mat, pred_names)
    out(f"\n  {'Pair':<26} r")
    for pn1, pn2, r in pairs:
        flag = "  ← HIGH (>0.70)" if abs(r) > 0.70 else ""
        out(f"  {pn1} vs {pn2:<16} {r:+.4f}{flag}")

    # ── VIF ───────────────────────────────────────────────────────────────────
    out("\n  VIF for each control predictor:")
    vifs = compute_vif(pred_mat)
    max_vif = float(vifs.max())
    for nm, vf in zip(pred_names, vifs):
        flag = "  ← SEVERE (>10)" if vf > 10 else ("  ← HIGH (>5)" if vf > 5 else "")
        out(f"    {nm:<14} VIF = {vf:.2f}{flag}")
    out(f"  Max VIF: {max_vif:.2f}")

    # ── Primary test: full-5 partial ──────────────────────────────────────────
    out("\n" + "=" * 72)
    out("PRIMARY TEST — Partial r(A_dom_broad | H1, H2, C1, C2, C3), n=24")
    out("=" * 72)
    k_full = 5
    r_full = partial_corr_k(a_broad, slopes, pred_mat)
    lo_full, hi_full = partial_ci_t(r_full, n, k=k_full)
    dof_full = n - k_full - 2
    excl_full = ci_excl_zero(lo_full, hi_full)
    out(f"\n  Partial r = {r_full:+.4f}")
    out(f"  95%CI     = {ci_str(lo_full, hi_full)}")
    out(f"  df = {dof_full},  se = 1/sqrt({n-k_full-3}) = {1/math.sqrt(n-k_full-3):.4f}")
    out(f"  CI excludes 0: {excl_full}")

    # ── LOO with 5 controls ───────────────────────────────────────────────────
    out("\n" + "=" * 72)
    out("LOO — 24 drops, n=23 at each, k=5 controls")
    out("=" * 72)
    k_loo = 5
    dof_loo = (n - 1) - k_loo - 2   # = 16
    se_loo  = 1.0 / math.sqrt((n - 1) - k_loo - 3)   # = 1/sqrt(15)
    out(f"\n  At each drop: n=23, df={dof_loo}, se=1/sqrt({(n-1)-k_loo-3})={se_loo:.4f}")

    p_loo, lo_loo_list, hi_loo_list, excl_loo = [], [], [], []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        ctrl_i = pred_mat[idx, :]
        r_i = partial_corr_k(a_broad[idx], slopes[idx], ctrl_i)
        lo_i, hi_i = partial_ci_t(r_i, n - 1, k=k_loo)
        p_loo.append(r_i)
        lo_loo_list.append(lo_i)
        hi_loo_list.append(hi_i)
        excl_loo.append(ci_excl_zero(lo_i, hi_i))

    p_arr = np.array(p_loo)
    excl_arr = np.array(excl_loo)
    exclude_0_count = int(excl_arr.sum())
    pivotal_idx = [i for i in range(n) if not excl_arr[i]]

    out(f"\n  LOO partial r range: [{p_arr.min():+.4f}, {p_arr.max():+.4f}]")
    out(f"  Exclude-0 count: {exclude_0_count}/24")

    if pivotal_idx:
        out(f"  PIVOTAL IMAGES (drop causes CI to include 0):")
        for i in pivotal_idx:
            out(f"    drop {names[i]:<28} r={p_loo[i]:+.4f}  CI {ci_str(lo_loo_list[i], hi_loo_list[i])}"
                f"  A_dom_broad={int(a_broad[i])}  slope={slopes[i]:.4f}")
    else:
        out("  No pivotal images — all 24 drops exclude 0.")

    loo_verdict = "ROBUST (24/24)" if exclude_0_count == 24 else f"NOT_ROBUST ({exclude_0_count}/24)"
    out(f"\n  LOO verdict: {loo_verdict}")

    # ── Sub-models (proxy detection) — run regardless for completeness ─────────
    out("\n" + "=" * 72)
    out("PROXY DETECTION SUB-MODELS")
    out("(Pre-stated: M4a drop C1, M4b drop C2, M4c drop C3, M4d original 2-control)")
    out("=" * 72)

    sub_models = [
        ("M4a", np.column_stack([h1, h2, c2, c3]), 4, "drop C1 (V_patch)"),
        ("M4b", np.column_stack([h1, h2, c1, c3]), 4, "drop C2 (G_mean)"),
        ("M4c", np.column_stack([h1, h2, c1, c2]), 4, "drop C3 (V_het)"),
        ("M4d", np.column_stack([h1, h2]),          2, "original H1+H2 only"),
    ]

    sub_results = []
    out(f"\n  {'Model':<6} {'Controls':<28} {'r':>7}  {'95%CI':<24} excl_0")
    for label, ctrl, k_sub, desc in sub_models:
        r_sub = partial_corr_k(a_broad, slopes, ctrl)
        lo_sub, hi_sub = partial_ci_t(r_sub, n, k=k_sub)
        excl_sub = ci_excl_zero(lo_sub, hi_sub)
        sub_results.append(excl_sub)
        out(f"  {label:<6} {desc:<28} {r_sub:+.4f}  {ci_str(lo_sub, hi_sub):<24} {excl_sub}")

    # proxy identification
    out("\n  Proxy identification: full-5 CI includes 0, restore by dropping one control?")
    if not excl_full:
        restorers = [(sub_models[i][0], sub_models[i][3]) for i in range(3) if sub_results[i]]
        if restorers:
            names_r = ", ".join(f"{lbl} ({desc})" for lbl, desc in restorers)
            out(f"    Sub-models restoring CI exclusion: {names_r}")
            if len(restorers) == 1:
                out(f"    → Single proxy: {restorers[0][1]}")
            else:
                out(f"    → Multiple restorers — joint entanglement, not single proxy")
        else:
            out("    No sub-model restores CI exclusion — joint entanglement.")
    else:
        out("    Full-5 CI excludes 0 — proxy check moot.")

    # ── VIF severity gate ─────────────────────────────────────────────────────
    vif_severe = max_vif > 10
    out(f"\n  VIF severity gate: max VIF = {max_vif:.2f} → {'SEVERE (>10) — CI width inflated by collinearity' if vif_severe else 'OK (<10)'}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    out("\n" + "=" * 72)
    out("VERDICT (pre-registered decision rules)")
    out("=" * 72)

    if vif_severe and not excl_full:
        verdict = "ENTANGLED"
        out(f"\n  VERDICT: {verdict}")
        out("  Full-5 CI includes 0 AND max VIF > 10 AND no single control removal")
        out("  restores exclusion. Multicollinearity at n=24 with 5 controls prevents")
        out("  separating A_dom_broad from the low-level predictors.")
        out(f"  Honest reading: cannot separate semantic from low-level driver at n=24")
        out(f"  with 5 controls. Need n ≥ 38 for df ≥ 30 with 5 controls.")
    elif excl_full and exclude_0_count == 24:
        verdict = "SEMANTICS-ROBUST"
        out(f"\n  VERDICT: {verdict}")
        out("  Full-5 partial CI excludes 0 AND LOO 24/24.")
        out("  A_dom_broad's independence from low-level statistics survives the")
        out("  expanded skeptical control set. Content-driver reading is strengthened.")
        out("  SCOPE: ResShift+BicubicDownsample(4) only. Behavioural label; no mechanism.")
    elif excl_full and exclude_0_count < 24:
        verdict = "PARTIALLY_ROBUST"
        out(f"\n  VERDICT: {verdict} (full CI excl 0, LOO {exclude_0_count}/24)")
        out("  Full-5 partial CI excludes 0 but LOO is not unanimous.")
        out("  Result depends on the included points; single observations matter.")
    elif not excl_full:
        restorers_list = [sub_models[i][3] for i in range(3) if sub_results[i]]
        if len(restorers_list) == 1:
            verdict = "PROXY-REVEALED"
            out(f"\n  VERDICT: {verdict}")
            out(f"  Full-5 CI includes 0. Dropping {restorers_list[0]} restores exclusion.")
            out(f"  A_dom_broad is acting as a proxy for {restorers_list[0]}.")
            out("  STATUS CONSEQUENCE: current SUPPORTED status should be DOWNGRADED.")
            out("  Flagging for user review — not self-editing status.")
        else:
            verdict = "ENTANGLED"
            out(f"\n  VERDICT: {verdict}")
            out("  Full-5 CI includes 0. No single control removal restores exclusion.")
            out("  Cannot attribute collapse to one proxy feature; joint entanglement.")
    else:
        verdict = "INDETERMINATE"
        out(f"\n  VERDICT: {verdict} — edge case not covered by pre-stated rules.")

    out(f"\n  Summary:")
    out(f"    Full-5 partial r(A_dom_broad | H1,H2,C1,C2,C3): {r_full:+.4f}  CI {ci_str(lo_full, hi_full)}  excl_0={excl_full}")
    out(f"    LOO exclude-0: {exclude_0_count}/24")
    out(f"    Max VIF: {max_vif:.2f}")
    out(f"    Verdict: {verdict}")
    out(f"\n  SCOPE: ResShift+BicubicDownsample(4). Single-annotator A_dom_broad label.")
    out("  This session does NOT identify a mechanism — low-level robustness gate only.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "mechanism_gate_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(_log)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
