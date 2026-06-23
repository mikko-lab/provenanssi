"""
eval/spatial_coherence.py — Direct measurement of null-space hallucination coherence.

PURPOSE
-------
Update 6 inferred that high-slope images have lower effective sample size n_eff
because ResShift generates more spatially correlated null-space hallucinations.
This script tests that inference DIRECTLY by measuring the spatial correlation of
the seed-to-seed null-space deviations.

METRIC — INDEPENDENCE ARGUMENT
-------------------------------
For each ensemble of N members:
  null_dev_i[p] = (I−A⁺A)x̂_i[p] − mean_seeds( (I−A⁺A)x̂_j[p] )
  null_dev_norm_i[p] = null_dev_i[p] / std_seeds( (I−A⁺A)x̂_j[p] )
                       (normalize per pixel so all pixels contribute equally)

Primary coherence metrics (chosen because ACF decays to sub-1/e within 1px for all images):

  rho_nn = mean over members of (spatial ACF of null_dev_norm_i evaluated at r=1)
           = average nearest-neighbor spatial correlation of seed-to-seed deviations

  Gamma  = mean over members of sum_{r=1}^{max_r} ACF_i(r) * count(r)/n
           = spatial coherence integral = key quantity for n_eff:
             n_eff ≈ n / (1 + 2 * Gamma)

MATHEMATICAL NOTE:
  The average over seeds of the within-seed spatial ACF of null_dev_norm_i equals
  the average across-seed correlation between pixel pairs at the same lag:
    (1/N) Σ_i ACF_spatial(null_dev_norm_i, Δ) = E_p[Cov_seeds(null_i[p],null_i[p+Δ])
                                                     / (std[p]*std[p+Δ])]
  This equivalence (by interchanging expectations) means our within-member ACF
  is measuring the across-seed spatial correlation structure — the direct mechanism
  for n_eff reduction.

WHY INDEPENDENT OF THE NOISE-FLOOR CALCULATION:
  - Noise floor: 5 independent N-sample windows → std of 5 scalar OLS slopes.
    Uses only per-bin SCALAR summaries; uses NO spatial pixel positions.
  - Coherence: 1 N-sample ensemble → spatial ACF of 2D deviation fields.
    Uses ONLY spatial structure; uses NO binning, no x_gt, no slope estimation.
  - rho_nn can be computed from N=1 member. Noise floor requires N×5 members.
    Structurally incomparable statistics from a shared underlying ensemble.

MECHANISM TESTED:
  slope → spatial coherence (rho_nn / Gamma) → lower n_eff → higher noise floor
  (a) coherence correlates with slope: mechanism DIRECTLY SUPPORTED
  (b) coherence does NOT correlate with slope: Update 6's causal claim needs softening

Usage:
    python eval/spatial_coherence.py

Output: printed report + eval/spatial_coherence_results.txt
Runtime: 7 images × N=12 passes ≈ 90s on MPS (uses cached ensemble from first run).
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

# ─── configuration ────────────────────────────────────────────────────────────
SCALE        = 4
N_MEMBERS    = 12
TARGET_SIZE  = 256
RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "research_sources")
OUT_PATH     = os.path.join(os.path.dirname(__file__), "spatial_coherence_results.txt")

# ─── images with known noise-floor data ───────────────────────────────────────
# slope and noise_floor_std from slope_noise_mechanism_results.txt (N=12 windows, K=5)
MAIN_IMAGES = [
    # (label, rel_path, null_frac, slope_mean_N12, noise_floor_std_N12)
    ("wood_grain",    "texture/wood_grain.png",           0.0159, 0.2323, 0.0186),
    ("boardwalk",     "natural/boardwalk_nature.jpg",      0.0046, 0.9475, 0.0286),
    ("wayuu_woman",   "faces/wayuu_woman.jpg",             0.0162, 1.5848, 0.0471),
    ("girl_sad_face", "faces/girl_sad_face.jpg",           0.0032, 2.7961, 0.0374),
    ("boy_face",      "faces/boy_face_venezuela.jpg",      0.0049, 3.1776, 0.0511),
    ("soft_blobs",    "synthetic/soft_blobs.png",          0.0014, 4.3725, 0.3391),
]

SANITY_IMAGES = [
    # White Gaussian noise → expected rho_nn ≈ 0, Gamma ≈ 0
    ("noise_gauss50", "synthetic/noise_gauss50.png", 0.3194, None, None),
]


# ─── image loading ────────────────────────────────────────────────────────────

def _load_crop(path: str, size: int = TARGET_SIZE) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h:
        new_w, new_h = size, int(round(h * size / w))
    else:
        new_w, new_h = int(round(w * size / h)), size
    img  = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    img  = img.crop((left, top, left + size, top + size))
    rgb  = np.array(img, dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _run_ensemble(engine, op, img_path: str, n: int):
    x_gt = _load_crop(img_path)
    y    = op.forward(x_gt)
    null_components = []
    for seed in range(n):
        x_hat = engine._sample(y, seed=seed)
        null_components.append(rectify(x_hat, y, op).null_component)
    return x_gt, null_components


# ─── ACF helpers ─────────────────────────────────────────────────────────────

def _radial_acf(field: np.ndarray, max_r: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the normalized radial ACF of a 2D field up to lag max_r.

    Returns (radial, counts_pos) where:
      radial[r]     = mean ACF at integer lag r (positive-quadrant average)
      counts_pos[r] = number of positive-quadrant pixel positions at lag r

    ACF is normalized so ACF[0] = 1.
    counts_pos[r] relates to full-quadrant N_full(r) via:
      N_full(r) = 4*counts_pos[r] - 4   (for r≥1; all 4 quadrants minus axis-double-counting)
    """
    H, W = field.shape
    f    = field - field.mean()

    # Linear (zero-padded) autocorrelation via FFT
    fft   = np.fft.fft2(f, s=(2 * H, 2 * W))
    power = np.abs(fft) ** 2
    acf   = np.fft.ifft2(power).real   # shape (2H, 2W), peak at [0,0]

    if acf[0, 0] == 0.0:
        return np.zeros(max_r), np.zeros(max_r, dtype=int)
    acf /= acf[0, 0]

    # Crop to positive lags only: (Δy,Δx) ∈ [0,H)×[0,W)
    acf_crop = acf[:H, :W]

    dy      = np.arange(H)[:, np.newaxis]
    dx      = np.arange(W)[np.newaxis, :]
    r_int   = np.sqrt(dy**2 + dx**2).astype(int)
    mask    = r_int < max_r

    radial     = np.zeros(max_r, dtype=float)
    counts_pos = np.zeros(max_r, dtype=int)
    np.add.at(radial,     r_int[mask], acf_crop[mask])
    np.add.at(counts_pos, r_int[mask], 1)
    radial /= np.maximum(counts_pos, 1)
    return radial, counts_pos


def _coherence_metrics(null_components: list[np.ndarray]) -> dict:
    """
    Compute spatial coherence metrics from a null-space component ensemble.

    Returns dict with:
      rho_nn   : mean nearest-neighbor ACF (r=1) across members — primary metric
      rho_nn_std
      gamma_nn : Σ_{r=1}^{5} rho(r) * N_full(r), averaged over members
                 where N_full(r) = 4*counts_pos[r]-4 (full 4-quadrant count)
                 n_eff ≈ n / (1 + gamma_nn) using first 5 rings only
                 (Gamma is stable when truncated before oscillating terms dominate)
      rho_r    : mean radial ACF profile (r=0..max_r-1)
      L_1e     : 1/e correlation length (usually 1px when ACF decays quickly)
    """
    N    = len(null_components)
    H, W = null_components[0].shape
    stack = np.stack(null_components, axis=0)

    mean_null = stack.mean(axis=0)
    std_null  = stack.std(axis=0, ddof=0)
    std_null  = np.where(std_null < 1e-12, 1e-12, std_null)

    max_r     = min(H, W) // 2
    rho_r_sum = np.zeros(max_r, dtype=float)
    rho_nn_list  = []
    gamma_nn_list = []
    counts_pos_arr = None

    for i in range(N):
        dev    = stack[i] - mean_null
        norm   = dev / std_null
        radial, counts_pos = _radial_acf(norm, max_r)
        rho_r_sum += radial

        if counts_pos_arr is None:
            counts_pos_arr = counts_pos

        rho_nn_list.append(float(radial[1]) if max_r > 1 else float("nan"))

        # Gamma using full-quadrant weights N_full(r) = 4*counts_pos[r] - 4
        # Truncate to first 5 rings to avoid oscillation dominating
        N_full = 4 * counts_pos.astype(float) - 4
        N_full[0] = 0   # exclude zero lag
        truncate = min(6, max_r)
        gamma_i = float(np.sum(radial[1:truncate] * N_full[1:truncate]))
        gamma_nn_list.append(gamma_i)

    rho_r_mean = rho_r_sum / N

    # 1/e correlation length
    threshold = 1.0 / np.e
    L_1e = float(max_r)
    for r in range(max_r):
        if rho_r_mean[r] < threshold:
            L_1e = float(r)
            break

    return {
        "rho_nn":            float(np.mean(rho_nn_list)),
        "rho_nn_std":        float(np.std(rho_nn_list)),
        "gamma_nn":          float(np.mean(gamma_nn_list)),
        "gamma_nn_std":      float(np.std(gamma_nn_list)),
        "L_1e":              L_1e,
        "rho_r":             rho_r_mean,
        "rho_nn_per_member": rho_nn_list,
    }


# ─── statistics ───────────────────────────────────────────────────────────────

def _pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    xc = np.array(x, dtype=float) - np.mean(x)
    yc = np.array(y, dtype=float) - np.mean(y)
    denom = np.sqrt((xc**2).sum() * (yc**2).sum())
    if denom < 1e-15:
        return float("nan"), float("nan"), float("nan")
    r = float((xc * yc).sum() / denom)
    r_clamped = np.clip(r, -0.9999, 0.9999)
    z    = 0.5 * np.log((1 + r_clamped) / (1 - r_clamped))
    se_z = 1.0 / np.sqrt(n - 3) if n > 3 else float("inf")
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
               6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    tc  = t_table.get(n - 2, 2.0)
    def ztanh(zz): return (np.exp(2*zz) - 1) / (np.exp(2*zz) + 1)
    return r, float(ztanh(z - tc * se_z)), float(ztanh(z + tc * se_z))


def _partial_r(x, y, z):
    n = len(x)
    if n < 4:
        return float("nan"), float("nan"), float("nan")
    x_arr, y_arr, z_arr = (np.array(v, dtype=float) for v in (x, y, z))
    def _resid(a, b):
        bc = b - b.mean()
        ac = a - a.mean()
        denom = (bc**2).sum()
        return ac if denom < 1e-15 else ac - (bc * ac).sum() / denom * bc
    return _pearson(_resid(x_arr, z_arr), _resid(y_arr, z_arr))


def _report_r(out, tag, x, y, x_label, y_label):
    r, lo, hi = _pearson(x, y)
    n = len(x)
    flag = "  [0 in CI]" if (lo <= 0 <= hi) else "  [CI excl. 0]"
    if np.isnan(r):
        out(f"  {tag} r({x_label}, {y_label}) = UNDEFINED (zero variance)  n={n}")
    else:
        out(f"  {tag} r({x_label}, {y_label}) = {r:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  n={n}{flag}")
    return r, lo, hi


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    op = BicubicDownsample(SCALE)
    out("Loading ResShift engine …")
    from engine.resshift import ResShiftEngine
    engine = ResShiftEngine(op)
    out()

    t_total = time.perf_counter()

    out("=" * 72)
    out("  SPATIAL COHERENCE OF NULL-SPACE HALLUCINATION — DIRECT MEASUREMENT")
    out("  ResShift + BicubicDownsample(4), N=12 members per image")
    out("=" * 72)
    out()
    out("  Primary metrics:")
    out("    rho_nn = mean nearest-neighbor (r=1) spatial ACF of normalized")
    out("             seed-to-seed null-space deviations.  Directly measures")
    out("             whether nearby pixels co-vary across seeds.")
    out("    Gamma  = spatial coherence integral Σ_r ACF(r)*count(r)/n.")
    out("             n_eff ≈ n / (1 + 2*Gamma) — key quantity for noise floor.")
    out("  Note: L (1/e length) is also shown but resolves to 1px for all images")
    out("  when the ACF decays very quickly; rho_nn and Gamma are the informative")
    out("  metrics in that regime.")
    out()

    # ──────────────────────────────────────────────────────────────────────
    # SANITY CHECK
    # ──────────────────────────────────────────────────────────────────────
    out("─" * 72)
    out("  SANITY CHECK — noise_gauss50")
    out("─" * 72)
    out()
    out("  Gaussian white-noise image: null-space deviations should be spatially")
    out("  uncorrelated → rho_nn ≈ 0, Gamma ≈ 0.")
    out()

    sanity_rho_nn = {}
    sanity_gamma  = {}
    for label, rel_path, null_frac, _, _ in SANITY_IMAGES:
        t0 = time.perf_counter()
        _, null_comps = _run_ensemble(engine, op, os.path.join(RESEARCH_DIR, rel_path), N_MEMBERS)
        dt = time.perf_counter() - t0
        m = _coherence_metrics(null_comps)
        out(f"  {label} ({dt:.0f}s):")
        out(f"    null_frac = {null_frac:.4f}")
        out(f"    rho_nn  = {m['rho_nn']:.4f} ± {m['rho_nn_std']:.4f}")
        out(f"    gamma_nn= {m['gamma_nn']:.3f} ± {m['gamma_nn_std']:.3f}  (n_eff/n ≈ {1/(1+max(m['gamma_nn'],0)+1e-9):.3f})")
        out(f"    L_1e    = {m['L_1e']:.1f} px")
        rho_r_str = [f"{m['rho_r'][r]:.4f}" for r in range(min(6, len(m["rho_r"])))]
        out(f"    rho_r[0..5]: {rho_r_str}")
        sanity_rho_nn[label] = m["rho_nn"]
        sanity_gamma[label]  = m["gamma_nn"]
    out()

    sanity_noise_rho = sanity_rho_nn.get("noise_gauss50", 0.0)
    out(f"  Sanity reference: noise_gauss50 rho_nn = {sanity_noise_rho:.4f}")
    out()

    # ──────────────────────────────────────────────────────────────────────
    # MAIN MEASUREMENTS
    # ──────────────────────────────────────────────────────────────────────
    out("─" * 72)
    out("  MAIN MEASUREMENTS")
    out("─" * 72)
    out()

    rows = []  # (label, null_frac, slope, nf_std, rho_nn, gamma)
    for label, rel_path, null_frac, slope, nf_std in MAIN_IMAGES:
        t0 = time.perf_counter()
        _, null_comps = _run_ensemble(engine, op, os.path.join(RESEARCH_DIR, rel_path), N_MEMBERS)
        dt = time.perf_counter() - t0
        m = _coherence_metrics(null_comps)
        out(f"  {label} (slope={slope:.4f}, {dt:.0f}s):")
        out(f"    rho_nn  = {m['rho_nn']:.4f} ± {m['rho_nn_std']:.4f}")
        out(f"    gamma_nn= {m['gamma_nn']:.3f} ± {m['gamma_nn_std']:.3f}  (n_eff/n ≈ {1/(1+max(m['gamma_nn'],0)+1e-9):.3f})")
        out(f"    L_1e    = {m['L_1e']:.1f} px")
        rho_r_vals = [f"{m['rho_r'][r]:.4f}" for r in range(min(6, len(m["rho_r"])))]
        out(f"    rho_r[0..5]: {rho_r_vals}")
        out()
        rows.append((label, null_frac, slope, nf_std, m["rho_nn"], m["gamma_nn"]))

    # ──────────────────────────────────────────────────────────────────────
    # DATA TABLE
    # ──────────────────────────────────────────────────────────────────────
    out("─" * 72)
    out("  DATA TABLE (sorted by slope)")
    out("─" * 72)
    out()
    sorted_rows = sorted(rows, key=lambda r: r[2])
    out(f"  {'Image':<16}  {'null_frac':>9}  {'slope':>7}  {'nf_std':>7}  {'rho_nn':>7}  {'gamma_nn':>8}  {'n_eff/n':>7}  note")
    out("  " + "─" * 80)
    for label, nf, sl, nfs, rnn, gam in sorted_rows:
        note = "[DEGENERATE]" if label == "soft_blobs" else ""
        neff_ratio = 1.0 / (1.0 + max(gam, 0.0) + 1e-9)
        out(f"  {label:<16}  {nf:>9.4f}  {sl:>7.4f}  {nfs:>7.4f}  {rnn:>7.4f}  {gam:>8.2f}  {neff_ratio:>7.3f}  {note}")
    out()

    # ──────────────────────────────────────────────────────────────────────
    # CORRELATION ANALYSIS (two metrics: rho_nn and Gamma)
    # ──────────────────────────────────────────────────────────────────────
    out("─" * 72)
    out("  CORRELATION ANALYSIS")
    out("─" * 72)
    out()
    out("  Two metrics: rho_nn (nearest-neighbour ACF at r=1, PRIMARY) and gamma_nn")
    out("  (first-5-ring coherence integral: Σ_{r=1}^{5} ρ(r)*N_full(r)).")
    out("  Mechanism prediction: POSITIVE correlation with slope and with noise_floor_std.")
    out()

    all_slopes   = [r[2] for r in rows]
    all_nf_std   = [r[3] for r in rows]
    all_rnn      = [r[4] for r in rows]
    all_gamma_nn = [r[5] for r in rows]

    excl      = [r for r in rows if r[0] != "soft_blobs"]
    e_slopes  = [r[2] for r in excl]
    e_nf_std  = [r[3] for r in excl]
    e_rnn     = [r[4] for r in excl]
    e_gamma   = [r[5] for r in excl]

    out("  ── rho_nn (primary) ────────────────────────────────────────────────")
    out()
    out("  A1. r(rho_nn, slope)")
    r_rnn_s_full, *_ = _report_r(out, "    Full (n=6)", all_rnn, all_slopes, "rho_nn", "slope")
    r_rnn_s_excl, lo_rnn_s, hi_rnn_s = _report_r(out, "    Excl. soft_blobs (n=5)", e_rnn, e_slopes, "rho_nn", "slope")
    out()
    out("  B1. r(rho_nn, noise_floor_std)")
    r_rnn_n_full, *_ = _report_r(out, "    Full (n=6)", all_rnn, all_nf_std, "rho_nn", "nf_std")
    r_rnn_n_excl, *_ = _report_r(out, "    Excl. soft_blobs (n=5)", e_rnn, e_nf_std, "rho_nn", "nf_std")
    out()
    out("  ── gamma_nn (secondary) ────────────────────────────────────────────")
    out()
    out("  A2. r(gamma_nn, slope)")
    r_gam_s_full, *_ = _report_r(out, "    Full (n=6)", all_gamma_nn, all_slopes, "gamma_nn", "slope")
    r_gam_s_excl, lo_gam_s, hi_gam_s = _report_r(out, "    Excl. soft_blobs (n=5)", e_gamma, e_slopes, "gamma_nn", "slope")
    out()
    out("  B2. r(gamma_nn, noise_floor_std)")
    r_gam_n_full, *_ = _report_r(out, "    Full (n=6)", all_gamma_nn, all_nf_std, "gamma_nn", "nf_std")
    r_gam_n_excl, *_ = _report_r(out, "    Excl. soft_blobs (n=5)", e_gamma, e_nf_std, "gamma_nn", "nf_std")
    out()
    out("  ── Direct slope↔noise (for mediation reference) ───────────────────")
    out()
    r_sn_full, *_ = _report_r(out, "    Full (n=6)", all_slopes, all_nf_std, "slope", "nf_std")
    r_sn_excl, *_ = _report_r(out, "    Excl. soft_blobs (n=5)", e_slopes, e_nf_std, "slope", "nf_std")
    out()
    out("  ── Mediation: partial r(slope, nf_std | rho_nn) ────────────────────")
    out()
    pr_rnn_f, *_ = _partial_r(all_slopes, all_nf_std, all_rnn)
    pr_rnn_e, pr_lo_e, pr_hi_e = _partial_r(e_slopes, e_nf_std, e_rnn)
    out(f"    Full (n=6)  partial_r(slope, nf_std | rho_nn) = {pr_rnn_f:+.3f}")
    out(f"    Excl. (n=5) partial_r(slope, nf_std | rho_nn) = {pr_rnn_e:+.3f}  "
        f"CI [{pr_lo_e:+.3f}, {pr_hi_e:+.3f}]")
    att_rnn = (1 - abs(pr_rnn_e)/max(abs(r_sn_excl), 1e-9)) if not np.isnan(pr_rnn_e) else float("nan")
    out(f"    Attenuation (excl.): {att_rnn:.0%}")
    out()
    out("  ── Mediation: partial r(slope, nf_std | gamma_nn) ──────────────────")
    out()
    pr_gam_f, *_ = _partial_r(all_slopes, all_nf_std, all_gamma_nn)
    pr_gam_e, pr_glo_e, pr_ghi_e = _partial_r(e_slopes, e_nf_std, e_gamma)
    out(f"    Full (n=6)  partial_r(slope, nf_std | gamma_nn) = {pr_gam_f:+.3f}")
    out(f"    Excl. (n=5) partial_r(slope, nf_std | gamma_nn) = {pr_gam_e:+.3f}  "
        f"CI [{pr_glo_e:+.3f}, {pr_ghi_e:+.3f}]")
    att_gam = (1 - abs(pr_gam_e)/max(abs(r_sn_excl), 1e-9)) if not np.isnan(pr_gam_e) else float("nan")
    out(f"    Attenuation (excl.): {att_gam:.0%}")
    out()

    # ──────────────────────────────────────────────────────────────────────
    # VERDICT
    # ──────────────────────────────────────────────────────────────────────
    out("─" * 72)
    out("  VERDICT")
    out("─" * 72)
    out()

    # Evidence rubric (applied to excl. set, n=5):
    # "POSITIVE_STRONG"   = r > 0 AND CI excludes 0
    # "POSITIVE_MODERATE" = r > 0.2 (moderate positive, CI may include 0)
    # "POSITIVE_WEAK"     = 0 < r ≤ 0.2
    # "NON_POSITIVE"      = r ≤ 0

    def _classify(r, lo, hi):
        if np.isnan(r):
            return "UNDEFINED"
        if r > 0 and lo > 0:
            return "POSITIVE_STRONG"
        if r > 0.2:
            return "POSITIVE_MODERATE"
        if r > 0:
            return "POSITIVE_WEAK"
        return "NON_POSITIVE"

    v_rnn = _classify(r_rnn_s_excl, lo_rnn_s, hi_rnn_s)
    v_gam = _classify(r_gam_s_excl, lo_gam_s, hi_gam_s)

    out(f"  rho_nn   ~ slope: {v_rnn}  (r={r_rnn_s_excl:+.3f}, excl. soft_blobs)")
    out(f"  gamma_nn ~ slope: {v_gam}  (r={r_gam_s_excl:+.3f}, excl. soft_blobs)")
    out()

    both_positive = all(v in ("POSITIVE_STRONG", "POSITIVE_MODERATE") for v in (v_rnn, v_gam))
    either_positive = any(v.startswith("POSITIVE") for v in (v_rnn, v_gam))

    if both_positive and (att_rnn > 0.2 or att_gam > 0.2):
        out("  (a) CONFIRMED — both metrics show positive coherence–slope correlation,")
        out("      and at least one mediates the slope↔noise-floor link.")
        out("      Update 6's mechanistic claim is DIRECTLY SUPPORTED: high-slope images")
        out("      produce more spatially correlated null-space hallucinations (reduced")
        out("      n_eff), which inflates the calibration-slope noise floor.")
        verdict_final = "a_confirmed"
    elif both_positive:
        out("  (a-weak) PARTIALLY SUPPORTED — both metrics show positive trends,")
        out("      but mediation is not clearly established (attenuation < 20%).")
        out("      Coherence likely relates to slope but the causal path to noise floor")
        out("      is not directly confirmed at n=5.")
        verdict_final = "a_weak"
    elif either_positive:
        out(f"  (a-mixed) MIXED — one metric shows positive trend, the other does not.")
        out(f"      rho_nn: {v_rnn}, gamma_nn: {v_gam}.")
        out("      Update 6's mechanism is plausible but not robustly supported.")
        verdict_final = "a_mixed"
    else:
        out("  (b) NOT SUPPORTED — neither metric shows a positive coherence–slope")
        out("      correlation. Update 6's causal claim (spatial coherence → lower n_eff")
        out("      → higher noise floor) is UNSUPPORTED by direct measurement.")
        out("      The empirical α≈0.35 fact from Update 6 stands; the proposed")
        out("      MECHANISM remains hypothesised, not confirmed.")
        out("      FLAG: Update 6's causal language should be softened to 'hypothesised'.")
        verdict_final = "b_not_supported"

    out()
    out("  CAVEATS:")
    out("  - n=5 (excl. soft_blobs): all correlations are severely underpowered.")
    out("    CIs span most of [−1, +1]. This test is indicative only.")
    out("  - Sanity check provides reference but not a within-analysis control.")
    out("  - n≥20 images spanning the slope range needed for definitive resolution.")
    out()

    total_time = time.perf_counter() - t_total
    out("─" * 72)
    out(f"  Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
    out(f"  ResShift passes: {N_MEMBERS * (len(MAIN_IMAGES) + len(SANITY_IMAGES))}")
    out("─" * 72)

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nResults written to {OUT_PATH}")
    return verdict_final


if __name__ == "__main__":
    main()
