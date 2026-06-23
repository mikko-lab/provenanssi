"""
eval/slope_noise_mechanism.py — Mechanistic analysis of slope→noise-floor relationship.

QUESTION: Why does the per-image slope noise floor appear to scale with slope magnitude?
Is this an estimator artifact (generic property of the OLS regression) or a model/data
effect (a real property of ResShift's behaviour on these images)?

METHOD:
  1. Analytic derivation — work out Var(β̂) from OLS algebra and state what it predicts
     about the slope→noise relationship under the iid-pixel assumption.
  2. Empirical test — fit a power law std(β̂) ∝ slope^α using existing per-window
     data (no new GPU passes needed). Test whether α is distinguishable from 0.
  3. N-scaling check — compare SD at N=12 vs N=48 for face images; under iid,
     SD ∝ 1/√N. Departures indicate non-iid structure.
  4. Verdict — estimator artifact (predicts α=0) vs model/data effect (α>0).

DATA SOURCE: all numbers are taken directly from results files computed in prior sessions.
No ResShift forward passes are run here.

Usage:
    python eval/slope_noise_mechanism.py

Output: printed report + eval/slope_noise_mechanism_results.txt
"""
from __future__ import annotations

import os
import sys
import numpy as np

OUT_PATH = os.path.join(os.path.dirname(__file__), "slope_noise_mechanism_results.txt")

# ─── all existing per-window slope data ─────────────────────────────────────
# Source: close_findings_results.txt + faces_noise_floor_results.txt
# Protocol: 5 non-overlapping N=12 windows for all images; additionally 5 N=48
# windows for the three face images.

PW_N12 = {
    "wood_grain":    [0.2645, 0.2355, 0.2154, 0.2123, 0.2337],   # null=0.0159
    "boardwalk":     [0.9602, 0.9556, 0.9508, 0.8932, 0.9776],   # null=0.0046
    "soft_blobs":    [4.3419, 4.7069, 4.1364, 4.7883, 3.8892],   # null=0.0014
    "boy_face":      [3.2046, 3.1504, 3.1621, 3.1106, 3.2603],   # null=0.0049
    "girl_sad_face": [2.7936, 2.8291, 2.7364, 2.7797, 2.8415],   # null=0.0032
    "wayuu_woman":   [1.5941, 1.6118, 1.6508, 1.5149, 1.5523],   # null=0.0162
}

PW_N48 = {
    "boy_face":      [3.1829, 3.1748, 3.1377, 3.0676, 3.1276],
    "girl_sad_face": [2.7834, 2.7409, 2.7367, 2.7012, 2.7428],
    "wayuu_woman":   [1.7483, 1.7409, 1.7442, 1.7144, 1.7001],
}

NULL_FRAC = {
    "wood_grain":    0.0159,
    "boardwalk":     0.0046,
    "soft_blobs":    0.0014,
    "boy_face":      0.0049,
    "girl_sad_face": 0.0032,
    "wayuu_woman":   0.0162,
}


def _stats(slopes: list[float]) -> tuple[float, float, float]:
    """Return (mean, std, range) of a list of slope estimates."""
    arr = np.array(slopes, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.max() - arr.min())


def _t_crit(df: int, p: float = 0.025) -> float:
    """Two-tailed t critical value for given df at significance p."""
    # Hard-coded for small df values used here; avoids scipy dependency.
    # Values from standard t-table (two-tailed 95% CI, p=0.025 each tail).
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(df, 2.0)  # fallback to z=2 for df>10


def _ols_log_log(
    slopes: list[float], stds: list[float]
) -> tuple[float, float, float, float, float, float]:
    """
    Fit log(std) = log(a) + alpha*log(slope) by OLS.
    Returns (alpha, log_a, r_squared, se_alpha, ci_lo, ci_hi)
    using t-distribution with n-2 degrees of freedom.
    """
    x = np.log(slopes)
    y = np.log(stds)
    n = len(x)
    x_bar, y_bar = x.mean(), y.mean()
    ss_xx = ((x - x_bar)**2).sum()
    ss_xy = ((x - x_bar) * (y - y_bar)).sum()
    alpha = ss_xy / ss_xx
    log_a = y_bar - alpha * x_bar
    y_hat = log_a + alpha * x
    ss_res = ((y - y_hat)**2).sum()
    ss_tot = ((y - y_bar)**2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sigma2 = ss_res / (n - 2) if n > 2 else float("nan")
    se_alpha = float(np.sqrt(sigma2 / ss_xx)) if n > 2 else float("nan")
    df = n - 2
    tc = _t_crit(df)
    ci_lo = alpha - tc * se_alpha
    ci_hi = alpha + tc * se_alpha
    return float(alpha), float(log_a), float(r2), se_alpha, ci_lo, ci_hi


def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("=" * 72)
    out("  SLOPE → NOISE-FLOOR MECHANISM ANALYSIS")
    out("  ResShift + BicubicDownsample(4) — existing data only, no new GPU passes")
    out("=" * 72)
    out()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — ANALYTIC DERIVATION
    # ══════════════════════════════════════════════════════════════════════
    out("─" * 72)
    out("  SECTION 1: ANALYTIC PREDICTION")
    out("─" * 72)
    out()
    out("  Calibration setup (exact, from layer/calibrate.py):")
    out("    stack = {x_out_i}, i=1..N   (rectified ensemble members)")
    out("    mean_x[p]   = mean_i(x_out_i[p])       (per-pixel ensemble mean)")
    out("    pred_std[p] = std_i(x_out_i[p])         (per-pixel ensemble std)")
    out("    actual_err[p] = |mean_x[p] - x_gt[p]|")
    out("    K=10 quantile bins by pred_std → (x_k, y_k) bin means")
    out("    slope  β̂ = Σ(x_k - x̄)(y_k - ȳ) / Σ(x_k - x̄)²   [OLS]")
    out()
    out("  Derivation of Var(β̂) across independent N-sample windows:")
    out()
    out("  Let μ(p) = E[x̂_j(p)], σ²(p) = Var[x̂_j(p)] across seeds.")
    out("  Since x_out_i = A⁺y + (I−A⁺A)x̂_i, the range part A⁺y is FIXED,")
    out("  so std_i(x_out_i[p]) = std_i((I−A⁺A)x̂_i[p]).")
    out()
    out("  Mean reconstruction: mean_x[p] = A⁺y[p] + (I−A⁺A)·(1/N Σ_i x̂_i[p])")
    out("  For null-space pixel p:  actual_err[p] = |(I−A⁺A)(mean_x[p] − x_gt[p])|")
    out("                                         = |bias[p] + δ[p]|")
    out("  where:")
    out("    bias[p] = (I−A⁺A)(μ(p) − x_gt[p])  [irreducible, fixed per image]")
    out("    δ[p]    = (I−A⁺A)(1/N Σ_i x̂_i[p] − μ(p))  [sampling noise ~N(0,σ²(p)/N)]")
    out()
    out("  In the regime bias(p) >> δ(p) (consistent model predictions):")
    out("    E[actual_err(p)] ≈ |bias(p)| ≡ β·pred_std(p)  (calibration slope β)")
    out("    Var_windows[actual_err(p)] ≈ Var[δ(p)] ≈ σ²(p)/N ≈ pred_std²(p)/N")
    out()
    out("  Bin k contains n_k pixels.  Under the IID PIXEL ASSUMPTION")
    out("  (independent actual_err across pixels within a bin):")
    out("    Var_windows[y_k] ≈ (1/n_k²) Σ_{p∈k} pred_std²(p)/N")
    out("                     ≈ x_k²/(n_k · N)    [since pred_std(p)≈x_k in bin k]")
    out()
    out("  OLS slope variance (heteroscedastic):")
    out("    Var(β̂) = Σ_k c_k² · Var(y_k)   where c_k = (x_k−x̄)/Σ(x_j−x̄)²")
    out("           = (1/(n_eff·N)) · Σ_k c_k² · x_k²")
    out("    where n_eff ≤ n_k accounts for pixel correlation.")
    out()
    out("  KEY PROPERTY — SCALE INVARIANCE: if all x_k → s·x_k (same distribution")
    out("  shape, rescaled), then c_k → c_k/s, Var(y_k) → s²·Var(y_k), and")
    out("  Var(β̂) is UNCHANGED.  The slope estimate's variance does not depend")
    out("  on the magnitude of predicted_std values.")
    out()
    out("  ┌─────────────────────────────────────────────────────────────────┐")
    out("  │ IID PREDICTION (null hypothesis):                               │")
    out("  │   SD(β̂) ∝ 1/√(n_eff · N)  ×  f(x_k distribution shape)        │")
    out("  │   α = 0 (slope noise floor is β-INDEPENDENT under iid pixels)  │")
    out("  └─────────────────────────────────────────────────────────────────┘")
    out()
    out("  If α > 0 empirically, the iid assumption is violated.  The only free")
    out("  parameter is n_eff — the effective number of independent pixels per bin.")
    out("  n_eff < n_k when pixel errors are spatially correlated within a window.")
    out("  If n_eff varies systematically with image type (lower for images where")
    out("  ResShift generates spatially coherent hallucinations), a correlation")
    out("  between slope and noise floor could arise through that mediating variable.")
    out("  That would be a MODEL/DATA EFFECT, not an estimator artifact.")
    out()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — EMPIRICAL TEST: POWER LAW FIT
    # ══════════════════════════════════════════════════════════════════════
    out("─" * 72)
    out("  SECTION 2: EMPIRICAL POWER LAW FIT  (N=12, 5 windows each)")
    out("─" * 72)
    out()

    images = list(PW_N12.keys())
    slope_means, slope_stds, slope_ranges = [], [], []
    rows = []
    for img in images:
        m, s, r = _stats(PW_N12[img])
        slope_means.append(m)
        slope_stds.append(s)
        slope_ranges.append(r)
        rows.append((img, NULL_FRAC.get(img, float("nan")), m, s, r))

    out(f"  {'Image':<16} {'null_frac':>9}  {'slope_mean':>10}  {'std':>8}  {'range':>8}  {'std/slope':>9}")
    out("  " + "─" * 66)
    for img, nf, m, s, r in rows:
        rel = s / m if m > 0 else float("nan")
        out(f"  {img:<16} {nf:>9.4f}  {m:>10.4f}  {s:>8.4f}  {r:>8.4f}  {rel:>9.4f}")
    out()

    # Power law fit: std = a * slope^alpha
    alpha, log_a, r2, se_alpha, ci_lo, ci_hi = _ols_log_log(slope_means, slope_stds)
    a = np.exp(log_a)
    df_full = len(slope_means) - 2
    tc_full = _t_crit(df_full)

    out(f"  Power law fit: std ≈ {a:.5f} × slope^α")
    out(f"    α̂ = {alpha:.3f}  (SE = {se_alpha:.3f},  t-corrected 95% CI (df={df_full}): [{ci_lo:.3f}, {ci_hi:.3f}])")
    out(f"    R² (log-log fit) = {r2:.3f}")
    out(f"    IID prediction:  α = 0  (β-independent noise floor)")
    out()

    # Check if α=0 is within CI
    zero_in_ci = ci_lo <= 0 <= ci_hi
    out(f"    α=0 within 95% CI? {'YES (cannot reject iid)' if zero_in_ci else 'NO (iid rejected at p<0.05)'}")
    out()

    # Predicted vs actual
    out("  Predicted noise floor (std) from fitted model vs actual:")
    out(f"  {'Image':<16} {'slope':>8}  {'actual_std':>10}  {'fitted_std':>10}  {'ratio':>6}")
    out("  " + "─" * 56)
    for img, nf, m, s, r in rows:
        fitted = float(a * m**alpha)
        ratio = s / fitted if fitted > 0 else float("nan")
        out(f"  {img:<16} {m:>8.4f}  {s:>10.4f}  {fitted:>10.4f}  {ratio:>6.2f}")
    out()

    # soft_blobs is qualitatively different regime — fit without it
    other_images = [(img, m, s) for img, _, m, s, _ in rows if img != "soft_blobs"]
    om, os_ = [x[1] for x in other_images], [x[2] for x in other_images]
    alpha2, log_a2, r2_2, se2, ci2_lo, ci2_hi = _ols_log_log(om, os_)
    a2 = np.exp(log_a2)
    df2 = len(om) - 2
    out("  Fit WITHOUT soft_blobs (n=5; soft_blobs is degenerate-regime outlier):")
    out(f"    α̂ = {alpha2:.3f}  (SE = {se2:.3f},  t-corrected 95% CI (df={df2}): [{ci2_lo:.3f}, {ci2_hi:.3f}])")
    out(f"    R² = {r2_2:.3f}")
    zero2_in_ci = ci2_lo <= 0 <= ci2_hi
    out(f"    α=0 within 95% CI? {'YES (cannot reject iid)' if zero2_in_ci else 'NO (iid rejected at p<0.05)'}")
    out()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3 — N-SCALING CHECK (1/√N test)
    # ══════════════════════════════════════════════════════════════════════
    out("─" * 72)
    out("  SECTION 3: N-SCALING CHECK  (does SD scale as 1/√N?)")
    out("─" * 72)
    out()
    out("  IID prediction: SD(β̂) ∝ 1/√N → SD(N=48)/SD(N=12) = 1/√4 = 0.500")
    out()
    out(f"  {'Image':<16} {'SD@N12':>8}  {'SD@N48':>8}  {'ratio':>6}  {'expected':>8}  {'ratio/0.5':>9}")
    out("  " + "─" * 60)
    for img in PW_N48:
        _, s12, _ = _stats(PW_N12[img])
        _, s48, _ = _stats(PW_N48[img])
        ratio = s48 / s12 if s12 > 0 else float("nan")
        out(f"  {img:<16} {s12:>8.4f}  {s48:>8.4f}  {ratio:>6.3f}  {'0.500':>8}  {ratio/0.5:>9.3f}")
    out()
    out("  Note: with only 5 windows, each SD estimate has high uncertainty.")
    out("  The SE of std from 5 iid windows ≈ std/√(2(n-1)) ≈ std/2.8.")
    out("  Ratios can deviate from 0.5 substantially by chance alone.")
    out()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4 — SOFT_BLOBS AS A SEPARATE REGIME
    # ══════════════════════════════════════════════════════════════════════
    out("─" * 72)
    out("  SECTION 4: SOFT_BLOBS — DEGENERATE REGIME")
    out("─" * 72)
    out()
    sb_m, sb_s, sb_r = _stats(PW_N12["soft_blobs"])
    bw_m, bw_s, bw_r = _stats(PW_N12["boardwalk"])
    out(f"  soft_blobs: slope={sb_m:.4f}, null_frac=0.0014 — almost no null-space energy.")
    out(f"  Ensemble std (predicted uncertainty) is near-zero for smooth blobs.")
    out(f"  The calibration bins have tiny x_k values; even small seed-to-seed")
    out(f"  fluctuations in the ensemble mean produce large changes in the regression")
    out(f"  slope (fitting a near-flat signal). This is an ILL-CONDITIONING effect.")
    out()
    out(f"  soft_blobs  std={sb_s:.4f}  range={sb_r:.4f}  (11× boardwalk std of {bw_s:.4f})")
    out(f"  This outlier behaviour is CONSISTENT with the IID estimator model:")
    out(f"  when n_k·pred_std² is tiny, Var(y_k) explodes → Var(β̂) explodes.")
    out(f"  It is an ESTIMATOR ARTIFACT of the degenerate-signal regime, unrelated")
    out(f"  to slope magnitude per se (there are low-slope images with low null that")
    out(f"  would show the same behaviour).")
    out()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5 — VERDICT AND IMPLICATION FOR §10 SNR
    # ══════════════════════════════════════════════════════════════════════
    out("─" * 72)
    out("  SECTION 5: VERDICT")
    out("─" * 72)
    out()

    # Coefficient of variation (relative noise) by image
    out("  Relative noise (std/slope) by image — β-independent would give constant:")
    out()
    out(f"  {'Image':<16} {'slope':>8}  {'std':>8}  {'std/slope':>10}")
    out("  " + "─" * 46)
    for img, nf, m, s, r in rows:
        rel = s / m
        out(f"  {img:<16} {m:>8.4f}  {s:>8.4f}  {rel:>10.4f}")
    out()
    out("  If noise floor were β-proportional (α=1): std/slope would be constant.")
    out("  If β-independent (α=0): std would be constant, std/slope ∝ 1/slope.")
    out()

    bw_m2, bw_s2, _ = _stats(PW_N12["boardwalk"])
    bf_m2, bf_s2, _ = _stats(PW_N12["boy_face"])
    slope_ratio = bf_m2 / bw_m2
    std_ratio = bf_s2 / bw_s2
    sqrt_ratio = np.sqrt(slope_ratio)

    all_in_ci = ci_lo <= 0 <= ci_hi
    some_in_ci = ci2_lo <= 0 <= ci2_hi

    out(f"  POWER LAW FIT RESULT: α̂ = {alpha:.2f} (full n=6, t-CI [{ci_lo:.2f},{ci_hi:.2f}]),  "
        f"α̂ = {alpha2:.2f} (excl. soft_blobs n=5, t-CI [{ci2_lo:.2f},{ci2_hi:.2f}])")
    out()
    out("  The boardwalk vs. boy_face comparison (matched null_frac ≈ 0.0046-0.0049,")
    out("  the cleanest controlled pair in the dataset):")
    out(f"    slope ratio: {bf_m2:.4f}/{bw_m2:.4f} = {slope_ratio:.2f}×")
    out(f"    std ratio:   {bf_s2:.4f}/{bw_s2:.4f} = {std_ratio:.2f}×")
    out(f"    If α=1 (proportional): expected std ratio = {slope_ratio:.2f}× — NOT supported")
    out(f"    If α=0 (independent):  expected std ratio = 1.00×  — NOT supported")
    out(f"    If α=0.5 (sqrt):       expected std ratio = {sqrt_ratio:.2f}× — supported (actual 1.78×)")
    out()
    out("  CONCLUSION — MODEL/DATA EFFECT, not pure estimator artifact:")
    out()
    out("  (a) The analytic derivation shows that under iid pixels, SD(β̂) is")
    out("      β-INDEPENDENT and scale-invariant (α=0). The OLS regression estimator")
    out("      itself does NOT produce slope-proportional noise.")
    out()
    out(f"  (b) The data (excl. soft_blobs, n=5) give α̂={alpha2:.2f}, t-CI [{ci2_lo:.2f},{ci2_hi:.2f}],")
    out(f"      excluding 0 at p<0.05 (df={df2}). The full fit (n=6) t-CI [{ci_lo:.2f},{ci_hi:.2f}]")
    out(f"      {'includes' if all_in_ci else 'also excludes'} 0, with soft_blobs as a high-leverage outlier.")
    out("      The restricted fit is the more reliable signal — soft_blobs is a")
    out("      known degenerate-regime case with a separate mechanism (§4).")
    out()
    out(f"  (c) The matched boardwalk/boy_face pair (same null_frac) shows {std_ratio:.2f}× std")
    out(f"      ratio for {slope_ratio:.1f}× slope ratio — consistent with α≈0.5, inconsistent")
    out(f"      with both α=0 and α=1.")
    out()
    out("  (d) soft_blobs is ill-conditioning from low signal, a separate estimator")
    out("      artifact not related to slope magnitude.")
    out()
    out("  MECHANISTIC INTERPRETATION: The slope noise floor scales as slope^α with")
    out("  α≈0.35 (sub-proportional). The OLS estimator predicts α=0 under iid")
    out("  pixels. The empirical α>0 therefore indicates that pixel reconstruction")
    out("  errors are NOT iid within bins — they are spatially correlated. The")
    out("  effective number of independent observations n_eff is lower than the raw")
    out("  pixel count n_k. Images with higher calibration slopes (faces) appear to")
    out("  have lower n_eff, suggesting ResShift generates more spatially coherent")
    out("  (correlated) hallucinations for those images. This is a MODEL/DATA EFFECT.")
    out()
    out("  CAVEAT: n=5 data points (excluding soft_blobs) spanning one order of")
    out("  magnitude in slope. The point estimate α̂=0.35 is plausible but imprecise.")
    out("  The conclusion (model effect, not estimator artifact) is supported by")
    out("  the analytic prediction AND the matched-pair evidence, not just the fit.")
    out()
    out("─" * 72)
    out("  IMPLICATION FOR §10 DOMAIN-SHIFT SNR")
    out("─" * 72)
    out()
    out("  The SNR calculation in §10 (natural↔faces, 9.2×) already used the")
    out("  FACES GROUP's own noise floor (boy_face N=12 range=0.1497) as the")
    out("  denominator — the side of the contrast with the HIGHER slope and HIGHER")
    out("  noise floor. This makes the SNR calculation CONSERVATIVE.")
    out()
    out("  If slope magnitude drives noise floor (slope-proportional noise floor),")
    out("  then the faces noise floor is partly inflated by faces having high slopes.")
    out("  Using that inflated denominator for the contrast PENALISES the SNR.")
    out("  The 'honest' SNR under a slope-normalised noise measure would be HIGHER,")
    out("  not lower, than the stated 9.2×.")
    out()
    out("  Concretely: natural mean slope ≈ 1.19 vs. faces mean slope ≈ 2.57.")
    out("  If noise floor ∝ slope, the natural group's noise floor is roughly")
    nat_scale = bw_s2 / bw_m2
    out(f"  {nat_scale:.4f}×slope = {nat_scale*1.19:.4f} for naturals, and {nat_scale*2.57:.4f} for faces.")
    out(f"  The actual worst noise floor used was 0.1497 (boy_face). The natural")
    out(f"  group's noise floor (boardwalk SD=0.0286) is already captured separately.")
    out()
    out("  VERDICT: the slope-noise mechanism, even if real, makes the §10 9.2×")
    out("  SNR MORE conservative, not less. The §10 verdict ('faces slope elevation")
    out("  is 9× above worst in-contrast noise floor') stands. The mechanism does")
    out("  NOT weaken §10 — if anything it implies the true SNR is higher.")
    out()
    out("  This finding does not require editing §10. Flagged here for completeness.")
    out()
    out("=" * 72)
    out("  SUMMARY")
    out("=" * 72)
    out()
    out(f"  Analytic prediction (iid pixels): α = 0 (slope-independent noise floor)")
    out(f"  Empirical fit (n=6):             α̂ = {alpha:.2f}, CI [{ci_lo:.2f}, {ci_hi:.2f}]")
    out(f"  Empirical fit (n=5, excl. soft):  α̂ = {alpha2:.2f}, CI [{ci2_lo:.2f}, {ci2_hi:.2f}]")
    out(f"  Soft_blobs:  degenerate-regime ill-conditioning (estimator artifact, not slope-driven)")
    out(f"  N-scaling:   noisy (5-window std estimates); consistent with 1/√N within uncertainty")
    out(f"  §10 impact:  NONE — the SNR calculation is already conservative on the mechanism side")
    out()
    out(f"  (a) vs (b) verdict: MODEL/DATA EFFECT (n_eff varies with image type).")
    out(f"  Evidence: α=0 rejected at p<0.05 (excl. soft_blobs); boardwalk/boy_face")
    out(f"  matched pair consistent with α≈0.5; analytic derivation shows estimator")
    out(f"  predicts α=0 under iid, so α>0 implicates the data structure, not the")
    out(f"  estimator. The effect is sub-proportional (α≈0.35–0.5 < 1).")
    out(f"  To characterise precisely: n≥20 images with repeat noise-floor measurements,")
    out(f"  spanning slope range with null_frac matched across slope levels.")
    out(f"  This is a future investigation — not a blocker for §10.")
    out()

    result_text = "\n".join(lines)
    with open(OUT_PATH, "w") as f:
        f.write(result_text)
    print(f"\nResults written to {OUT_PATH}")


if __name__ == "__main__":
    main()
