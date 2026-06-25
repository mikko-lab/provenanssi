"""
Calibration of the provenance uncertainty map (R8).

What calibration measures
--------------------------
The ensemble produces a per-pixel uncertainty estimate (the standard
deviation of x_out_i across members).  For that estimate to be useful,
it must PREDICT how wrong the final reconstruction is:

    predicted_std[p]  = std_i(x_out_i[p])          (ensemble spread)
    actual_error[p]   = |mean_i(x_out_i[p]) − x_gt[p]|   (true error)

A perfectly calibrated uncertainty map satisfies actual_error ≈ predicted_std
pixel-by-pixel, so the reliability curve (predicted vs actual, binned by
predicted) lies on the diagonal.

Why the range space is excluded
---------------------------------
Range-space pixels have predicted_std ≈ 0 (the range component A⁺y is
identical across all ensemble members) AND actual_error ≈ 0 (A⁺y = A⁺Ax_gt
exactly for x_gt in the range space).  Including them trivially inflates
the "well-calibrated" count without testing anything.

Pixels with predicted_std < min_predicted_std are excluded from the
reliability curve.  This is purely a structural filter (based on the
uncertainty the ensemble assigned, not on pixel content or magnitude),
consistent with R2.

Calibration metrics
-------------------
Three complementary metrics are computed on the binned reliability curve:

1. Pearson r — correlation of (bin_predicted_std, bin_actual_error).
   Well-calibrated: r ≈ +1 (high uncertainty → high error, monotone).
   Miscalibrated:   r ≈  0 or negative.

2. slope — linear-regression slope of actual_error ~ predicted_std.
   Well-calibrated: slope ≈ 1 (one unit of uncertainty = one unit of error).
   Overconfident:   slope >> 1 (uncertainty understates actual error).
   Underconfident:  slope << 1.

3. ECE (Expected Calibration Error) — size-weighted mean |actual − predicted|
   per bin.  Well-calibrated: ECE ≈ 0.

is_calibrated() takes ALL thresholds as explicit documented parameters
(none are magic numbers).  Default values are chosen so the oracle oracle
tests in test_calibrate.py pass / fail unambiguously.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Output of calibrate().

    Attributes
    ----------
    bin_predicted_std : ndarray, shape (B,)
        Mean predicted std in each bin (x-axis of the reliability curve).
    bin_actual_error : ndarray, shape (B,)
        Mean actual error in each bin (y-axis of the reliability curve).
    n_per_bin : ndarray, shape (B,), dtype int
        Number of pixels in each bin.
    pearson_r : float
        Pearson correlation of (bin_predicted_std, bin_actual_error).
        NaN if fewer than 2 non-empty bins.
    slope : float
        Linear-regression slope: actual_error ~ predicted_std.
        NaN if fewer than 2 non-empty bins.
    ece : float
        Expected Calibration Error: Σ_b (n_b/N) |actual_b − predicted_b|.
        NaN if no calibrated pixels.
    n_pixels_calibrated : int
        Number of pixels with predicted_std ≥ min_predicted_std.
    """
    bin_predicted_std: np.ndarray
    bin_actual_error: np.ndarray
    n_per_bin: np.ndarray
    pearson_r: float
    slope: float
    ece: float
    n_pixels_calibrated: int


def calibrate(
    x_outs: list[np.ndarray],
    x_gt: np.ndarray,
    n_bins: int = 10,
    min_predicted_std: float = 1e-6,
) -> CalibrationResult:
    """Compute the reliability curve and calibration metrics.

    Parameters
    ----------
    x_outs : list of ndarray, all same shape
        Rectified ensemble {x_out_i}.  Must have ≥ 2 members.
    x_gt : ndarray, same shape as x_outs[0]
        Ground-truth image.  R4: used here only for evaluation.
        Never feed this to the oracle or to any layer function.
    n_bins : int
        Number of reliability bins.  Pixels are sorted by predicted_std
        and split into n_bins equal-count quantile bins.
    min_predicted_std : float
        Pixels with predicted_std < this are excluded from the curve.
        Covers range-space pixels (std ≈ 0 for all operators).
        This is an EXPLICIT calibration parameter — not a magic number.
        Default: 1e-6 (safely below any genuine null-space spread but
        above float64 rounding noise on identical ensemble members).

    Returns
    -------
    CalibrationResult
        Reliability curve and scalar metrics.  See CalibrationResult docs.

    Notes
    -----
    Pearson r and slope are computed on the per-bin means, not per-pixel.
    This is more robust when many pixels share the same predicted_std
    (as in the oracle oracle tests where pixels within a region are identical
    by construction).  With n_bins=3 bins that cleanly separate 3 oracle
    regions the metrics are analytically predictable.
    """
    if len(x_outs) < 2:
        raise ValueError(f"calibrate requires at least 2 ensemble members, got {len(x_outs)}")

    stack = np.stack([np.asarray(x, dtype=np.float64) for x in x_outs], axis=0)
    mean_x = stack.mean(axis=0)
    pred_std = stack.std(axis=0)

    x_gt_arr = np.asarray(x_gt, dtype=np.float64)
    actual_err = np.abs(mean_x - x_gt_arr)

    mask = pred_std.ravel() >= min_predicted_std
    pred_flat = pred_std.ravel()[mask]
    err_flat = actual_err.ravel()[mask]
    n_cal = int(mask.sum())

    if n_cal == 0:
        empty = np.array([], dtype=np.float64)
        return CalibrationResult(
            bin_predicted_std=empty,
            bin_actual_error=empty,
            n_per_bin=np.array([], dtype=int),
            pearson_r=float("nan"),
            slope=float("nan"),
            ece=float("nan"),
            n_pixels_calibrated=0,
        )

    effective_bins = min(n_bins, n_cal)
    # quantile-based bin edges: equal pixel count per bin
    quantile_edges = np.linspace(0, 100, effective_bins + 1)
    bin_edges = np.percentile(pred_flat, quantile_edges)
    # avoid duplicate edges collapsing bins
    bin_edges = np.unique(bin_edges)
    actual_n_bins = len(bin_edges) - 1

    bin_pred = np.empty(actual_n_bins)
    bin_actual = np.empty(actual_n_bins)
    bin_n = np.zeros(actual_n_bins, dtype=int)

    for b in range(actual_n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b < actual_n_bins - 1:
            in_bin = (pred_flat >= lo) & (pred_flat < hi)
        else:
            in_bin = pred_flat >= lo     # last bin: inclusive on right
        bin_n[b] = int(in_bin.sum())
        if bin_n[b] > 0:
            bin_pred[b] = pred_flat[in_bin].mean()
            bin_actual[b] = err_flat[in_bin].mean()
        else:
            bin_pred[b] = float("nan")
            bin_actual[b] = float("nan")

    nonempty = ~np.isnan(bin_pred)
    if nonempty.sum() < 2:
        pearson_r = float("nan")
        slope = float("nan")
    else:
        bp = bin_pred[nonempty]
        ba = bin_actual[nonempty]
        # Pearson r
        bp_c = bp - bp.mean()
        ba_c = ba - ba.mean()
        denom = np.sqrt((bp_c**2).sum() * (ba_c**2).sum())
        pearson_r = float((bp_c * ba_c).sum() / denom) if denom > 0 else float("nan")
        # linear regression slope (actual ~ predicted)
        var_pred = (bp_c**2).sum()
        slope = float((bp_c * ba_c).sum() / var_pred) if var_pred > 0 else float("nan")

    # ECE: size-weighted mean |actual_b - predicted_b|
    nonempty_n = bin_n[nonempty]
    ece = float(
        (nonempty_n * np.abs(bin_actual[nonempty] - bin_pred[nonempty])).sum() / nonempty_n.sum()
    ) if nonempty.sum() > 0 else float("nan")

    return CalibrationResult(
        bin_predicted_std=bin_pred,
        bin_actual_error=bin_actual,
        n_per_bin=bin_n,
        pearson_r=pearson_r,
        slope=slope,
        ece=ece,
        n_pixels_calibrated=n_cal,
    )


def is_calibrated(
    result: CalibrationResult,
    min_pearson_r: float,
    max_ece: float,
    min_slope: float,
    max_slope: float,
) -> bool:
    """Return True if all calibration criteria are satisfied.

    ALL four thresholds are EXPLICIT parameters — none are magic numbers.
    The caller decides what counts as acceptable calibration.

    Parameters
    ----------
    result : CalibrationResult
        Output of calibrate().
    min_pearson_r : float
        Minimum acceptable Pearson r of (bin_predicted_std, bin_actual_error).
        A well-calibrated ensemble has r close to +1; r below this threshold
        means the uncertainty does not predict error (even in rank order).
        Recommended starting value: 0.9.
    max_ece : float
        Maximum acceptable ECE (Expected Calibration Error).
        ECE = 0 means perfect magnitude calibration.
        Recommended starting value: 0.3 (in the units of x_out).
    min_slope : float
        Minimum acceptable regression slope (actual ~ predicted).
        Slope < 1 → overconfident (uncertainty overstates error).
        Recommended minimum: 0.5.
    max_slope : float
        Maximum acceptable regression slope.
        Slope > 1 → underconfident (uncertainty understates error).
        Recommended maximum: 2.0.

    Returns
    -------
    bool
        True if r ≥ min_pearson_r AND ece ≤ max_ece AND min_slope ≤ slope ≤ max_slope.
        Any NaN metric returns False (too few calibrated pixels).
    """
    r = result.pearson_r
    e = result.ece
    s = result.slope

    if any(np.isnan(v) for v in (r, e, s)):
        return False

    return (r >= min_pearson_r) and (e <= max_ece) and (min_slope <= s <= max_slope)
