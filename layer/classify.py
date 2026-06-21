"""
Per-pixel provenance classification: {measured, recovered, invented}.

R2 invariant (hard constraint, never relax)
-------------------------------------------
"invented" means: NOT DETERMINED BY THE INPUT — i.e., the pixel's
energy lives in the null space of A.  It does NOT mean "large",
"surprising", "high-frequency", or any content-based criterion.
The classification criterion is PURELY structural:

    null_fraction[i] = |null_component[i]|
                       ─────────────────────────────────────────
                       |null_component[i]| + |range_component[i]| + eps

This is the fraction of a pixel's energy that A cannot explain.  It
is zero for range-space-only pixels regardless of their magnitude.

Classification rule (explicit thresholds — not magic numbers)
-------------------------------------------------------------
Both thresholds are required constructor parameters.  The caller must
choose them; the defaults below are NOT silently used in production.

    null_fraction > invented_threshold          → INVENTED
    null_fraction ≤ invented_threshold
      AND ensemble_variance > recovered_threshold → RECOVERED
    else                                         → MEASURED

"recovered" captures the case where the range component required a
non-trivial inverse (e.g., deconvolution) rather than a direct copy
from y — indicated by inter-seed ensemble disagreement in the NULL
space around the pixel (not about the pixel's own content).  For v1
without a real model ensemble, ensemble_variance can be None, in which
case nothing is classified as "recovered" and only the null fraction
determines the label.

Thresholds reference
--------------------
  invented_threshold: fraction in (0, 1).
      Rule: set above the maximum null_fraction achievable by floating-
      point round-trip errors on range-space-only estimates (≈ 1e-14 /
      typical signal magnitude ≈ 1e-13).  Set below any genuine null
      content the model supplies (≥ 0.01 in practice).
      Default: 1e-6 (as a fraction) — well-separated from both sides.

  recovered_threshold: per-pixel variance in signal units².
      Only active when ensemble_variance is supplied.
      Default: 1e-12 — below any genuine cross-seed disagreement.

eps (denominator stabiliser)
----------------------------
  eps = 1e-8 in signal units.  Prevents division-by-zero when both
  components are near-zero (e.g., a black pixel).  Does not affect
  the classification of any pixel with non-negligible signal energy.
"""

from __future__ import annotations
import numpy as np

MEASURED: int = 0
RECOVERED: int = 1
INVENTED: int = 2

_LABEL_NAMES = {MEASURED: "measured", RECOVERED: "recovered", INVENTED: "invented"}

DENOMINATOR_EPS: float = 1e-8
"""Denominator stabiliser for null_fraction.  Prevents /0 on dark pixels."""


def classify(
    null_component: np.ndarray,
    range_component: np.ndarray,
    invented_threshold: float,
    recovered_threshold: float,
    ensemble_variance: np.ndarray | None = None,
) -> np.ndarray:
    """Per-pixel provenance label array.

    Parameters
    ----------
    null_component : ndarray
        (I − A⁺A)·x̂ from RectifyResult.null_component.
    range_component : ndarray
        A⁺y from RectifyResult.range_component.
    invented_threshold : float
        null_fraction above this → INVENTED.  Must be in (0, 1).
        See module docstring for how to choose it.
    recovered_threshold : float
        ensemble_variance above this (and null_fraction ≤ invented_threshold)
        → RECOVERED.  Only relevant when ensemble_variance is supplied.
    ensemble_variance : ndarray or None
        Per-pixel variance from ensemble_stats().  If None, nothing is
        labelled RECOVERED; only the INVENTED / MEASURED split applies.

    Returns
    -------
    labels : ndarray, dtype uint8, same shape as null_component
        Per-pixel integer label: MEASURED=0, RECOVERED=1, INVENTED=2.
        Use LABEL_NAMES[label] for the string form.

    Notes
    -----
    This function ONLY reads null_component and ensemble_variance.
    It never reads x_out directly or compares pixel magnitudes to any
    content-derived threshold.  That is the R2 invariant.
    """
    if invented_threshold <= 0 or invented_threshold >= 1:
        raise ValueError(
            f"invented_threshold must be in (0, 1), got {invented_threshold}"
        )

    null = np.abs(np.asarray(null_component, dtype=np.float64))
    rng = np.abs(np.asarray(range_component, dtype=np.float64))

    # null_fraction: fraction of a pixel's energy that is null-space.
    # Purely structural — does not depend on whether the value is
    # "large" or "surprising" (R2).
    null_fraction = null / (null + rng + DENOMINATOR_EPS)

    labels = np.full(null.shape, MEASURED, dtype=np.uint8)

    if ensemble_variance is not None:
        var = np.asarray(ensemble_variance, dtype=np.float64)
        recovered_mask = (null_fraction <= invented_threshold) & (var > recovered_threshold)
        labels[recovered_mask] = RECOVERED

    labels[null_fraction > invented_threshold] = INVENTED

    return labels


def label_name(label: int) -> str:
    """Return the string name for an integer label."""
    return _LABEL_NAMES[label]
