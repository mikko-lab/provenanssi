"""
Ensemble statistics over a set of rectified estimates.

After rectification, all members x_out_i share the SAME range component
(A⁺y is fixed by the measurement) and differ ONLY in their null component
(I−A⁺A)·x̂_i.  Therefore:

    per-pixel variance = variance of null components only
                       = pure epistemic uncertainty about the null space

The range space contributes ZERO variance — not because we average it away,
but because it is identical across all ensemble members by construction (R3).

This property is numerically verifiable and tested in test_layer.py:
    var(A⁺A · x_out_i)  ≤ 1e-12    (range part: identical, zero variance)
    var((I−A⁺A) · x_out_i) > 0     (null part: varies between members)
"""

from __future__ import annotations
import numpy as np


def ensemble_stats(
    x_outs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pixel mean and variance of a set of rectified estimates.

    Parameters
    ----------
    x_outs : list of ndarray, all the same shape
        Rectified estimates {x_out_i}.  Must already be rectified
        (i.e., each satisfies ‖A·x_out_i − y‖ ≤ eps); call
        layer.decompose.rectify() on each before passing here.

    Returns
    -------
    mean : ndarray, same shape as x_outs[0]
        Per-pixel mean across the ensemble.
    variance : ndarray, same shape as x_outs[0]
        Per-pixel variance (ddof=0).  Reflects null-space disagreement only.
        Should be ≤ 1e-12 in the range space of A.
    """
    if len(x_outs) < 2:
        raise ValueError(
            f"ensemble_stats requires at least 2 members, got {len(x_outs)}"
        )
    stack = np.stack([np.asarray(x, dtype=np.float64) for x in x_outs], axis=0)
    return stack.mean(axis=0), stack.var(axis=0)
