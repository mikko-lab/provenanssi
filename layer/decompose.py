"""
Provenance rectification: rectify(x̂, y, operator) → RectifyResult.

DDNM-style data-consistent rectification (R3):

    x_out = A⁺y + (I − A⁺A)·x̂

This forces x_out to satisfy A·x_out = y (noiseless) by replacing the
range component of x̂ with A⁺y, while preserving the null component
(I−A⁺A)·x̂ exactly as the model supplied it.

Why this is the right decomposition
-------------------------------------
  range_component = A⁺y      — determined entirely by the measurement.
                               These pixels are "measured" or "recovered."
  null_component  = (I−A⁺A)·x̂ — NOT in the measurement.  Whatever the
                               model put here, the input gave us zero
                               information about it.  These pixels are
                               "invented."  (R2: large values are NOT the
                               criterion — only null-space membership is.)

Data consistency (R3)
---------------------
rectify() ALWAYS asserts ‖A·x_out − y‖ ≤ consistency_eps and raises
ValueError if violated.  This is the single most important guardrail.
consistency_eps defaults to CONSISTENCY_EPS (below) and may be overridden
for special operator regimes — but it may NOT be set to ∞ to silence a
failure.  A violated residual means the operator math is wrong; fix it.

Typical achieved residuals (noiseless case):
  BoxDownsample  : ≤ 1e-14   (exact arithmetic, no FFT)
  MaskOperator   : ≤ 1e-14   (exact arithmetic)
  CircularBlur   : ≤ 1e-12   (FFT round-trip errors)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from operators.base import Operator

CONSISTENCY_EPS: float = 1e-10
"""Default tolerance for ‖A·x_out − y‖.  Covers all three v1 operators
with margin.  For exact operators (BoxDownsample, MaskOperator) the
achieved residual is typically ≤ 1e-14."""


@dataclass
class RectifyResult:
    """Output of rectify().

    Attributes
    ----------
    x_out : ndarray
        The rectified estimate.  Always satisfies ‖A·x_out − y‖ ≤ eps.
    range_component : ndarray
        A⁺y — same shape as x_out.  Determined by the measurement alone.
    null_component : ndarray
        (I−A⁺A)·x̂ — same shape as x_out.  The "invented" part.
    residual : float
        ‖A·x_out − y‖ computed after rectification.  Logged for R5.
    """
    x_out: np.ndarray
    range_component: np.ndarray
    null_component: np.ndarray
    residual: float


def rectify(
    x_hat: np.ndarray,
    y: np.ndarray,
    operator: Operator,
    consistency_eps: float = CONSISTENCY_EPS,
) -> RectifyResult:
    """Enforce data consistency and decompose x̂ into range + null.

    Parameters
    ----------
    x_hat : ndarray
        Raw model estimate (or oracle output), same shape as the
        high-quality image x.
    y : ndarray
        Observed degraded measurement.  Must satisfy y = A·x for some x.
    operator : Operator
        The known linear degradation operator A.
    consistency_eps : float
        Tolerance for ‖A·x_out − y‖.  Default CONSISTENCY_EPS = 1e-10.
        Raise if violated — do not silently accept.

    Returns
    -------
    RectifyResult
        x_out, range_component, null_component, residual.

    Raises
    ------
    ValueError
        If ‖A·x_out − y‖ > consistency_eps.  This means operator math is
        wrong.  Fix the code, do not raise the tolerance.
    """
    x_hat = np.asarray(x_hat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    range_component = operator.pinv(y)
    null_component = x_hat - operator.project(x_hat)
    x_out = range_component + null_component

    residual = float(np.linalg.norm(operator.forward(x_out) - y))
    if residual > consistency_eps:
        raise ValueError(
            f"Data consistency violated (R3): "
            f"‖A·x_out − y‖ = {residual:.6e} > consistency_eps = {consistency_eps:.6e}. "
            f"The operator math must be wrong — fix the code, do not raise the tolerance."
        )

    return RectifyResult(
        x_out=x_out,
        range_component=range_component,
        null_component=null_component,
        residual=residual,
    )
