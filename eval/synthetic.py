"""
Synthetic case generator for calibration testing (R4, R8).

Produces ground-truth images, degraded measurements, and oracle ensembles
where the null-space content of each member is drawn from a controlled
distribution.  This lets calibrate() be evaluated against exact known truth.

R4 compliance
-------------
x_gt is produced here, stored in SyntheticCase, and passed to calibrate()
for evaluation ONLY.  It is NEVER fed to OracleEngine.restore() or to any
layer function.  The oracle only receives y = A·x_gt (the degraded observation)
and caller-chosen null_content arrays — never x_gt itself.

Calibration design
------------------
Each "null region" specifies:
  mask   : boolean ndarray, True where this region covers null-space pixels
  bias   : float — the oracle guesses (x_gt + bias) for pixels in this region
  spread : float — std of ensemble perturbation around that guess

Under this model, with a large ensemble:
  mean(x_out_i)[p] ≈ A⁺y[p] + bias[p]     (for null-space pixels p)
  std(x_out_i)[p]  ≈ spread[p]

  actual_error[p]  = |mean(x_out_i)[p] − x_gt[p]|
                   = |(I−A⁺A)·bias|[p]    (for null-space pixels)
  predicted_std[p] = spread[p]

Well-calibrated:   spread ∝ |bias|   (uncertainty predicts error)
Miscalibrated:     spread inversely related to |bias|

When x_gt = 0 (the default), actual_error = |bias| exactly (large-N limit).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from operators.base import Operator
from engine.oracle import OracleEngine
from layer.decompose import rectify


@dataclass
class NullRegion:
    """Specification for one null-space region in a synthetic case.

    Parameters
    ----------
    mask : ndarray, bool
        True for every pixel this region covers.  Should lie within the
        null space of the operator (otherwise the oracle discards the
        null content in those pixels via the (I−A⁺A) projection, and
        bias / spread have no effect on the result).
    bias : float
        The oracle's mean guess ADDED TO x_gt for pixels in this region.
        Controls actual_error: with x_gt=0 and large ensemble,
        actual_error ≈ |bias| at every pixel in the region.
    spread : float
        Std of the per-member perturbation (zero-mean Gaussian) around
        (x_gt + bias).  Controls predicted_std: with large ensemble,
        predicted_std ≈ spread at every pixel in the region.
    """
    mask: np.ndarray
    bias: float
    spread: float


@dataclass
class SyntheticCase:
    """A fully controlled test case for calibration evaluation.

    Attributes
    ----------
    x_gt : ndarray
        Ground-truth image.  R4: used ONLY in calibrate() for evaluation,
        never passed to the oracle or to any layer function.
    y : ndarray
        Degraded measurement y = A·x_gt.
    x_outs : list of ndarray
        Rectified ensemble {x_out_i}.  Each member satisfies ‖A·x_out_i−y‖≤ε.
        Produced by oracle → rectify pipeline.
    regions : list of NullRegion
        The region specifications that produced this case.  Stored for
        test introspection; not consumed by calibrate().
    """
    x_gt: np.ndarray
    y: np.ndarray
    x_outs: list[np.ndarray]
    regions: list[NullRegion] = field(default_factory=list)


def make_null_calibration_case(
    operator: Operator,
    x_shape: tuple[int, ...],
    regions: list[NullRegion],
    n_ensemble: int,
    seed: int,
) -> SyntheticCase:
    """Generate a synthetic calibration case with per-region null control.

    Parameters
    ----------
    operator : Operator
        The known linear degradation operator A.
    x_shape : tuple of int
        Shape of the high-quality image x (e.g. (H, W) or (C, H, W)).
    regions : list of NullRegion
        Null-space regions with independent bias and spread.  Regions may
        overlap; later regions overwrite earlier ones at shared pixels.
    n_ensemble : int
        Number of ensemble members to generate.
    seed : int
        Random seed for reproducibility (R5).

    Returns
    -------
    SyntheticCase
        x_gt, y, rectified ensemble x_outs.

    Notes
    -----
    x_gt is set to zeros by default — this makes actual_error = |bias|
    analytically, independent of any range-component contribution, giving
    the cleanest possible calibration signal.

    For each ensemble member i and each pixel p in region r's mask:
        null_content_i[p] = region_r.bias + region_r.spread * ε_i[p]
    where ε_i[p] ~ N(0, 1).  After oracle.restore():
        x_out_i = A⁺y + (I−A⁺A)·null_content_i
    The range part A⁺y = A⁺(A·0) = 0 for x_gt=0.
    """
    rng = np.random.default_rng(seed)

    x_gt = np.zeros(x_shape, dtype=np.float64)
    y = operator.forward(x_gt)

    oracle = OracleEngine(operator)
    x_outs: list[np.ndarray] = []

    for _ in range(n_ensemble):
        null_content = np.zeros(x_shape, dtype=np.float64)
        for region in regions:
            noise = rng.standard_normal(x_shape)
            null_content[region.mask] = region.bias + region.spread * noise[region.mask]

        x_hat = oracle.restore(y, null_content)
        result = rectify(x_hat, y, operator)
        x_outs.append(result.x_out)

    return SyntheticCase(x_gt=x_gt, y=y, x_outs=x_outs, regions=regions)
