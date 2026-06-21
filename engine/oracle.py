"""
OracleEngine — a fake restorer with zero learning, used to test the layer.

Purpose
-------
When testing the provenance layer we need a "model" whose outputs we
control exactly.  OracleEngine takes a CALLER-SUPPLIED null-space content
and returns:

    x̂ = A⁺y + (I − A⁺A)·n

where:
  A⁺y        — range component, determined entirely by the measurement y.
  (I−A⁺A)·n  — null component, the "invented" part.  Callers choose n.

This lets tests verify the layer against exact ground truth:
  - range part is always exactly A⁺y (same for all ensemble members)
  - null part is exactly (I−A⁺A)·n (caller controls what gets invented)
  - with n = 0 the output is purely in the range space — nothing invented

OracleEngine is NOT a model and is NOT used in production.
Its only role is to give the layer tests a ground-truth oracle.
"""

import numpy as np
from typing import Sequence
from operators.base import Operator


class OracleEngine:
    """Fake restorer for testing the provenance layer.

    Parameters
    ----------
    operator : Operator
        The known linear degradation operator A for which this engine
        provides "restored" estimates.
    """

    def __init__(self, operator: Operator):
        self.op = operator

    def restore(self, y: np.ndarray, null_content: np.ndarray) -> np.ndarray:
        """Produce a controlled estimate x̂ = A⁺y + (I − A⁺A)·null_content.

        Parameters
        ----------
        y : ndarray
            Observed degraded measurement (output of forward(x)).
        null_content : ndarray, same shape as x
            The desired null-space signal.  Only the null-space component
            (I−A⁺A)·null_content enters the result; any range-space energy
            in null_content is discarded by the projection.  Pass zeros to
            get a pure range-space estimate (x̂ = A⁺y).

        Returns
        -------
        x̂ : ndarray, same shape as null_content
            Range part: A⁺y — identical for all calls with the same y.
            Null part:  (I−A⁺A)·null_content — caller-controlled.
        """
        range_part = self.op.pinv(y)
        null_part = null_content.astype(np.float64) - self.op.project(null_content)
        return range_part + null_part

    def ensemble(
        self,
        y: np.ndarray,
        null_contents: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        """Produce an ensemble of estimates with the same range part.

        Each element x̂_i = A⁺y + (I − A⁺A)·n_i.  The range component
        A⁺y is IDENTICAL across all members; only the null component varies.

        Parameters
        ----------
        y : ndarray
            Observed measurement.
        null_contents : sequence of ndarray
            One null_content array per ensemble member.

        Returns
        -------
        list of ndarray, one x̂_i per null_content.
        """
        return [self.restore(y, n) for n in null_contents]
