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

Engine ABC conformance (engine/base.py)
---------------------------------------
restore(y)          generates a random null element from the internal RNG
                    when called without a null_content argument.  Calls
                    that supply null_content explicitly still work as before
                    (backward-compatible optional parameter).

ensemble(y, n_or_contents)
                    accepts EITHER an int n (generate n random members) OR
                    a list of null_content arrays (original interface, used
                    by all existing layer tests).
"""

import numpy as np
from typing import Sequence
from operators.base import Operator
from engine.base import Engine


class OracleEngine(Engine):
    """Fake restorer for testing the provenance layer.

    Parameters
    ----------
    operator : Operator
        The known linear degradation operator A.
    null_sigma : float
        Standard deviation of the random null content generated when
        restore(y) is called WITHOUT a null_content argument.
        Only relevant for the new Engine-ABC interface; all existing
        tests pass null_content explicitly and are unaffected.
        Default: 1.0.
    seed : int
        Random seed for the internal RNG (reproducibility, R5).
        Default: 0.
    """

    def __init__(
        self,
        operator: Operator,
        null_sigma: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.op = operator
        self._null_sigma = null_sigma
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Engine ABC interface

    def restore(self, y: np.ndarray, null_content: np.ndarray | None = None) -> np.ndarray:
        """Produce x̂ = A⁺y + (I − A⁺A)·null_content.

        Parameters
        ----------
        y : ndarray
            Observed degraded measurement (output of forward(x)).
        null_content : ndarray or None
            The desired null-space signal, same shape as x.
            - If provided (existing interface): use exactly this array.
              Only the null-space component (I−A⁺A)·null_content enters
              the result; any range-space energy is discarded.
            - If None (Engine ABC interface): draw random noise
              from N(0, null_sigma²) projected to the null space.
              Pass zeros to get a pure range-space estimate (x̂ = A⁺y).

        Returns
        -------
        x̂ : ndarray, same shape as null_content (or A⁺y if None)
        """
        range_part = self.op.pinv(y)
        if null_content is None:
            null_content = self._rng.standard_normal(range_part.shape) * self._null_sigma
        null_part = null_content.astype(np.float64) - self.op.project(null_content)
        return range_part + null_part

    def ensemble(
        self,
        y: np.ndarray,
        null_contents_or_n: int | Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        """Produce an ensemble of estimates with the same range part.

        Each element x̂_i = A⁺y + (I − A⁺A)·n_i.  The range component
        A⁺y is IDENTICAL across all members; only the null component varies.

        Parameters
        ----------
        y : ndarray
            Observed measurement.
        null_contents_or_n : int OR sequence of ndarray
            - int n: generate n random members using internal RNG
              (Engine ABC interface).
            - list of ndarray: use each as null_content for one member
              (original interface, used by all existing layer tests).

        Returns
        -------
        list of ndarray, one x̂_i per member.
        """
        if isinstance(null_contents_or_n, int):
            return [self.restore(y) for _ in range(null_contents_or_n)]
        return [self.restore(y, n) for n in null_contents_or_n]
