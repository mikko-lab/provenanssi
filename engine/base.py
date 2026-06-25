"""
Engine ABC for the provenanssi provenance layer.

An Engine wraps a (possibly stochastic) image restorer behind a fixed
interface.  The layer never calls the restorer directly — it always goes
through Engine.restore() or Engine.ensemble(), then through rectify() to
enforce data consistency (R3).

Interface contract
------------------
restore(y) -> x̂
    One restoration.  MAY be stochastic (different result each call).
    Must return an ndarray the same shape as the operator's x-domain.

ensemble(y, n) -> [x̂_1, …, x̂_n]
    n stochastic restorations of the same y.  The caller rectifies each
    member; the Engine is not responsible for data consistency.

What the Engine does NOT do
---------------------------
- Enforce ‖A·x̂ − y‖ ≤ ε.  That is rectify()'s job (R3).
- Use x_gt.  Ground truth is evaluation-only (R4).
- Choose the provenance label.  That is classify()'s job (R2).
"""

from abc import ABC, abstractmethod
import numpy as np


class Engine(ABC):
    """Abstract base for a restorative model."""

    @abstractmethod
    def restore(self, y: np.ndarray) -> np.ndarray:
        """Produce one (possibly stochastic) restoration x̂ from measurement y.

        Parameters
        ----------
        y : ndarray
            Degraded measurement, shape matching the operator's forward() output.

        Returns
        -------
        x̂ : ndarray, shape matching the operator's x-domain
            Raw restoration before rectification.  Data consistency is not
            guaranteed here — call rectify() after.
        """

    @abstractmethod
    def ensemble(self, y: np.ndarray, n: int) -> list[np.ndarray]:
        """Produce n stochastic restorations of y.

        Parameters
        ----------
        y : ndarray
            Degraded measurement.
        n : int
            Number of ensemble members.  Must be ≥ 2 for ensemble_stats()
            to work, but the Engine itself does not enforce this.

        Returns
        -------
        list of ndarray, length n
            Raw restorations {x̂_i}.  Rectify each before passing to
            ensemble_stats() or classify().
        """
