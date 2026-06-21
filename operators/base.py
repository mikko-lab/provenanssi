"""
Operator ABC for the provenanssi provenance layer.

All operators represent a known linear degradation  y = A · x  where:
  - x  is the unknown high-quality image, shape (C, H, W) or (H, W)
  - y  is the observed (degraded) measurement, shape determined by A

Shape contract
--------------
forward(x)  : x ∈ R^n  →  y ∈ R^m   (apply A)
pinv(y)     : y ∈ R^m  →  x ∈ R^n   (apply Moore-Penrose A⁺)
project(x)  : x ∈ R^n  →  x ∈ R^n   (apply A⁺A, the range-space projector)

Linearity requirement (R7)
--------------------------
All three maps MUST be exact linear maps — no approximations, no learned
components, no thresholding, no clipping beyond float precision.
The provenance science depends on:
  A · A⁺ · A  ==  A          (pseudo-inverse property)
  (A⁺A)²      ==  A⁺A        (projector idempotence)
These are verified per-operator by tests/test_operators.py.

Input/output arrays are NumPy ndarrays.  PyTorch tensors may be used
internally but the public interface takes and returns ndarray.
"""

from abc import ABC, abstractmethod
import numpy as np


class Operator(ABC):
    """Abstract base for a known linear degradation operator A."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the degradation: y = A · x.

        Parameters
        ----------
        x : ndarray, shape (C, H, W) or (H, W)
            High-quality signal in the image domain.

        Returns
        -------
        y : ndarray, shape determined by the operator
            Degraded observation.
        """

    @abstractmethod
    def pinv(self, y: np.ndarray) -> np.ndarray:
        """Apply the Moore-Penrose pseudo-inverse: x̂ = A⁺ · y.

        For the noiseless case this satisfies  A · A⁺ · y = y  when y is
        in the range of A.  Must be an exact linear map — no clamping.

        Parameters
        ----------
        y : ndarray, shape matching forward() output
            Degraded observation.

        Returns
        -------
        x̂ : ndarray, same shape as x
        """

    @abstractmethod
    def project(self, x: np.ndarray) -> np.ndarray:
        """Apply the range-space projector: A⁺A · x.

        This is the orthogonal projection onto the row space of A.
        The null-space complement is  (I − A⁺A) · x = x − project(x).

        Must satisfy  project(project(x)) == project(x)  to numerical
        precision (idempotence).  Equivalent to  pinv(forward(x)).

        Parameters
        ----------
        x : ndarray, same shape as forward() input

        Returns
        -------
        projected : ndarray, same shape as x
        """
