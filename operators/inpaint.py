"""
Inpainting operator: MaskOperator(mask).

Degradation model:  y = A · x = mask ⊙ x   (elementwise binary mask)

This operator is its own pseudo-inverse and its own projector:
  A = A⁺ = A⁺A = M   (0/1 diagonal, idempotent, symmetric)

Proof: for D = diag(d_1,...,d_n) with d_i ∈ {0,1}:
  D⁺ = diag(1/d_i if d_i≠0, else 0) = diag(d_i) = D.
  D⁺D = D² = D.  All four Moore-Penrose conditions are satisfied exactly.

Shape convention
----------------
x, y : same shape — (C, H, W) or (H, W).
full HxW output is preserved (holes are zeroed, not compacted) so the
range projector A⁺A keeps its spatial interpretation.

Null space
----------
null(A) = {x : mask ⊙ x = 0} = signals supported only on the hole region.
These are the pixels that inpainting must "invent" — A is blind to them.
The null-space component of any x̂ is (1 − mask) ⊙ x̂.
"""

import numpy as np
from .base import Operator


class MaskOperator(Operator):
    """Masking operator for inpainting: keeps observed pixels, zeros holes.

    Parameters
    ----------
    mask : ndarray, dtype bool or float, shape (H, W) or (C, H, W)
        1 / True  = observed pixel (kept).
        0 / False = hole to inpaint (zeroed and invented by the prior).

    zero_threshold is not needed here: the operator has no ambiguous
    spectral boundary — every pixel is either exactly observed or exactly
    a hole.
    """

    def __init__(self, mask: np.ndarray):
        self.mask = mask.astype(np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply A: y = mask ⊙ x.  Zeros the hole pixels."""
        return x.astype(np.float64) * self.mask

    def pinv(self, y: np.ndarray) -> np.ndarray:
        """Apply A⁺ = A: mask ⊙ y.

        Since A is a symmetric idempotent diagonal, A⁺ = A.
        """
        return y.astype(np.float64) * self.mask

    def project(self, x: np.ndarray) -> np.ndarray:
        """Apply A⁺A = A: project onto the observed region.

        Identical to forward() and pinv() for this operator.
        The null-space complement is  (I − A⁺A)·x = (1 − mask) ⊙ x.
        """
        return x.astype(np.float64) * self.mask
