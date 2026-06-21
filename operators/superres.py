"""
Super-resolution downsampling operator (box / area-average).

Degradation model:  y = A · x
  where A averages each non-overlapping s×s block to one pixel.

Shape convention
----------------
x : ndarray, float64, shape (C, H, W) or (H, W)
y : ndarray, float64, shape (C, H//s, W//s) or (H//s, W//s)

H and W must be integer multiples of the scale factor s.

Pseudo-inverse derivation
-------------------------
As a matrix, A ∈ R^{m × n} (m = HW/s², n = HW) has rows with s² entries
each equal to 1/s² and zeros elsewhere (non-overlapping blocks).

  A · Aᵀ = (1/s²) · I_m

so  A⁺ = Aᵀ · (A Aᵀ)⁻¹ = s² · Aᵀ.

Aᵀ applied to y spreads each y[i,j] to the s×s block as y[i,j] / s²,
so s² · Aᵀ(y) is plain nearest-neighbour upsampling (each pixel repeated).

Consequences verified in tests/test_operators.py (R7):
  A · A⁺ · A  ==  A          (row-space reconstruction)
  (A⁺A)²      ==  A⁺A        (projector idempotence)
"""

import numpy as np
from .base import Operator


class BoxDownsample(Operator):
    """Area-average (box) downsampling by integer scale factor s."""

    def __init__(self, scale: int):
        if scale < 1:
            raise ValueError(f"scale must be ≥ 1, got {scale}")
        self.scale = scale

    # ------------------------------------------------------------------
    # internal helpers — keep shapes consistent

    def _ensure_3d(self, arr: np.ndarray):
        """Return (arr_3d, was_2d)."""
        if arr.ndim == 2:
            return arr[np.newaxis].astype(np.float64), True
        if arr.ndim == 3:
            return arr.astype(np.float64), False
        raise ValueError(f"Expected 2-D or 3-D array, got shape {arr.shape}")

    def _maybe_squeeze(self, arr: np.ndarray, was_2d: bool) -> np.ndarray:
        return arr[0] if was_2d else arr

    # ------------------------------------------------------------------
    # Operator interface

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply A: area-average each s×s block.

        x : (C, H, W) or (H, W)  →  y : (C, H//s, W//s) or (H//s, W//s)
        """
        s = self.scale
        x3, was_2d = self._ensure_3d(x)
        C, H, W = x3.shape
        if H % s != 0 or W % s != 0:
            raise ValueError(
                f"Spatial dims ({H}, {W}) must be divisible by scale={s}"
            )
        # reshape so the block dimensions are explicit, then average
        y = x3.reshape(C, H // s, s, W // s, s).mean(axis=(2, 4))
        return self._maybe_squeeze(y, was_2d)

    def pinv(self, y: np.ndarray) -> np.ndarray:
        """Apply A⁺ = s² · Aᵀ: nearest-neighbour upsample.

        y : (C, H//s, W//s) or (H//s, W//s)  →  x̂ : (C, H, W) or (H, W)

        This is an exact linear map.  Combining with forward() recovers the
        block-mean image: pinv(forward(x)) == project(x).
        """
        s = self.scale
        y3, was_2d = self._ensure_3d(y)
        # repeat along spatial axes — pure linear operation
        x_hat = np.repeat(np.repeat(y3, s, axis=1), s, axis=2)
        return self._maybe_squeeze(x_hat, was_2d)

    def project(self, x: np.ndarray) -> np.ndarray:
        """Apply A⁺A: downsample then upsample (range-space projector).

        The result is a block-constant image where each s×s tile holds
        the mean of the corresponding block in x.  The null-space component
        is  x − project(x)  (zero-mean within each block).
        """
        return self.pinv(self.forward(x))
