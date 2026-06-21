"""
Circular blur / deblur operator: CircularBlur(kernel, zero_threshold).

Degradation model:  y = A · x = k ★ x   (circular convolution)

Under circular boundary conditions, A is exactly a circulant matrix and
is diagonalised by the 2-D DFT:
  Y(ω) = K(ω) · X(ω)   for all frequency bins ω

where K = fft2(kernel_zero_padded).

Pseudo-inverse in the Fourier domain
-------------------------------------
  K⁺(ω) = 1 / K(ω)   if |K(ω)| > zero_threshold
           0            otherwise

The threshold partitions the spectrum into two disjoint sets:
  • kept bins  (|K| > zero_threshold): range space of A  — "measured"
  • killed bins (|K| ≤ zero_threshold): null space of A — "invented"

Projector:
  (A⁺A)(ω) = K⁺(ω) · K(ω) = 1 if |K(ω)| > zero_threshold, else 0

This is a binary mask in the Fourier domain, implemented directly in
project() rather than as pinv(forward(x)) to avoid two round-trips.

zero_threshold — the measured/null boundary
-------------------------------------------
This parameter MUST be set explicitly by the caller, not buried as a
magic number.  It determines which Fourier bins are declared "measured"
vs "invented."  Two regimes:

  Full-rank kernels (e.g. Gaussian):
      all |K(ω)| > 0 in theory; set threshold below min|K| so no bins
      are killed → project ≈ identity, null space empty.

  Rank-deficient kernels (e.g. L×L box on N×N image with L | N):
      |K| = 0 exactly at N/L harmonic bins per axis; set threshold to
      cleanly separate these zeros from non-zero bins.  The killed bins
      define a non-trivial null space of known dimension.

Shape convention
----------------
x, y : same shape — (C, H, W) or (H, W).
Kernel must be 2-D.  Both spatial dims of x must be ≥ kernel dims.
"""

import numpy as np
from .base import Operator


class CircularBlur(Operator):
    """Circular convolution degradation with known kernel.

    Parameters
    ----------
    kernel : ndarray, shape (kH, kW), dtype float64
        The known blur kernel, assumed to be normalised (sums to 1).
        Must be center-aligned: kernel[kH//2, kW//2] is the peak.
    zero_threshold : float
        Fourier bins with |K(ω)| ≤ zero_threshold are treated as null-space
        (A⁺ maps them to zero; project kills them).  This is the explicit
        measured/null boundary — not a magic number.  Caller must choose it.
    """

    def __init__(self, kernel: np.ndarray, zero_threshold: float):
        if np.ndim(kernel) != 2:
            raise ValueError("kernel must be 2-D")
        if zero_threshold < 0:
            raise ValueError("zero_threshold must be ≥ 0")
        self.kernel = np.asarray(kernel, dtype=np.float64)
        self.zero_threshold = zero_threshold

    # ------------------------------------------------------------------
    # Internals

    def _ensure_3d(self, arr: np.ndarray):
        if arr.ndim == 2:
            return arr[np.newaxis].astype(np.float64), True
        if arr.ndim == 3:
            return arr.astype(np.float64), False
        raise ValueError(f"Expected 2-D or 3-D array, got shape {arr.shape}")

    def _maybe_squeeze(self, arr: np.ndarray, was_2d: bool) -> np.ndarray:
        return arr[0] if was_2d else arr

    def _kernel_fft(self, H: int, W: int) -> np.ndarray:
        """Compute DFT of the kernel zero-padded to (H, W).

        The kernel is shifted so its center lands at (0, 0) before padding,
        giving proper circular convolution without phase offset.
        """
        kH, kW = self.kernel.shape
        if kH > H or kW > W:
            raise ValueError(
                f"Kernel ({kH},{kW}) is larger than image ({H},{W})"
            )
        k = np.zeros((H, W), dtype=np.float64)
        k[:kH, :kW] = self.kernel
        k = np.roll(k, -(kH // 2), axis=0)
        k = np.roll(k, -(kW // 2), axis=1)
        return np.fft.fft2(k)

    def _pinv_filter(self, K: np.ndarray) -> np.ndarray:
        """K⁺(ω): 1/K(ω) where |K| > zero_threshold, else 0."""
        safe = np.where(np.abs(K) > self.zero_threshold, K, 1.0)
        return np.where(np.abs(K) > self.zero_threshold, 1.0 / safe, 0.0)

    def _projection_mask(self, K: np.ndarray) -> np.ndarray:
        """Binary mask: 1 where |K| > zero_threshold, else 0."""
        return (np.abs(K) > self.zero_threshold).astype(np.float64)

    # ------------------------------------------------------------------
    # Operator interface

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply A: circular convolution y = k ★ x."""
        x3, was_2d = self._ensure_3d(x)
        C, H, W = x3.shape
        K = self._kernel_fft(H, W)
        result = np.stack([
            np.fft.ifft2(K * np.fft.fft2(x3[c])).real for c in range(C)
        ])
        return self._maybe_squeeze(result, was_2d)

    def pinv(self, y: np.ndarray) -> np.ndarray:
        """Apply A⁺: Fourier-domain deconvolution.

        K⁺(ω) = 1/K(ω) at kept bins, 0 at null bins.
        Exact linear map — no iterative solver, no regularisation.
        """
        y3, was_2d = self._ensure_3d(y)
        C, H, W = y3.shape
        K = self._kernel_fft(H, W)
        K_pinv = self._pinv_filter(K)
        result = np.stack([
            np.fft.ifft2(K_pinv * np.fft.fft2(y3[c])).real for c in range(C)
        ])
        return self._maybe_squeeze(result, was_2d)

    def project(self, x: np.ndarray) -> np.ndarray:
        """Apply A⁺A: binary mask in Fourier domain.

        Keeps the kept-bin energy, annihilates the null-bin energy.
        Implemented directly (one FFT pair) rather than as pinv(forward(x))
        to avoid double round-trip errors.
        """
        x3, was_2d = self._ensure_3d(x)
        C, H, W = x3.shape
        K = self._kernel_fft(H, W)
        mask = self._projection_mask(K)
        result = np.stack([
            np.fft.ifft2(mask * np.fft.fft2(x3[c])).real for c in range(C)
        ])
        return self._maybe_squeeze(result, was_2d)

    # ------------------------------------------------------------------
    # Diagnostics (used in tests and provenance layer)

    def null_space_dim(self, H: int, W: int) -> int:
        """Number of Fourier bins with |K(ω)| ≤ zero_threshold."""
        return int(np.sum(np.abs(self._kernel_fft(H, W)) <= self.zero_threshold))

    def range_space_dim(self, H: int, W: int) -> int:
        """Number of Fourier bins with |K(ω)| > zero_threshold."""
        return H * W - self.null_space_dim(H, W)
