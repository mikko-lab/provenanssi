"""
operators/bicubic.py — BicubicDownsample operator.

Degradation model: y = A · x
where A = S_s · C_h: circular convolution with the antialiasing bicubic
kernel h, followed by stride-s subsampling.

Kernel convention
-----------------
Keys cubic with parameter a = -0.5 (MATLAB imresize default).
  h(t) = (a+2)|t|³ - (a+3)|t|² + 1,    0 ≤ |t| < 1
  h(t) = a|t|³ - 5a|t|² + 8a|t| - 4a,  1 ≤ |t| < 2
  h(t) = 0,                              |t| ≥ 2

Scaled for downsampling by s: h_s(i) = h(i/s), i = −2s … +2s.
Support: 4s+1 taps. Normalised to sum=1 (DC preserved).

Note h(±2) = 0 identically for all a, so the outermost taps are zero.

Boundary convention
-------------------
Circular (FFT-based). Differs from MATLAB's reflect padding by
O(4s / N) at the image borders; negligible for N >> 4s.

ResShift alignment
------------------
ResShift was trained with activate_matlab=True, scale=0.25 (x4 down).
MATLAB imresize uses the same Keys cubic a=-0.5 kernel and produces
identical results for interior pixels (boundary treatment differs).

Pseudo-inverse (A⁺) — Fourier aliased inversion
------------------------------------------------
For a separable bicubic operator the subsampled-convolution aliasing
equation in the DFT domain is:

  Y[k_h, k_w] = (1/s²) · Σ_{m₁=0}^{s-1} Σ_{m₂=0}^{s-1}
                  K_h[k_h + m₁·Mh] · K_w[k_w + m₂·Mw]
                  · X[k_h + m₁·Mh, k_w + m₂·Mw]

where K_h, K_w are the length-H, length-W DFTs of the 1D kernel.

The minimum-norm (Moore-Penrose) pseudo-inverse is:

  X̂[k_h + m₁·Mh, k_w + m₂·Mw] =
      s² · K_h*[k_h + m₁·Mh] · K_w*[k_w + m₂·Mw] · Y[k_h, k_w]
      / D[k_h, k_w]

  D[k_h, k_w] = D_h[k_h] · D_w[k_w]
  D_h[k_h]    = Σ_m |K_h[k_h + m·Mh]|²
  D_w[k_w]    = Σ_m |K_w[k_w + m·Mw]|²

When D < zero_threshold: X̂ = 0 (treated as null-space frequency).

This satisfies A·A⁺·A = A and A⁺·A·A⁺ = A⁺ algebraically; the achieved
floating-point tolerance is measured in BICUBIC_TOL (see test_operators.py).

Tolerance vs box/mask/deblur
-----------------------------
BoxDownsample/MaskOperator achieve 1e-12 because their pinv is a
closed-form exact map (no FFT, no division by a denominator).
BicubicDownsample uses four FFT round-trips and a frequency-domain
division, accumulating O(N · ε · log N) error per pass ×10 passes.
For image sizes in the test suite (12×12 to 96×96) the achieved
tolerance is BICUBIC_TOL ≈ 1e-10. Box/mask/deblur tolerances are
unchanged.

Condition number
----------------
  max|A⁺| ≈ s² / min(D_h[k_h] · D_w[k_w])
           ≈ s² / min|K_h|² for a well-designed antialiasing filter
  For s=4, a=-0.5: typically ≤ 50 (well-conditioned in the passband).
  See TestBicubicDownsampleAnalytical.test_condition_number.
"""

from __future__ import annotations
import numpy as np
from operators.base import Operator


# ---------------------------------------------------------------------------
# Kernel construction

def _keys_cubic_1d(scale: int, a: float = -0.5) -> np.ndarray:
    """1D Keys cubic antialiasing kernel for downsampling by `scale`.

    Keys cubic with parameter a (a=-0.5 = MATLAB convention):
      h(t) = (a+2)|t|³ - (a+3)|t|² + 1,    0 ≤ |t| < 1
      h(t) = a|t|³ - 5a|t|² + 8a|t| - 4a,  1 ≤ |t| < 2
      h(t) = 0,                              |t| ≥ 2

    Scaled version: h_s(i) = h(i/scale) for integer i in [-2s, ..., 2s].
    Total length: 4·scale + 1 taps.
    Normalised so that the taps sum to 1 (DC gain = 1).
    """
    n_half = 2 * scale
    t_int = np.arange(-n_half, n_half + 1)
    t = t_int.astype(np.float64) / scale
    at = np.abs(t)
    k = np.where(
        at < 1,
        (a + 2) * at**3 - (a + 3) * at**2 + 1,
        np.where(
            at < 2,
            a * at**3 - 5 * a * at**2 + 8 * a * at - 4 * a,
            0.0,
        ),
    )
    k /= k.sum()
    return k


# ---------------------------------------------------------------------------
# Operator

class BicubicDownsample(Operator):
    """Bicubic downsampling operator: antialiasing circular conv + stride-s subsample.

    Matches ResShift's training degradation (bicubic x4, a=-0.5).

    Parameters
    ----------
    scale : int
        Downsampling factor s.  forward(x) has spatial size H/s × W/s.
    a : float
        Keys cubic parameter.  a=-0.5 matches MATLAB imresize (default).
    zero_threshold : float
        D[k] values below this are treated as null space; pinv returns 0
        at those HR frequencies.  For the bicubic antialiasing kernel D is
        never near zero in the passband, so this threshold is rarely active.
    """

    def __init__(
        self,
        scale: int,
        a: float = -0.5,
        zero_threshold: float = 1e-8,
    ) -> None:
        if scale < 1:
            raise ValueError(f"scale must be ≥ 1, got {scale}")
        self.scale = scale
        self.a = a
        self.zero_threshold = zero_threshold
        self._kernel_1d = _keys_cubic_1d(scale, a)

    # ------------------------------------------------------------------
    # Internal helpers

    def _kernel_fft(self, n: int) -> np.ndarray:
        """Complex DFT (length n) of the centred bicubic kernel.

        The kernel is placed at position 0 (centre tap) with negative
        offsets wrapped to the end, matching the standard convention for
        circular convolution via FFT.
        """
        k = self._kernel_1d
        n_half = len(k) // 2          # index of the centre tap
        k_padded = np.zeros(n, dtype=np.float64)
        for i, v in enumerate(k):
            k_padded[(i - n_half) % n] += v   # circular placement (wraps for n < len(k))
        return np.fft.fft(k_padded)

    @staticmethod
    def _ensure_3d(arr: np.ndarray) -> tuple[np.ndarray, bool]:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 2:
            return arr[np.newaxis], True
        if arr.ndim == 3:
            return arr, False
        raise ValueError(f"Expected 2-D or 3-D input, got {arr.ndim}-D")

    @staticmethod
    def _maybe_squeeze(arr: np.ndarray, was_2d: bool) -> np.ndarray:
        return arr[0] if was_2d else arr

    # ------------------------------------------------------------------
    # Operator ABC

    def forward(self, x: np.ndarray) -> np.ndarray:
        """y = A·x: bicubic antialiasing circular conv then stride-s subsample.

        Shape: (C, H, W) or (H, W) → (C, H/s, W/s) or (H/s, W/s).
        H and W must be divisible by scale.
        """
        s = self.scale
        x3, was_2d = self._ensure_3d(x)
        C, H, W = x3.shape
        if H % s != 0 or W % s != 0:
            raise ValueError(
                f"Spatial dims ({H}, {W}) must be divisible by scale={s}"
            )
        K_h = self._kernel_fft(H)                   # (H,) complex
        K_w = self._kernel_fft(W)                   # (W,) complex
        K_2d = K_h[:, None] * K_w[None, :]          # (H, W) separable 2D filter

        y_ch = []
        for c in range(C):
            X = np.fft.fft2(x3[c])
            conv = np.fft.ifft2(X * K_2d).real      # circular 2D convolution
            y_ch.append(conv[::s, ::s])              # stride-s subsampling

        return self._maybe_squeeze(np.stack(y_ch), was_2d)

    def pinv(self, y: np.ndarray) -> np.ndarray:
        """x̂ = A⁺·y: Fourier-domain aliased pseudo-inverse.

        Inverts the subsampled-convolution aliasing equation exactly (up to
        float64 round-off).  See module docstring for the formula.
        """
        s = self.scale
        y3, was_2d = self._ensure_3d(y)
        C, Mh, Mw = y3.shape
        H, W = Mh * s, Mw * s

        K_h = self._kernel_fft(H)                   # (H,) complex
        K_w = self._kernel_fft(W)                   # (W,) complex

        # Reshape to alias groups: K_h_rs[m, k] = K_h[m·Mh + k]
        K_h_rs = K_h.reshape(s, Mh)                 # (s, Mh)
        K_w_rs = K_w.reshape(s, Mw)                 # (s, Mw)

        # Aliased energy denominators
        D_h = (np.abs(K_h_rs) ** 2).sum(axis=0)     # (Mh,)
        D_w = (np.abs(K_w_rs) ** 2).sum(axis=0)     # (Mw,)
        D = D_h[:, None] * D_w[None, :]             # (Mh, Mw)

        D_safe = np.where(D > self.zero_threshold, D, 1.0)
        active = D > self.zero_threshold             # (Mh, Mw) bool

        x_ch = []
        for c in range(C):
            Y = np.fft.fft2(y3[c])                  # (Mh, Mw) complex
            # Y / D, zero where inactive (null-space frequencies)
            Y_over_D = np.where(active, Y / D_safe, 0.0)

            X_pinv = np.zeros((H, W), dtype=complex)
            for m1 in range(s):
                for m2 in range(s):
                    # HR block [m1·Mh:(m1+1)·Mh, m2·Mw:(m2+1)·Mw]
                    X_pinv[m1 * Mh:(m1 + 1) * Mh,
                           m2 * Mw:(m2 + 1) * Mw] = (
                        s ** 2
                        * np.conj(K_h_rs[m1, :, None])
                        * np.conj(K_w_rs[m2, None, :])
                        * Y_over_D
                    )
            x_ch.append(np.fft.ifft2(X_pinv).real)

        return self._maybe_squeeze(np.stack(x_ch), was_2d)

    def project(self, x: np.ndarray) -> np.ndarray:
        """A⁺A·x: orthogonal projection onto the row space of A."""
        return self.pinv(self.forward(x))
