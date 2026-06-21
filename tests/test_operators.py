"""
R7 operator tests: A·A⁺·A = A  and  (A⁺A)² = A⁺A

Tolerance policy
----------------
TOL = 1e-12 (absolute).  This is our declared precision floor.
If a test fails and a temptation arises to raise TOL to make it pass,
STOP — diagnose the implementation instead.  The science rests on these
being correct linear maps, not approximately-correct ones.

All test arrays are constructed with *known structure* so failures have
an interpretable cause, not just "random noise mismatch."
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from operators.superres import BoxDownsample
from operators.inpaint import MaskOperator
from operators.deblur import CircularBlur

TOL = 1e-12


# ---------------------------------------------------------------------------
# Fixtures / helpers

def block_constant(s: int, h: int, w: int, c: int = None) -> np.ndarray:
    """Array that is constant within each s×s block.

    Lives entirely in the range space of BoxDownsample(s).
    project() must return it unchanged.
    """
    rng = np.random.default_rng(0)
    bh, bw = h // s, w // s
    if c is None:
        blocks = rng.standard_normal((bh, bw))
        # repeat to fill blocks
        return np.repeat(np.repeat(blocks, s, axis=0), s, axis=1)
    else:
        blocks = rng.standard_normal((c, bh, bw))
        return np.repeat(np.repeat(blocks, s, axis=1), s, axis=2)


def null_space_only(s: int, h: int, w: int) -> np.ndarray:
    """Array with zero mean in every s×s block — lives in null space.

    forward() must map it to zero.  project() must map it to zero.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal((h, w))
    # subtract block mean from each block so all block means are zero
    bh, bw = h // s, w // s
    x_bc = x.reshape(bh, s, bw, s)
    means = x_bc.mean(axis=(1, 3), keepdims=True)
    x_bc = x_bc - means
    return x_bc.reshape(h, w)


def general_array(h: int, w: int, c: int = None, seed: int = 42) -> np.ndarray:
    """General-purpose float64 array with values in a moderate range."""
    rng = np.random.default_rng(seed)
    shape = (h, w) if c is None else (c, h, w)
    return rng.standard_normal(shape)


# ---------------------------------------------------------------------------
# Parametrise over scale factors and spatial sizes

SCALES = [2, 3, 4]
# (H, W) pairs — all divisible by lcm(2, 3, 4) = 12
HW_PAIRS = [(12, 12), (12, 24), (24, 12)]


@pytest.mark.parametrize("s", SCALES)
@pytest.mark.parametrize("H,W", HW_PAIRS)
class TestBoxDownsample:

    # ------------------------------------------------------------------
    # Core operator-math properties (R7)

    def test_pseudo_inverse_property_2d(self, s, H, W):
        """A · A⁺ · A == A  (2-D input)."""
        op = BoxDownsample(s)
        x = general_array(H, W)
        lhs = op.forward(op.pinv(op.forward(x)))
        rhs = op.forward(x)
        np.testing.assert_allclose(lhs, rhs, atol=TOL,
            err_msg=f"A·A⁺·A ≠ A  (scale={s}, shape=({H},{W}))")

    def test_pseudo_inverse_property_3d(self, s, H, W):
        """A · A⁺ · A == A  (3-D / RGB input)."""
        op = BoxDownsample(s)
        x = general_array(H, W, c=3)
        lhs = op.forward(op.pinv(op.forward(x)))
        rhs = op.forward(x)
        np.testing.assert_allclose(lhs, rhs, atol=TOL,
            err_msg=f"A·A⁺·A ≠ A  (scale={s}, shape=(3,{H},{W}))")

    def test_projector_idempotence_2d(self, s, H, W):
        """(A⁺A)² == A⁺A  (2-D input)."""
        op = BoxDownsample(s)
        x = general_array(H, W)
        Px = op.project(x)
        PPx = op.project(Px)
        np.testing.assert_allclose(PPx, Px, atol=TOL,
            err_msg=f"(A⁺A)² ≠ A⁺A  (scale={s}, shape=({H},{W}))")

    def test_projector_idempotence_3d(self, s, H, W):
        """(A⁺A)² == A⁺A  (3-D / RGB input)."""
        op = BoxDownsample(s)
        x = general_array(H, W, c=3)
        Px = op.project(x)
        PPx = op.project(Px)
        np.testing.assert_allclose(PPx, Px, atol=TOL,
            err_msg=f"(A⁺A)² ≠ A⁺A  (scale={s}, shape=(3,{H},{W}))")

    def test_project_equals_pinv_of_forward(self, s, H, W):
        """project(x) == pinv(forward(x))  — interface consistency."""
        op = BoxDownsample(s)
        x = general_array(H, W)
        np.testing.assert_allclose(
            op.project(x), op.pinv(op.forward(x)), atol=TOL,
            err_msg=f"project ≠ pinv∘forward  (scale={s}, shape=({H},{W}))")

    # ------------------------------------------------------------------
    # Linearity — catches clipping, rounding, or other non-linearities

    def test_forward_is_linear(self, s, H, W):
        """A(α·x + β·z) == α·A(x) + β·A(z)."""
        op = BoxDownsample(s)
        x = general_array(H, W, seed=10)
        z = general_array(H, W, seed=11)
        alpha, beta = 3.7, -2.1
        lhs = op.forward(alpha * x + beta * z)
        rhs = alpha * op.forward(x) + beta * op.forward(z)
        np.testing.assert_allclose(lhs, rhs, atol=TOL,
            err_msg=f"forward not linear  (scale={s})")

    def test_pinv_is_linear(self, s, H, W):
        """A⁺(α·y + β·w) == α·A⁺(y) + β·A⁺(w)."""
        op = BoxDownsample(s)
        y1 = general_array(H // s, W // s, seed=20)
        y2 = general_array(H // s, W // s, seed=21)
        alpha, beta = -1.5, 4.2
        lhs = op.pinv(alpha * y1 + beta * y2)
        rhs = alpha * op.pinv(y1) + beta * op.pinv(y2)
        np.testing.assert_allclose(lhs, rhs, atol=TOL,
            err_msg=f"pinv not linear  (scale={s})")

    # ------------------------------------------------------------------
    # Range-space / null-space structure

    def test_range_space_is_fixed_by_projector(self, s, H, W):
        """For x in range(A):  project(x) == x  (projector is identity on range)."""
        op = BoxDownsample(s)
        x_range = block_constant(s, H, W)
        np.testing.assert_allclose(op.project(x_range), x_range, atol=TOL,
            err_msg=f"projector does not fix range space  (scale={s})")

    def test_null_space_maps_to_zero_under_forward(self, s, H, W):
        """For x in null(A):  forward(x) == 0."""
        op = BoxDownsample(s)
        x_null = null_space_only(s, H, W)
        y = op.forward(x_null)
        np.testing.assert_allclose(y, 0.0, atol=TOL,
            err_msg=f"null-space element not annihilated by forward  (scale={s})")

    def test_null_space_maps_to_zero_under_project(self, s, H, W):
        """For x in null(A):  project(x) == 0."""
        op = BoxDownsample(s)
        x_null = null_space_only(s, H, W)
        Px = op.project(x_null)
        np.testing.assert_allclose(Px, 0.0, atol=TOL,
            err_msg=f"projector does not annihilate null space  (scale={s})")

    def test_range_null_decomposition_partitions_x(self, s, H, W):
        """x == project(x) + (x - project(x))  — tautology but verifies dtype stability."""
        op = BoxDownsample(s)
        x = general_array(H, W)
        Px = op.project(x)
        null_component = x - Px
        # range and null components are orthogonal: dot product ~ 0
        inner = np.sum(Px * null_component)
        norm_product = np.linalg.norm(Px) * np.linalg.norm(null_component)
        # relative orthogonality (skip if either component is near-zero)
        if norm_product > 1e-10:
            assert abs(inner) / norm_product < TOL, (
                f"range and null components not orthogonal  "
                f"(scale={s}, relative inner={abs(inner)/norm_product:.2e})")


# ---------------------------------------------------------------------------
# Analytical / known-value tests (not parametrised — small, readable)

class TestBoxDownsampleAnalytical:
    """Verify specific numeric results against manually computed answers."""

    def test_s2_forward_known_blocks(self):
        """2×2 blocks: forward output is the block mean."""
        op = BoxDownsample(2)
        x = np.array([
            [1., 3., 5., 7.],
            [3., 1., 7., 5.],
            [2., 4., 6., 8.],
            [4., 2., 8., 6.],
        ], dtype=np.float64)
        # block (0,0): mean([1,3,3,1]) = 2; block (0,1): mean([5,7,7,5]) = 6
        # block (1,0): mean([2,4,4,2]) = 3; block (1,1): mean([6,8,8,6]) = 7
        expected = np.array([[2., 6.], [3., 7.]], dtype=np.float64)
        np.testing.assert_allclose(op.forward(x), expected, atol=TOL)

    def test_s2_pinv_is_nn_upsample(self):
        """pinv is nearest-neighbour upsample (each low-res pixel → 2×2 tile)."""
        op = BoxDownsample(2)
        y = np.array([[1., 2.], [3., 4.]], dtype=np.float64)
        expected = np.array([
            [1., 1., 2., 2.],
            [1., 1., 2., 2.],
            [3., 3., 4., 4.],
            [3., 3., 4., 4.],
        ], dtype=np.float64)
        np.testing.assert_allclose(op.pinv(y), expected, atol=TOL)

    def test_s2_project_block_constant_is_identity(self):
        """Block-constant image: project is identity."""
        op = BoxDownsample(2)
        x = np.array([
            [5., 5., 9., 9.],
            [5., 5., 9., 9.],
            [3., 3., 7., 7.],
            [3., 3., 7., 7.],
        ], dtype=np.float64)
        np.testing.assert_allclose(op.project(x), x, atol=TOL)

    def test_s2_project_extracts_block_means(self):
        """project replaces each block by its mean."""
        op = BoxDownsample(2)
        x = np.array([
            [0., 4., 0., 4.],
            [4., 0., 4., 0.],
            [1., 3., 2., 6.],
            [3., 1., 6., 2.],
        ], dtype=np.float64)
        # block means: (0,0)=2, (0,1)=2, (1,0)=2, (1,1)=4
        expected = np.array([
            [2., 2., 2., 2.],
            [2., 2., 2., 2.],
            [2., 2., 4., 4.],
            [2., 2., 4., 4.],
        ], dtype=np.float64)
        np.testing.assert_allclose(op.project(x), expected, atol=TOL)

    def test_s2_null_space_element(self):
        """Element known to be in null space: forward → 0, project → 0."""
        op = BoxDownsample(2)
        # within each 2×2 block: [+1,-1; -1,+1] — zero mean
        x = np.array([
            [+1., -1., +1., -1.],
            [-1., +1., -1., +1.],
            [+1., -1., +1., -1.],
            [-1., +1., -1., +1.],
        ], dtype=np.float64)
        np.testing.assert_allclose(op.forward(x), 0.0, atol=TOL)
        np.testing.assert_allclose(op.project(x), 0.0, atol=TOL)

    def test_s3_forward_known(self):
        """Scale 3: each 3×3 block averages to one value."""
        op = BoxDownsample(3)
        # 6×6 image: top-left 3×3 block all 9s, top-right 3×3 all 0s,
        # bottom-left all 3s, bottom-right all 6s
        x = np.zeros((6, 6), dtype=np.float64)
        x[:3, :3] = 9.0
        x[:3, 3:] = 0.0
        x[3:, :3] = 3.0
        x[3:, 3:] = 6.0
        expected = np.array([[9., 0.], [3., 6.]], dtype=np.float64)
        np.testing.assert_allclose(op.forward(x), expected, atol=TOL)

    def test_shape_error_on_non_divisible(self):
        """forward raises ValueError when dims are not divisible by scale."""
        op = BoxDownsample(3)
        x = np.ones((5, 6), dtype=np.float64)
        with pytest.raises(ValueError, match="divisible"):
            op.forward(x)

    def test_scale_1_is_identity(self):
        """Scale-1 operator is identity on all three maps."""
        op = BoxDownsample(1)
        x = general_array(8, 8)
        np.testing.assert_allclose(op.forward(x), x, atol=TOL)
        np.testing.assert_allclose(op.pinv(x), x, atol=TOL)
        np.testing.assert_allclose(op.project(x), x, atol=TOL)


# ===========================================================================
# MaskOperator tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Mask fixtures

def random_mask(H: int, W: int, fraction_observed: float = 0.6, seed: int = 99) -> np.ndarray:
    """Random binary mask with ~fraction_observed pixels set to 1."""
    rng = np.random.default_rng(seed)
    return (rng.random((H, W)) < fraction_observed).astype(np.float64)


def center_hole_mask(H: int, W: int) -> np.ndarray:
    """Mask with a rectangular hole in the centre."""
    m = np.ones((H, W), dtype=np.float64)
    m[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 0.0
    return m


def mask_null_element(mask: np.ndarray) -> np.ndarray:
    """Signal in null(A): nonzero only where mask=0 (the holes)."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(mask.shape)
    return x * (1.0 - mask)


def mask_range_element(mask: np.ndarray) -> np.ndarray:
    """Signal in range(A): nonzero only where mask=1 (observed)."""
    rng = np.random.default_rng(6)
    x = rng.standard_normal(mask.shape)
    return x * mask


MASKS = [
    random_mask(8, 8),
    center_hole_mask(12, 16),
    random_mask(6, 10, fraction_observed=0.3),
]


@pytest.mark.parametrize("mask", MASKS, ids=["random60", "center_hole", "sparse30"])
class TestMaskOperator:

    def test_pseudo_inverse_property(self, mask):
        """A · A⁺ · A == A."""
        op = MaskOperator(mask)
        x = general_array(*mask.shape[-2:])
        lhs = op.forward(op.pinv(op.forward(x)))
        rhs = op.forward(x)
        np.testing.assert_allclose(lhs, rhs, atol=TOL,
            err_msg="A·A⁺·A ≠ A")

    def test_projector_idempotence(self, mask):
        """(A⁺A)² == A⁺A."""
        op = MaskOperator(mask)
        x = general_array(*mask.shape[-2:])
        Px = op.project(x)
        np.testing.assert_allclose(op.project(Px), Px, atol=TOL,
            err_msg="(A⁺A)² ≠ A⁺A")

    def test_project_equals_pinv_of_forward(self, mask):
        """project(x) == pinv(forward(x))."""
        op = MaskOperator(mask)
        x = general_array(*mask.shape[-2:])
        np.testing.assert_allclose(
            op.project(x), op.pinv(op.forward(x)), atol=TOL)

    def test_forward_equals_project(self, mask):
        """forward == pinv == project for mask operator (all are M⊙·)."""
        op = MaskOperator(mask)
        x = general_array(*mask.shape[-2:])
        np.testing.assert_allclose(op.forward(x), op.project(x), atol=TOL)
        np.testing.assert_allclose(op.pinv(x), op.project(x), atol=TOL)

    def test_forward_is_linear(self, mask):
        """A(α·x + β·z) == α·A(x) + β·A(z)."""
        op = MaskOperator(mask)
        H, W = mask.shape[-2:]
        x = general_array(H, W, seed=10)
        z = general_array(H, W, seed=11)
        alpha, beta = 3.7, -2.1
        lhs = op.forward(alpha * x + beta * z)
        rhs = alpha * op.forward(x) + beta * op.forward(z)
        np.testing.assert_allclose(lhs, rhs, atol=TOL)

    def test_pinv_is_linear(self, mask):
        """A⁺(α·y + β·w) == α·A⁺(y) + β·A⁺(w)."""
        op = MaskOperator(mask)
        H, W = mask.shape[-2:]
        y1 = general_array(H, W, seed=20)
        y2 = general_array(H, W, seed=21)
        alpha, beta = -1.5, 4.2
        lhs = op.pinv(alpha * y1 + beta * y2)
        rhs = alpha * op.pinv(y1) + beta * op.pinv(y2)
        np.testing.assert_allclose(lhs, rhs, atol=TOL)

    def test_range_space_fixed_by_projector(self, mask):
        """x in range(A) → project(x) == x."""
        op = MaskOperator(mask)
        x_range = mask_range_element(mask)
        np.testing.assert_allclose(op.project(x_range), x_range, atol=TOL)

    def test_null_space_maps_to_zero(self, mask):
        """x in null(A) → forward(x) == 0 and project(x) == 0."""
        op = MaskOperator(mask)
        x_null = mask_null_element(mask)
        np.testing.assert_allclose(op.forward(x_null), 0.0, atol=TOL)
        np.testing.assert_allclose(op.project(x_null), 0.0, atol=TOL)

    def test_range_null_orthogonal(self, mask):
        """Range and null components of a general x are orthogonal."""
        op = MaskOperator(mask)
        x = general_array(*mask.shape[-2:])
        Px = op.project(x)
        null_comp = x - Px
        inner = np.sum(Px * null_comp)
        norm_product = np.linalg.norm(Px) * np.linalg.norm(null_comp)
        if norm_product > 1e-10:
            assert abs(inner) / norm_product < TOL


class TestMaskOperatorAnalytical:
    """Known-value tests for MaskOperator."""

    def test_forward_zeros_holes(self):
        """forward zeroes exactly the hole pixels."""
        mask = np.array([[1., 0.], [0., 1.]], dtype=np.float64)
        op = MaskOperator(mask)
        x = np.array([[3., 7.], [5., 2.]], dtype=np.float64)
        expected = np.array([[3., 0.], [0., 2.]], dtype=np.float64)
        np.testing.assert_allclose(op.forward(x), expected, atol=TOL)

    def test_pinv_same_as_forward(self):
        """pinv(y) == mask ⊙ y, same as forward."""
        mask = np.array([[1., 0.], [0., 1.]], dtype=np.float64)
        op = MaskOperator(mask)
        y = np.array([[9., 3.], [4., 6.]], dtype=np.float64)
        np.testing.assert_allclose(op.pinv(y), op.forward(y), atol=TOL)

    def test_null_space_is_hole_region(self):
        """Signal nonzero only in holes is annihilated by forward and project."""
        mask = np.array([[1., 1., 0.], [1., 0., 0.], [1., 1., 1.]], dtype=np.float64)
        op = MaskOperator(mask)
        # nonzero only where mask=0
        x_null = (1.0 - mask) * np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float64)
        np.testing.assert_allclose(op.forward(x_null), 0.0, atol=TOL)
        np.testing.assert_allclose(op.project(x_null), 0.0, atol=TOL)

    def test_range_space_is_observed_region(self):
        """Signal nonzero only in observed region is fixed by project."""
        mask = np.array([[1., 0., 1.], [0., 1., 0.]], dtype=np.float64)
        op = MaskOperator(mask)
        x_range = mask * np.array([[2., 9., 4.], [8., 3., 7.]], dtype=np.float64)
        np.testing.assert_allclose(op.project(x_range), x_range, atol=TOL)

    def test_all_zeros_mask(self):
        """All-zero mask: forward maps everything to zero."""
        mask = np.zeros((4, 4), dtype=np.float64)
        op = MaskOperator(mask)
        x = general_array(4, 4)
        np.testing.assert_allclose(op.forward(x), 0.0, atol=TOL)
        np.testing.assert_allclose(op.project(x), 0.0, atol=TOL)

    def test_all_ones_mask_is_identity(self):
        """All-one mask: all three maps are identity."""
        mask = np.ones((4, 4), dtype=np.float64)
        op = MaskOperator(mask)
        x = general_array(4, 4)
        np.testing.assert_allclose(op.forward(x), x, atol=TOL)
        np.testing.assert_allclose(op.pinv(x), x, atol=TOL)
        np.testing.assert_allclose(op.project(x), x, atol=TOL)

    def test_3d_input(self):
        """MaskOperator broadcasts over channel dimension (3-D input)."""
        mask = np.array([[1., 0.], [0., 1.]], dtype=np.float64)
        op = MaskOperator(mask)
        x = np.ones((3, 2, 2), dtype=np.float64) * 5.0
        expected = np.array([[[5., 0.], [0., 5.]]] * 3, dtype=np.float64)
        np.testing.assert_allclose(op.forward(x), expected, atol=TOL)


# ===========================================================================
# CircularBlur tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Kernel constructors (used in tests — specify kernels precisely)

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Symmetric 2-D Gaussian kernel, normalised to sum=1."""
    coords = np.arange(size) - size // 2
    row, col = np.meshgrid(coords, coords, indexing="ij")
    k = np.exp(-(row ** 2 + col ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def box_kernel(size: int) -> np.ndarray:
    """Uniform L×L box kernel, normalised to sum=1."""
    k = np.ones((size, size), dtype=np.float64)
    return k / (size * size)


def deblur_null_element(op: CircularBlur, H: int, W: int, seed: int = 7) -> np.ndarray:
    """Real-valued signal in null(A): Fourier support only at killed bins."""
    K = op._kernel_fft(H, W)
    null_mask = (np.abs(K) <= op.zero_threshold).astype(np.float64)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((H, W))
    X = np.fft.fft2(x)
    return np.fft.ifft2(X * null_mask).real


def deblur_range_element(op: CircularBlur, H: int, W: int, seed: int = 8) -> np.ndarray:
    """Real-valued signal in range(A): Fourier support only at kept bins."""
    K = op._kernel_fft(H, W)
    range_mask = (np.abs(K) > op.zero_threshold).astype(np.float64)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((H, W))
    X = np.fft.fft2(x)
    return np.fft.ifft2(X * range_mask).real


# ---------------------------------------------------------------------------
# Gaussian (full-rank) operator tests
#
# Parameters chosen so max|K_pinv| stays small (≈12) to keep
# the condition number well below TOL/eps ≈ 1e4:
#   sigma=0.5, kernel=5×5, image=8×8 → min|K| ≈ 0.085

GAUSSIAN_OP = CircularBlur(gaussian_kernel(5, 0.5), zero_threshold=1e-6)
GAUSSIAN_HW = (8, 8)


class TestCircularBlurGaussian:
    """Full-rank Gaussian blur: null space is empty, project ≈ identity."""

    @property
    def op(self):
        return GAUSSIAN_OP

    @property
    def H(self):
        return GAUSSIAN_HW[0]

    @property
    def W(self):
        return GAUSSIAN_HW[1]

    def test_min_kernel_magnitude_above_threshold(self):
        """Verify no Gaussian DFT bins are killed — null space is empty."""
        K = self.op._kernel_fft(self.H, self.W)
        min_absK = np.min(np.abs(K))
        assert min_absK > self.op.zero_threshold, (
            f"min|K|={min_absK:.2e} ≤ zero_threshold={self.op.zero_threshold:.2e}: "
            f"Gaussian has unexpected spectral zero(s)")

    def test_null_space_dim_zero(self):
        """Gaussian operator: null space dimension is 0."""
        null_dim = self.op.null_space_dim(self.H, self.W)
        assert null_dim == 0, f"Expected null_dim=0, got {null_dim}"

    def test_project_is_approximately_identity(self):
        """Full-rank Gaussian: project(x) ≈ x for all x."""
        x = general_array(self.H, self.W)
        Px = self.op.project(x)
        np.testing.assert_allclose(Px, x, atol=TOL,
            err_msg="Gaussian project ≠ identity (null space should be empty)")

    def test_pseudo_inverse_property(self):
        """A · A⁺ · A == A."""
        x = general_array(self.H, self.W)
        lhs = self.op.forward(self.op.pinv(self.op.forward(x)))
        rhs = self.op.forward(x)
        np.testing.assert_allclose(lhs, rhs, atol=TOL)

    def test_projector_idempotence(self):
        """(A⁺A)² == A⁺A."""
        x = general_array(self.H, self.W)
        Px = self.op.project(x)
        np.testing.assert_allclose(self.op.project(Px), Px, atol=TOL)

    def test_project_equals_pinv_of_forward(self):
        """project(x) == pinv(forward(x))."""
        x = general_array(self.H, self.W)
        np.testing.assert_allclose(
            self.op.project(x), self.op.pinv(self.op.forward(x)), atol=TOL)

    def test_forward_is_linear(self):
        """A(α·x + β·z) == α·A(x) + β·A(z)."""
        x = general_array(self.H, self.W, seed=10)
        z = general_array(self.H, self.W, seed=11)
        alpha, beta = 3.7, -2.1
        lhs = self.op.forward(alpha * x + beta * z)
        rhs = alpha * self.op.forward(x) + beta * self.op.forward(z)
        np.testing.assert_allclose(lhs, rhs, atol=TOL)

    def test_pinv_is_linear(self):
        """A⁺(α·y + β·w) == α·A⁺(y) + β·A⁺(w).

        Condition number of this Gaussian is low (~12) so floating-point
        linearity error stays well below TOL=1e-12.
        """
        y1 = general_array(self.H, self.W, seed=20)
        y2 = general_array(self.H, self.W, seed=21)
        alpha, beta = -1.5, 4.2
        lhs = self.op.pinv(alpha * y1 + beta * y2)
        rhs = alpha * self.op.pinv(y1) + beta * self.op.pinv(y2)
        np.testing.assert_allclose(lhs, rhs, atol=TOL)

    def test_3d_input(self):
        """CircularBlur handles (C, H, W) input."""
        x = general_array(self.H, self.W, c=3)
        Px = self.op.project(x)
        np.testing.assert_allclose(Px, x, atol=TOL)

    def test_range_space_fixed(self):
        """Range-space element fixed by projector (trivially all signals for Gaussian)."""
        x_range = deblur_range_element(self.op, self.H, self.W)
        np.testing.assert_allclose(
            self.op.project(x_range), x_range, atol=TOL)


# ---------------------------------------------------------------------------
# Box (rank-deficient) operator tests
#
# L=4 box kernel on 12×12 image: L divides N=12 exactly, so DFT zeros
# land precisely on integer bins.  Spectral zeros at ω ∈ {3,6,9} per
# axis (1D); 2D null dim = 12² − 9² = 144 − 81 = 63.
# max|K_pinv| at non-zero bins ≈ 20 (condition number is low).
# zero_threshold = 1e-6: genuine zeros are ≈1e-16, kept bins ≥ 0.05.

BOX_OP = CircularBlur(box_kernel(4), zero_threshold=1e-6)
BOX_HW = (12, 12)


class TestCircularBlurBox:
    """Rank-deficient box blur: non-trivial null space of dimension 63."""

    @property
    def op(self):
        return BOX_OP

    @property
    def H(self):
        return BOX_HW[0]

    @property
    def W(self):
        return BOX_HW[1]

    def test_null_space_dim(self):
        """4×4 box on 12×12: null space dimension = 12² − 9² = 63."""
        null_dim = self.op.null_space_dim(self.H, self.W)
        assert null_dim == 63, f"Expected null_dim=63, got {null_dim}"

    def test_range_space_dim(self):
        """Range space dimension = 81 (the 9×9 non-zero-bin grid)."""
        range_dim = self.op.range_space_dim(self.H, self.W)
        assert range_dim == 81, f"Expected range_dim=81, got {range_dim}"

    def test_zero_bins_at_expected_locations(self):
        """Spectral zeros are exactly at (ω_h, ω_w) where ω_h∈{3,6,9} or ω_w∈{3,6,9}."""
        K = self.op._kernel_fft(self.H, self.W)
        zero_bins = np.abs(K) <= self.op.zero_threshold
        # expected zero locations: ω_h or ω_w is a multiple of N/L = 3 (nonzero)
        expected = np.zeros((self.H, self.W), dtype=bool)
        multiples = {3, 6, 9}
        for i in range(self.H):
            for j in range(self.W):
                if i in multiples or j in multiples:
                    expected[i, j] = True
        np.testing.assert_array_equal(zero_bins, expected,
            err_msg="Spectral zero locations do not match theory")

    def test_genuine_zeros_are_machine_epsilon(self):
        """Spectral zeros are ≤ 1e-14 (not just below zero_threshold=1e-6)."""
        K = self.op._kernel_fft(self.H, self.W)
        null_mask = np.abs(K) <= self.op.zero_threshold
        max_null_absK = np.max(np.abs(K[null_mask]))
        assert max_null_absK < 1e-14, (
            f"'Zero' bin has |K|={max_null_absK:.2e}; expected machine epsilon. "
            f"Spectral zeros are not exact — threshold choice or kernel is wrong.")

    def test_pseudo_inverse_property(self):
        """A · A⁺ · A == A at TOL=1e-12.

        At zero bins K≈0 (machine epsilon), K_pinv=0, so K·K_pinv·K = 0 ≈ K.
        The residual is at most max|K_zero|·max|X|/N² ≪ 1e-12.
        """
        x = general_array(self.H, self.W)
        lhs = self.op.forward(self.op.pinv(self.op.forward(x)))
        rhs = self.op.forward(x)
        np.testing.assert_allclose(lhs, rhs, atol=TOL,
            err_msg="A·A⁺·A ≠ A for box kernel")

    def test_projector_idempotence(self):
        """(A⁺A)² == A⁺A."""
        x = general_array(self.H, self.W)
        Px = self.op.project(x)
        np.testing.assert_allclose(self.op.project(Px), Px, atol=TOL)

    def test_project_equals_pinv_of_forward(self):
        """project(x) == pinv(forward(x))."""
        x = general_array(self.H, self.W)
        np.testing.assert_allclose(
            self.op.project(x), self.op.pinv(self.op.forward(x)), atol=TOL)

    def test_forward_is_linear(self):
        """A(α·x + β·z) == α·A(x) + β·A(z)."""
        x = general_array(self.H, self.W, seed=10)
        z = general_array(self.H, self.W, seed=11)
        alpha, beta = 3.7, -2.1
        lhs = self.op.forward(alpha * x + beta * z)
        rhs = alpha * self.op.forward(x) + beta * self.op.forward(z)
        np.testing.assert_allclose(lhs, rhs, atol=TOL)

    def test_pinv_is_linear(self):
        """A⁺(α·y + β·w) == α·A⁺(y) + β·A⁺(w).

        max|K_pinv| ≈ 20 for this box kernel, so condition number is low
        and floating-point linearity error stays well below TOL=1e-12.
        """
        y1 = general_array(self.H, self.W, seed=20)
        y2 = general_array(self.H, self.W, seed=21)
        alpha, beta = -1.5, 4.2
        lhs = self.op.pinv(alpha * y1 + beta * y2)
        rhs = alpha * self.op.pinv(y1) + beta * self.op.pinv(y2)
        np.testing.assert_allclose(lhs, rhs, atol=TOL)

    def test_null_space_annihilated_by_forward(self):
        """x in null(A): forward(x) ≈ 0."""
        x_null = deblur_null_element(self.op, self.H, self.W)
        y = self.op.forward(x_null)
        np.testing.assert_allclose(y, 0.0, atol=TOL,
            err_msg="null-space element not annihilated by forward")

    def test_null_space_annihilated_by_project(self):
        """x in null(A): project(x) ≈ 0."""
        x_null = deblur_null_element(self.op, self.H, self.W)
        Px = self.op.project(x_null)
        np.testing.assert_allclose(Px, 0.0, atol=TOL,
            err_msg="null-space element not annihilated by project")

    def test_range_space_fixed_by_projector(self):
        """x in range(A): project(x) == x."""
        x_range = deblur_range_element(self.op, self.H, self.W)
        np.testing.assert_allclose(self.op.project(x_range), x_range, atol=TOL,
            err_msg="range-space element not fixed by projector")

    def test_project_not_identity(self):
        """For rank-deficient box kernel, project(x) ≠ x in general."""
        rng = np.random.default_rng(99)
        x = rng.standard_normal((self.H, self.W))
        Px = self.op.project(x)
        # null-space component must be non-trivial for a generic signal
        null_norm = np.linalg.norm(x - Px)
        assert null_norm > 0.1, (
            f"project appears to be identity (null component norm={null_norm:.3e}); "
            f"expected non-trivial null space for 4×4 box kernel")

    def test_range_null_orthogonal(self):
        """Range and null components are orthogonal."""
        x = general_array(self.H, self.W)
        Px = self.op.project(x)
        null_comp = x - Px
        inner = np.sum(Px * null_comp)
        norm_product = np.linalg.norm(Px) * np.linalg.norm(null_comp)
        if norm_product > 1e-10:
            assert abs(inner) / norm_product < TOL

    def test_3d_input(self):
        """CircularBlur handles (C, H, W) input."""
        x = general_array(self.H, self.W, c=3)
        lhs = self.op.forward(self.op.pinv(self.op.forward(x)))
        rhs = self.op.forward(x)
        np.testing.assert_allclose(lhs, rhs, atol=TOL)

    def test_analytical_dc_component(self):
        """DC component (constant image): forward = input (K[0,0]=1), pinv recovers it."""
        x_dc = np.ones((self.H, self.W), dtype=np.float64)
        y = self.op.forward(x_dc)
        np.testing.assert_allclose(y, x_dc, atol=TOL,
            err_msg="K[0,0] ≠ 1: box kernel not normalised")
        x_rec = self.op.pinv(y)
        np.testing.assert_allclose(x_rec, x_dc, atol=TOL,
            err_msg="pinv(forward(DC)) ≠ DC")

    def test_analytical_null_sinusoid(self):
        """cos(2π·3·h/12): known null-space element — forward maps it to 0."""
        h = np.arange(self.H)
        x_null = np.tile(np.cos(2 * np.pi * 3 * h / self.H), (self.W, 1)).T
        y = self.op.forward(x_null)
        np.testing.assert_allclose(y, 0.0, atol=TOL,
            err_msg="cosine at spectral zero not annihilated")
