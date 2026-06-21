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
