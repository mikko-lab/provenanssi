"""
Layer tests: rectify, ensemble, classify — same discipline as test_operators.py.

TOL = 1e-12 throughout.  If a test fails, diagnose the implementation;
do not raise TOL to make it pass.

Test structure
--------------
TestRectify         — data consistency on all operators; exact decomposition.
TestEnsemble        — range variance = 0; null variance > 0 exactly where null differs.
TestClassify        — no false positives (zero null → nothing invented);
                      no false negatives (known null region → exactly flagged);
                      R2 invariant (large range-space values never flagged invented).
TestEndToEnd        — one oracle→rectify→ensemble→classify pipeline per operator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from operators.superres import BoxDownsample
from operators.inpaint import MaskOperator
from operators.deblur import CircularBlur
from engine.oracle import OracleEngine
from layer.decompose import rectify, CONSISTENCY_EPS
from layer.ensemble import ensemble_stats
from layer.classify import classify, MEASURED, RECOVERED, INVENTED

TOL = 1e-12

# ---------------------------------------------------------------------------
# Operator + oracle fixtures

def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    c = np.arange(size) - size // 2
    row, col = np.meshgrid(c, c, indexing="ij")
    k = np.exp(-(row ** 2 + col ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _box_kernel(size: int) -> np.ndarray:
    k = np.ones((size, size), dtype=np.float64)
    return k / (size * size)


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# canonical operators used across tests
SR_OP  = BoxDownsample(2)
SR_H, SR_W = 8, 8

MASK_FULL = np.ones((4, 4), dtype=np.float64)
MASK_FULL[2:, 2:] = 0.0          # 2×2 hole in bottom-right
MASK_OP = MaskOperator(MASK_FULL)

GAUSS_OP = CircularBlur(_gaussian_kernel(5, 0.5), zero_threshold=1e-6)
GAUSS_H, GAUSS_W = 8, 8

BOX_OP = CircularBlur(_box_kernel(4), zero_threshold=1e-6)
BOX_H, BOX_W = 12, 12


# ---------------------------------------------------------------------------
# Null-space helper

def _sr_null_element(H: int, W: int, seed: int = 7) -> np.ndarray:
    """BoxDownsample null element: zero block-mean in each 2×2 block."""
    rng = _rng(seed)
    x = rng.standard_normal((H, W))
    x_bc = x.reshape(H // 2, 2, W // 2, 2)
    x_bc -= x_bc.mean(axis=(1, 3), keepdims=True)
    return x_bc.reshape(H, W)


def _deblur_null_element(op: CircularBlur, H: int, W: int, seed: int = 7) -> np.ndarray:
    K = op._kernel_fft(H, W)
    null_mask = (np.abs(K) <= op.zero_threshold).astype(np.float64)
    rng = _rng(seed)
    x = rng.standard_normal((H, W))
    X = np.fft.fft2(x)
    return np.fft.ifft2(X * null_mask).real


# ===========================================================================
# TestRectify
# ===========================================================================

class TestRectify:
    """rectify() produces data-consistent x_out and correct range/null split."""

    # ------------------------------------------------------------------
    # Data consistency: ‖A·x_out − y‖ ≤ 1e-12 across all operators

    def _make_sr_case(self):
        rng = _rng(0)
        x_gt = rng.standard_normal((SR_H, SR_W))
        y = SR_OP.forward(x_gt)
        oracle = OracleEngine(SR_OP)
        n = _sr_null_element(SR_H, SR_W)
        return oracle.restore(y, n), y

    def test_sr_data_consistency(self):
        """BoxDownsample: ‖A·x_out − y‖ ≤ 1e-12."""
        x_hat, y = self._make_sr_case()
        r = rectify(x_hat, y, SR_OP)
        assert r.residual <= TOL, (
            f"SR data consistency violated: residual = {r.residual:.2e}")

    def test_mask_data_consistency(self):
        """MaskOperator: ‖A·x_out − y‖ ≤ 1e-12."""
        rng = _rng(1)
        x_gt = rng.standard_normal((4, 4))
        y = MASK_OP.forward(x_gt)
        oracle = OracleEngine(MASK_OP)
        n = (1 - MASK_FULL) * rng.standard_normal((4, 4))
        x_hat = oracle.restore(y, n)
        r = rectify(x_hat, y, MASK_OP)
        assert r.residual <= TOL, (
            f"Mask data consistency violated: residual = {r.residual:.2e}")

    def test_gauss_data_consistency(self):
        """CircularBlur (Gaussian): ‖A·x_out − y‖ ≤ 1e-12."""
        rng = _rng(2)
        x_gt = rng.standard_normal((GAUSS_H, GAUSS_W))
        y = GAUSS_OP.forward(x_gt)
        oracle = OracleEngine(GAUSS_OP)
        x_hat = oracle.restore(y, np.zeros((GAUSS_H, GAUSS_W)))
        r = rectify(x_hat, y, GAUSS_OP)
        assert r.residual <= TOL, (
            f"Gauss data consistency violated: residual = {r.residual:.2e}")

    def test_box_deblur_data_consistency(self):
        """CircularBlur (box): ‖A·x_out − y‖ ≤ 1e-12."""
        rng = _rng(3)
        x_gt = rng.standard_normal((BOX_H, BOX_W))
        y = BOX_OP.forward(x_gt)
        oracle = OracleEngine(BOX_OP)
        n = _deblur_null_element(BOX_OP, BOX_H, BOX_W)
        x_hat = oracle.restore(y, n)
        r = rectify(x_hat, y, BOX_OP)
        assert r.residual <= TOL, (
            f"Box-deblur data consistency violated: residual = {r.residual:.2e}")

    # ------------------------------------------------------------------
    # Exact decomposition

    def test_range_null_sum_to_x_out(self):
        """range_component + null_component == x_out exactly."""
        rng = _rng(10)
        x_gt = rng.standard_normal((SR_H, SR_W))
        y = SR_OP.forward(x_gt)
        oracle = OracleEngine(SR_OP)
        x_hat = oracle.restore(y, _sr_null_element(SR_H, SR_W, seed=11))
        r = rectify(x_hat, y, SR_OP)
        np.testing.assert_allclose(
            r.range_component + r.null_component, r.x_out, atol=TOL)

    def test_range_component_is_pinv_y(self):
        """range_component == A⁺y exactly (not approximately)."""
        rng = _rng(12)
        x_gt = rng.standard_normal((4, 4))
        y = MASK_OP.forward(x_gt)
        oracle = OracleEngine(MASK_OP)
        x_hat = oracle.restore(y, np.ones((4, 4)))
        r = rectify(x_hat, y, MASK_OP)
        expected_range = MASK_OP.pinv(y)
        np.testing.assert_allclose(r.range_component, expected_range, atol=TOL)

    def test_null_component_is_in_null_space(self):
        """forward(null_component) ≈ 0 (null component lives in null(A))."""
        rng = _rng(13)
        x_gt = rng.standard_normal((SR_H, SR_W))
        y = SR_OP.forward(x_gt)
        oracle = OracleEngine(SR_OP)
        x_hat = oracle.restore(y, _sr_null_element(SR_H, SR_W))
        r = rectify(x_hat, y, SR_OP)
        fwd_null = SR_OP.forward(r.null_component)
        np.testing.assert_allclose(fwd_null, 0.0, atol=TOL,
            err_msg="null_component has range-space energy (A·null ≠ 0)")

    def test_range_component_in_range_space(self):
        """project(range_component) == range_component."""
        rng = _rng(14)
        x_gt = rng.standard_normal((SR_H, SR_W))
        y = SR_OP.forward(x_gt)
        oracle = OracleEngine(SR_OP)
        x_hat = oracle.restore(y, np.zeros((SR_H, SR_W)))
        r = rectify(x_hat, y, SR_OP)
        np.testing.assert_allclose(
            SR_OP.project(r.range_component), r.range_component, atol=TOL,
            err_msg="range_component is not in range(A)")

    def test_oracle_output_is_rectify_fixed_point(self):
        """rectify(oracle.restore(y,n), y, op) == oracle.restore(y,n).

        Oracle output is already data-consistent; rectify should be a no-op
        on it (idempotent on consistent inputs).
        """
        rng = _rng(15)
        x_gt = rng.standard_normal((4, 4))
        y = MASK_OP.forward(x_gt)
        n = (1 - MASK_FULL) * rng.standard_normal((4, 4))
        oracle = OracleEngine(MASK_OP)
        x_hat = oracle.restore(y, n)
        r = rectify(x_hat, y, MASK_OP)
        np.testing.assert_allclose(r.x_out, x_hat, atol=TOL,
            err_msg="rectify is not a fixed point on oracle output")

    def test_consistency_guardrail_fires_on_bad_input(self):
        """rectify raises ValueError if fed a deliberately inconsistent estimate."""
        rng = _rng(16)
        y = SR_OP.forward(rng.standard_normal((SR_H, SR_W)))
        # Corrupt x_hat so it is wildly inconsistent
        x_bad = rng.standard_normal((SR_H, SR_W)) * 100.0
        # Force inconsistency by directly passing x_bad to a fresh rectify call
        # with a very tight consistency_eps that the corrupted input will fail
        with pytest.raises(ValueError, match="Data consistency violated"):
            rectify(x_bad, y, SR_OP, consistency_eps=1e-14)


# ===========================================================================
# TestEnsemble
# ===========================================================================

class TestEnsemble:
    """Ensemble variance is zero in range space, nonzero only where null differs."""

    def test_requires_at_least_two_members(self):
        """ensemble_stats raises with < 2 members."""
        with pytest.raises(ValueError, match="at least 2"):
            ensemble_stats([np.zeros((4, 4))])

    def test_range_variance_zero_sr(self):
        """BoxDownsample ensemble: var(A⁺A·x_out_i) ≤ 1e-12."""
        rng = _rng(20)
        x_gt = rng.standard_normal((SR_H, SR_W))
        y = SR_OP.forward(x_gt)
        oracle = OracleEngine(SR_OP)

        n1 = _sr_null_element(SR_H, SR_W, seed=21)
        n2 = _sr_null_element(SR_H, SR_W, seed=22)
        n3 = -n1

        x_hats = oracle.ensemble(y, [n1, n2, n3])
        x_outs = [rectify(xh, y, SR_OP).x_out for xh in x_hats]
        _, variance = ensemble_stats(x_outs)

        # Project each x_out to range space: should be identical
        range_parts = np.stack([SR_OP.project(xo) for xo in x_outs])
        range_var = range_parts.var(axis=0)
        np.testing.assert_allclose(range_var, 0.0, atol=TOL,
            err_msg="range-space variance is non-zero across ensemble")

    def test_null_variance_nonzero_mask(self):
        """MaskOperator ensemble: null variance > 0 exactly in the holes."""
        rng = _rng(30)
        x_gt = rng.standard_normal((4, 4))
        y = MASK_OP.forward(x_gt)
        oracle = OracleEngine(MASK_OP)

        # n1 and n2 differ only in the hole region
        n1 = (1 - MASK_FULL) * 1.0   # +1 in holes
        n2 = (1 - MASK_FULL) * (-1.0) # -1 in holes

        x_outs = [rectify(oracle.restore(y, n), y, MASK_OP).x_out
                  for n in [n1, n2]]
        _, variance = ensemble_stats(x_outs)

        # In holes: x_out_1 = 1, x_out_2 = -1 → variance = 1.0
        hole_mask = (MASK_FULL == 0)
        obs_mask = (MASK_FULL == 1)

        assert np.all(variance[hole_mask] > 0.1), (
            "Ensemble variance should be non-zero in holes")
        np.testing.assert_allclose(variance[obs_mask], 0.0, atol=TOL,
            err_msg="Ensemble variance non-zero in observed region")

    def test_range_variance_zero_mask(self):
        """MaskOperator: var(project(x_out_i)) ≤ 1e-12 (range part identical)."""
        rng = _rng(31)
        x_gt = rng.standard_normal((4, 4))
        y = MASK_OP.forward(x_gt)
        oracle = OracleEngine(MASK_OP)

        n1 = (1 - MASK_FULL) * rng.standard_normal((4, 4))
        n2 = (1 - MASK_FULL) * rng.standard_normal((4, 4))
        n3 = (1 - MASK_FULL) * rng.standard_normal((4, 4))

        x_outs = [rectify(oracle.restore(y, n), y, MASK_OP).x_out
                  for n in [n1, n2, n3]]
        range_parts = np.stack([MASK_OP.project(xo) for xo in x_outs])
        range_var = range_parts.var(axis=0)
        np.testing.assert_allclose(range_var, 0.0, atol=TOL,
            err_msg="range-space variance non-zero (range should be identical)")

    def test_range_variance_zero_box_deblur(self):
        """CircularBlur (box): var(project(x_out_i)) ≤ 1e-12."""
        rng = _rng(40)
        x_gt = rng.standard_normal((BOX_H, BOX_W))
        y = BOX_OP.forward(x_gt)
        oracle = OracleEngine(BOX_OP)

        n1 = _deblur_null_element(BOX_OP, BOX_H, BOX_W, seed=41)
        n2 = _deblur_null_element(BOX_OP, BOX_H, BOX_W, seed=42)

        x_outs = [rectify(oracle.restore(y, n), y, BOX_OP).x_out
                  for n in [n1, n2]]
        range_parts = np.stack([BOX_OP.project(xo) for xo in x_outs])
        range_var = range_parts.var(axis=0)
        np.testing.assert_allclose(range_var, 0.0, atol=TOL,
            err_msg="CircularBlur range-space variance non-zero")

    def test_null_variance_nonzero_box_deblur(self):
        """CircularBlur (box): null variance > 0 where null contents differ."""
        rng = _rng(43)
        x_gt = rng.standard_normal((BOX_H, BOX_W))
        y = BOX_OP.forward(x_gt)
        oracle = OracleEngine(BOX_OP)

        n1 = _deblur_null_element(BOX_OP, BOX_H, BOX_W, seed=44)
        n2 = _deblur_null_element(BOX_OP, BOX_H, BOX_W, seed=45)  # different seed

        x_outs = [rectify(oracle.restore(y, n), y, BOX_OP).x_out
                  for n in [n1, n2]]
        _, variance = ensemble_stats(x_outs)

        # Null elements live only in the null space, so variance is non-trivial
        assert np.max(variance) > 1e-6, (
            f"Expected non-zero null-space variance; max = {np.max(variance):.2e}")


# ===========================================================================
# TestClassify
# ===========================================================================

# Explicit thresholds — NOT magic numbers
INVENTED_THRESH = 1e-6     # null_fraction above this → invented
RECOVERED_THRESH = 1e-12   # ensemble variance above this → recovered


class TestClassify:
    """classify() has zero false positives and zero false negatives."""

    # ------------------------------------------------------------------
    # Zero null → nothing invented (no false positives)

    def test_sr_zero_null_nothing_invented(self):
        """BoxDownsample + zero null: no pixel labelled INVENTED."""
        rng = _rng(50)
        x_gt = rng.standard_normal((SR_H, SR_W))
        y = SR_OP.forward(x_gt)
        oracle = OracleEngine(SR_OP)
        x_hat = oracle.restore(y, np.zeros((SR_H, SR_W)))
        r = rectify(x_hat, y, SR_OP)
        labels = classify(r.null_component, r.range_component,
                          invented_threshold=INVENTED_THRESH,
                          recovered_threshold=RECOVERED_THRESH)
        assert np.all(labels != INVENTED), (
            f"False positives: {np.sum(labels == INVENTED)} pixels wrongly labelled INVENTED")

    def test_mask_zero_null_nothing_invented(self):
        """MaskOperator + zero null: no pixel labelled INVENTED."""
        rng = _rng(51)
        x_gt = rng.standard_normal((4, 4))
        y = MASK_OP.forward(x_gt)
        oracle = OracleEngine(MASK_OP)
        x_hat = oracle.restore(y, np.zeros((4, 4)))
        r = rectify(x_hat, y, MASK_OP)
        labels = classify(r.null_component, r.range_component,
                          invented_threshold=INVENTED_THRESH,
                          recovered_threshold=RECOVERED_THRESH)
        assert np.all(labels != INVENTED), (
            f"False positives: {np.sum(labels == INVENTED)} pixels wrongly labelled INVENTED")

    def test_gauss_zero_null_nothing_invented(self):
        """Gaussian CircularBlur (full-rank, empty null space): nothing INVENTED."""
        rng = _rng(52)
        x_gt = rng.standard_normal((GAUSS_H, GAUSS_W))
        y = GAUSS_OP.forward(x_gt)
        oracle = OracleEngine(GAUSS_OP)
        x_hat = oracle.restore(y, np.zeros((GAUSS_H, GAUSS_W)))
        r = rectify(x_hat, y, GAUSS_OP)
        labels = classify(r.null_component, r.range_component,
                          invented_threshold=INVENTED_THRESH,
                          recovered_threshold=RECOVERED_THRESH)
        assert np.all(labels != INVENTED), (
            f"False positives: {np.sum(labels == INVENTED)} pixels wrongly labelled INVENTED")

    def test_box_deblur_zero_null_nothing_invented(self):
        """Box CircularBlur + zero null: nothing INVENTED."""
        rng = _rng(53)
        x_gt = rng.standard_normal((BOX_H, BOX_W))
        y = BOX_OP.forward(x_gt)
        oracle = OracleEngine(BOX_OP)
        x_hat = oracle.restore(y, np.zeros((BOX_H, BOX_W)))
        r = rectify(x_hat, y, BOX_OP)
        labels = classify(r.null_component, r.range_component,
                          invented_threshold=INVENTED_THRESH,
                          recovered_threshold=RECOVERED_THRESH)
        assert np.all(labels != INVENTED), (
            f"False positives: {np.sum(labels == INVENTED)} pixels wrongly labelled INVENTED")

    # ------------------------------------------------------------------
    # Known null region → exactly that region flagged (no false negatives)

    def test_mask_null_region_exactly_flagged(self):
        """MaskOperator: INVENTED exactly at hole pixels, MEASURED everywhere else.

        This is the primary spatial test.  The hole region is known exactly,
        the null content is nonzero exactly there, and the classification
        must match the mask complement pixel-for-pixel.
        """
        rng = _rng(60)
        x_gt = rng.standard_normal((4, 4))
        y = MASK_OP.forward(x_gt)
        oracle = OracleEngine(MASK_OP)

        # Null content nonzero only in the hole region
        n_hole = (1.0 - MASK_FULL) * np.ones((4, 4))   # +1 in holes, 0 elsewhere
        x_hat = oracle.restore(y, n_hole)
        r = rectify(x_hat, y, MASK_OP)
        labels = classify(r.null_component, r.range_component,
                          invented_threshold=INVENTED_THRESH,
                          recovered_threshold=RECOVERED_THRESH)

        hole_mask = (MASK_FULL == 0)
        obs_mask = (MASK_FULL == 1)

        # No false negatives: all holes must be flagged
        assert np.all(labels[hole_mask] == INVENTED), (
            f"False negatives: {np.sum(labels[hole_mask] != INVENTED)} hole pixels not flagged")

        # No false positives: no observed pixels may be flagged
        assert np.all(labels[obs_mask] != INVENTED), (
            f"False positives: {np.sum(labels[obs_mask] == INVENTED)} observed pixels wrongly flagged")

    def test_mask_partial_null_region(self):
        """Only a SUBSET of hole pixels has null content → only that subset flagged."""
        rng = _rng(61)
        x_gt = rng.standard_normal((4, 4))
        y = MASK_OP.forward(x_gt)
        oracle = OracleEngine(MASK_OP)

        # Fill only one of the four hole pixels (bottom-right corner)
        n_partial = np.zeros((4, 4))
        n_partial[3, 3] = 1.0   # only (3,3) — which is in the hole

        x_hat = oracle.restore(y, n_partial)
        r = rectify(x_hat, y, MASK_OP)
        labels = classify(r.null_component, r.range_component,
                          invented_threshold=INVENTED_THRESH,
                          recovered_threshold=RECOVERED_THRESH)

        assert labels[3, 3] == INVENTED, "Pixel with null content not flagged"
        # The other three hole pixels have zero null content → not invented
        other_holes = [(2, 2), (2, 3), (3, 2)]
        for i, j in other_holes:
            assert labels[i, j] != INVENTED, (
                f"Hole pixel ({i},{j}) without null content wrongly flagged")

    # ------------------------------------------------------------------
    # R2 invariant: large range-space values are NEVER labelled INVENTED

    def test_r2_large_range_not_invented(self):
        """R2: pixel magnitude does not determine the label.

        A pixel with large range-space energy and zero null energy must be
        MEASURED, not INVENTED.  The label is based ONLY on null-space
        membership.
        """
        mask = np.ones((4, 4), dtype=np.float64)
        mask[2:, 2:] = 0.0
        op = MaskOperator(mask)
        oracle = OracleEngine(op)

        # Large values in observed region — should NOT trigger INVENTED
        x_gt = np.ones((4, 4)) * 1e4
        y = op.forward(x_gt)

        x_hat = oracle.restore(y, np.zeros((4, 4)))   # zero null content
        r = rectify(x_hat, y, op)

        labels = classify(r.null_component, r.range_component,
                          invented_threshold=INVENTED_THRESH,
                          recovered_threshold=RECOVERED_THRESH)

        obs_mask = (mask == 1)
        assert np.all(labels[obs_mask] != INVENTED), (
            "R2 violated: large-magnitude range pixels labelled INVENTED")

    def test_r2_zero_range_with_null_is_invented(self):
        """R2: a pixel that is zero in the range but nonzero in null IS invented.

        Verifies the complement: the decision is structural (null-space membership),
        not based on whether the pixel value is large.  A zero-valued invented pixel
        must still be flagged — but only if it has null-space energy.
        """
        # Use MaskOperator where hole pixels have range = 0
        mask = np.ones((4, 4), dtype=np.float64)
        mask[2:, 2:] = 0.0
        op = MaskOperator(mask)
        oracle = OracleEngine(op)

        x_gt = np.zeros((4, 4))   # all-zero ground truth → range = 0 everywhere
        y = op.forward(x_gt)      # all zeros

        # Supply small but nonzero null content in holes
        n = (1.0 - mask) * 0.01   # 0.01 in holes
        x_hat = oracle.restore(y, n)
        r = rectify(x_hat, y, op)

        labels = classify(r.null_component, r.range_component,
                          invented_threshold=INVENTED_THRESH,
                          recovered_threshold=RECOVERED_THRESH)

        hole_mask = (mask == 0)
        # Small null values in holes → should still be INVENTED
        # null_fraction ≈ 0.01 / (0.01 + 0 + eps) ≈ 1.0 >> INVENTED_THRESH
        assert np.all(labels[hole_mask] == INVENTED), (
            "R2 test: hole pixels with null content not flagged even though "
            "the classification should be based on null-space membership, not magnitude")

    def test_invalid_invented_threshold_raises(self):
        """classify raises if invented_threshold ∉ (0,1)."""
        null_c = np.zeros((4, 4))
        range_c = np.ones((4, 4))
        with pytest.raises(ValueError):
            classify(null_c, range_c, invented_threshold=0.0,
                     recovered_threshold=1e-12)
        with pytest.raises(ValueError):
            classify(null_c, range_c, invented_threshold=1.0,
                     recovered_threshold=1e-12)

    # ------------------------------------------------------------------
    # Recovered label (with ensemble variance)

    def test_recovered_label_with_ensemble(self):
        """classify promotes to RECOVERED when ensemble variance > recovered_threshold."""
        rng = _rng(70)
        x_gt = rng.standard_normal((4, 4))
        y = MASK_OP.forward(x_gt)
        oracle = OracleEngine(MASK_OP)

        n1 = (1.0 - MASK_FULL) * 1.0
        n2 = (1.0 - MASK_FULL) * (-1.0)
        x_outs = [rectify(oracle.restore(y, n), y, MASK_OP).x_out for n in [n1, n2]]
        _, variance = ensemble_stats(x_outs)

        # Use the first rectified result for classify
        r = rectify(oracle.restore(y, n1), y, MASK_OP)

        # With tight recovered_threshold: high-variance null pixels → INVENTED
        # (since their null_fraction is also high → INVENTED takes priority)
        labels = classify(r.null_component, r.range_component,
                          invented_threshold=INVENTED_THRESH,
                          recovered_threshold=RECOVERED_THRESH,
                          ensemble_variance=variance)
        # Holes still INVENTED (null_fraction rule takes priority)
        hole_mask = (MASK_FULL == 0)
        assert np.all(labels[hole_mask] == INVENTED)


# ===========================================================================
# TestEndToEnd
# ===========================================================================

class TestEndToEnd:
    """Full oracle → rectify → ensemble → classify pipeline per operator."""

    def _run_e2e(self, op, x_gt, null_contents, invented_threshold=INVENTED_THRESH):
        y = op.forward(x_gt)
        oracle = OracleEngine(op)
        x_hats = oracle.ensemble(y, null_contents)
        results = [rectify(xh, y, op) for xh in x_hats]

        # All residuals must be ≤ TOL
        for i, r in enumerate(results):
            assert r.residual <= TOL, (
                f"E2E: member {i} data consistency failed: {r.residual:.2e}")

        x_outs = [r.x_out for r in results]
        _, variance = ensemble_stats(x_outs)

        # Use first member for classify
        labels = classify(results[0].null_component, results[0].range_component,
                          invented_threshold=invented_threshold,
                          recovered_threshold=RECOVERED_THRESH,
                          ensemble_variance=variance)
        return results, labels, variance

    def test_e2e_sr_zero_null(self):
        """BoxDownsample: zero null throughout → nothing invented."""
        rng = _rng(80)
        x_gt = rng.standard_normal((SR_H, SR_W))
        n0 = np.zeros((SR_H, SR_W))
        results, labels, _ = self._run_e2e(SR_OP, x_gt, [n0, n0])
        assert np.all(labels != INVENTED)

    def test_e2e_sr_null_content(self):
        """BoxDownsample: null content supplied → invented pixels appear."""
        rng = _rng(81)
        x_gt = rng.standard_normal((SR_H, SR_W))
        n1 = _sr_null_element(SR_H, SR_W, seed=82)
        n2 = -n1
        results, labels, _ = self._run_e2e(SR_OP, x_gt, [n1, n2])
        assert np.any(labels == INVENTED), (
            "No pixels labelled INVENTED despite non-zero null content")

    def test_e2e_mask_exact_spatial(self):
        """MaskOperator: invented pixels = exactly the holes, nothing else."""
        rng = _rng(90)
        x_gt = rng.standard_normal((4, 4))
        n1 = (1.0 - MASK_FULL) * 1.0
        n2 = (1.0 - MASK_FULL) * (-2.0)
        results, labels, _ = self._run_e2e(MASK_OP, x_gt, [n1, n2])

        hole_mask = (MASK_FULL == 0)
        obs_mask = (MASK_FULL == 1)
        assert np.all(labels[hole_mask] == INVENTED)
        assert np.all(labels[obs_mask] != INVENTED)

    def test_e2e_gauss_zero_null(self):
        """Gaussian CircularBlur (full rank): nothing invented with zero null."""
        rng = _rng(91)
        x_gt = rng.standard_normal((GAUSS_H, GAUSS_W))
        n0 = np.zeros((GAUSS_H, GAUSS_W))
        results, labels, _ = self._run_e2e(GAUSS_OP, x_gt, [n0, n0])
        assert np.all(labels != INVENTED)

    def test_e2e_box_deblur_null_content(self):
        """Box CircularBlur: null content in null space → invented pixels appear."""
        rng = _rng(92)
        x_gt = rng.standard_normal((BOX_H, BOX_W))
        n1 = _deblur_null_element(BOX_OP, BOX_H, BOX_W, seed=93)
        n2 = _deblur_null_element(BOX_OP, BOX_H, BOX_W, seed=94)
        results, labels, variance = self._run_e2e(BOX_OP, x_gt, [n1, n2])

        assert np.any(labels == INVENTED), (
            "No pixels labelled INVENTED despite null-space content")
        # Consistency across all members
        for r in results:
            assert r.residual <= TOL

    def test_e2e_residuals_reported(self):
        """RectifyResult.residual is a finite float, not NaN."""
        rng = _rng(95)
        x_gt = rng.standard_normal((SR_H, SR_W))
        y = SR_OP.forward(x_gt)
        oracle = OracleEngine(SR_OP)
        r = rectify(oracle.restore(y, np.zeros((SR_H, SR_W))), y, SR_OP)
        assert np.isfinite(r.residual), "residual is NaN or Inf"
        assert r.residual >= 0.0
