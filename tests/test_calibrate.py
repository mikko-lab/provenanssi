"""
Calibration tests (R8) — eval/synthetic.py + layer/calibrate.py.

TOL = 1e-12 throughout for exact-arithmetic checks.
Calibration tolerances are wider (the reliability curve is a statistical
object; finite-ensemble sampling adds noise at the ~spread/√N scale).

Test structure
--------------
TestRangeSpaceCalibration  — zero null content: pred_std ≈ 0 everywhere,
                             n_pixels_calibrated = 0 after filter.
TestMonotonicRelationship  — well-calibrated oracle: r ≈ +1, slope ≈ 1.
TestMiscalibrationDetected — deliberately inverted (bias↑, spread↓ vs
                             bias↓, spread↑): r < 0, ECE >> threshold,
                             is_calibrated() returns False.  This is the
                             critical test: a metric that never says "bad"
                             is useless (R6).
TestEndToEnd               — full pipeline from SyntheticCase to
                             is_calibrated() verdict on both cases.

Miscalibration design
---------------------
3 null regions with (bias, spread) = (1.5, 0.5), (1.0, 1.0), (0.5, 1.5).
Sorted by predicted_std: bins go [0.5, 1.0, 1.5].
Corresponding actual errors:                    [1.5, 1.0, 0.5].
→ slope ≈ −1,  r ≈ −1,  ECE ≈ 0.67.
All three metrics detect miscalibration unambiguously.

Calibration thresholds used in is_calibrated() calls
------------------------------------------------------
  min_pearson_r = 0.9   (well-calibrated: r ≈ +1 >> 0.9)
  max_ece       = 0.3   (well-calibrated: ECE ≈ 0 << 0.3;
                          miscalibrated: ECE ≈ 0.67 >> 0.3)
  min_slope     = 0.5   (well-calibrated: slope ≈ 1 ∈ [0.5, 2.0])
  max_slope     = 2.0   (miscalibrated:   slope ≈ −1 < 0.5)
These are ALL explicit documented parameters of is_calibrated().
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from operators.inpaint import MaskOperator
from eval.synthetic import make_null_calibration_case, NullRegion
from layer.calibrate import calibrate, is_calibrated

TOL = 1e-12

# ---------------------------------------------------------------------------
# Shared calibration thresholds (all explicit — no magic numbers)
MIN_R    = 0.9    # min Pearson r for "calibrated"
MAX_ECE  = 0.3    # max ECE for "calibrated"
MIN_SLOP = 0.5    # min slope for "calibrated"
MAX_SLOP = 2.0    # max slope for "calibrated"

# ---------------------------------------------------------------------------
# Fixtures

def _mask_op(H: int = 12, W: int = 12, hole_rows: tuple[int, int] = (6, 12)) -> tuple:
    """Return (MaskOperator, mask array, x_shape) with rows[hole_rows] as holes."""
    mask = np.ones((H, W), dtype=np.float64)
    mask[hole_rows[0]:hole_rows[1], :] = 0.0
    return MaskOperator(mask), mask, (H, W)


def _well_calibrated_regions(H: int = 12, W: int = 12) -> list[NullRegion]:
    """3 hole regions with bias = spread = 0.5 / 1.0 / 1.5."""
    mask = np.zeros((H, W), dtype=bool)

    region_a = mask.copy(); region_a[6:8, :] = True
    region_b = mask.copy(); region_b[8:10, :] = True
    region_c = mask.copy(); region_c[10:12, :] = True

    return [
        NullRegion(mask=region_a, bias=0.5, spread=0.5),
        NullRegion(mask=region_b, bias=1.0, spread=1.0),
        NullRegion(mask=region_c, bias=1.5, spread=1.5),
    ]


def _miscalibrated_regions(H: int = 12, W: int = 12) -> list[NullRegion]:
    """3 hole regions with bias INVERTED relative to spread.

    Region A: small spread (0.5) but large bias (1.5) → overconfident.
    Region B: medium spread (1.0) and medium bias (1.0) → neutral.
    Region C: large spread (1.5) but small bias (0.5) → underconfident.

    When the reliability curve bins by predicted_std:
      bin 1 (pred ≈ 0.5): actual ≈ 1.5   (error >> uncertainty)
      bin 2 (pred ≈ 1.0): actual ≈ 1.0
      bin 3 (pred ≈ 1.5): actual ≈ 0.5   (error << uncertainty)

    → slope ≈ −1, r ≈ −1, ECE ≈ 0.67.
    """
    mask = np.zeros((H, W), dtype=bool)

    region_a = mask.copy(); region_a[6:8, :] = True
    region_b = mask.copy(); region_b[8:10, :] = True
    region_c = mask.copy(); region_c[10:12, :] = True

    return [
        NullRegion(mask=region_a, bias=1.5, spread=0.5),
        NullRegion(mask=region_b, bias=1.0, spread=1.0),
        NullRegion(mask=region_c, bias=0.5, spread=1.5),
    ]


N_ENSEMBLE = 400    # large enough that sample mean ≈ true mean to 0.1%
N_BINS     = 3      # one bin per region — clean separation guaranteed


# ---------------------------------------------------------------------------
class TestRangeSpaceCalibration:
    """Zero null content: all x_out_i are identical → pred_std = 0 everywhere."""

    def test_all_pred_std_zero(self):
        op, _, x_shape = _mask_op()
        x_gt = np.zeros(x_shape)
        y = op.forward(x_gt)

        from engine.oracle import OracleEngine
        from layer.decompose import rectify as lrectify
        oracle = OracleEngine(op)
        x_outs = []
        for _ in range(5):
            null_content = np.zeros(x_shape)
            x_hat = oracle.restore(y, null_content)
            x_outs.append(lrectify(x_hat, y, op).x_out)

        stack = np.stack(x_outs)
        pred_std = stack.std(axis=0)
        assert pred_std.max() <= TOL, (
            f"Expected zero pred_std everywhere with zero null content; "
            f"got max = {pred_std.max():.3e}"
        )

    def test_no_calibrated_pixels(self):
        op, _, x_shape = _mask_op()
        x_gt = np.zeros(x_shape)
        y = op.forward(x_gt)

        from engine.oracle import OracleEngine
        from layer.decompose import rectify as lrectify
        oracle = OracleEngine(op)
        x_outs = []
        for _ in range(5):
            x_hat = oracle.restore(y, np.zeros(x_shape))
            x_outs.append(lrectify(x_hat, y, op).x_out)

        result = calibrate(x_outs, x_gt, n_bins=N_BINS, min_predicted_std=1e-6)
        assert result.n_pixels_calibrated == 0

    def test_actual_error_zero_range_pixels(self):
        """Observed (range) pixels have actual_error = 0 to TOL with x_gt = 0."""
        op, mask, x_shape = _mask_op()
        x_gt = np.zeros(x_shape)
        y = op.forward(x_gt)

        from engine.oracle import OracleEngine
        from layer.decompose import rectify as lrectify
        oracle = OracleEngine(op)
        x_outs = []
        for _ in range(5):
            x_hat = oracle.restore(y, np.zeros(x_shape))
            x_outs.append(lrectify(x_hat, y, op).x_out)

        mean_x = np.stack(x_outs).mean(axis=0)
        actual_err = np.abs(mean_x - x_gt)
        # range pixels (mask = 1): actual error exactly 0
        assert actual_err[mask == 1].max() <= TOL


# ---------------------------------------------------------------------------
class TestMonotonicRelationship:
    """Well-calibrated oracle: pred_std and actual_error are proportional."""

    @pytest.fixture(scope="class")
    def well_calibrated_result(self):
        op, _, x_shape = _mask_op()
        regions = _well_calibrated_regions()
        case = make_null_calibration_case(op, x_shape, regions, N_ENSEMBLE, seed=0)
        return calibrate(case.x_outs, case.x_gt, n_bins=N_BINS, min_predicted_std=1e-6)

    def test_n_pixels_calibrated_positive(self, well_calibrated_result):
        assert well_calibrated_result.n_pixels_calibrated > 0

    def test_reliability_curve_has_3_bins(self, well_calibrated_result):
        assert len(well_calibrated_result.bin_predicted_std) == N_BINS

    def test_positive_pearson_r(self, well_calibrated_result):
        r = well_calibrated_result.pearson_r
        assert r > 0.9, f"Expected r > 0.9 for well-calibrated oracle; got {r:.4f}"

    def test_slope_near_one(self, well_calibrated_result):
        s = well_calibrated_result.slope
        assert 0.5 <= s <= 2.0, f"Expected slope in [0.5, 2.0]; got {s:.4f}"

    def test_low_ece(self, well_calibrated_result):
        e = well_calibrated_result.ece
        assert e < 0.3, f"Expected ECE < 0.3 for well-calibrated oracle; got {e:.4f}"

    def test_bin_predicted_stds_increasing(self, well_calibrated_result):
        bp = well_calibrated_result.bin_predicted_std
        assert np.all(np.diff(bp) > 0), (
            f"Bins should be sorted by predicted_std; got {bp}"
        )

    def test_bin_actual_errors_roughly_match_predicted(self, well_calibrated_result):
        """Each bin's actual_error should be within 30% of predicted_std."""
        bp = well_calibrated_result.bin_predicted_std
        ba = well_calibrated_result.bin_actual_error
        for i, (p, a) in enumerate(zip(bp, ba)):
            assert abs(a - p) < 0.3 * p + 0.1, (
                f"Bin {i}: actual_error={a:.3f} is too far from predicted_std={p:.3f}"
            )

    def test_is_calibrated_returns_true(self, well_calibrated_result):
        assert is_calibrated(
            well_calibrated_result,
            min_pearson_r=MIN_R,
            max_ece=MAX_ECE,
            min_slope=MIN_SLOP,
            max_slope=MAX_SLOP,
        ), "Expected is_calibrated() = True for well-calibrated oracle"


# ---------------------------------------------------------------------------
class TestMiscalibrationDetected:
    """Critical test (R6): the metric MUST flag the miscalibrated case as bad.

    A calibration metric that always returns True is useless — this test
    proves it has teeth.

    Construction: bias and spread are INVERTED across the 3 regions.
    The oracle that reports high uncertainty (spread) in region C is actually
    most accurate there (small bias from x_gt), while region A (low spread,
    high confidence) has the largest actual error.

    Expected outcome:
      r     ≈ −1    (uncertainty ANTI-predicts error)
      slope ≈ −1    (below MIN_SLOP = 0.5)
      ECE   ≈ 0.67  (above MAX_ECE = 0.3)
      is_calibrated() → False
    """

    @pytest.fixture(scope="class")
    def bad_result(self):
        op, _, x_shape = _mask_op()
        regions = _miscalibrated_regions()
        case = make_null_calibration_case(op, x_shape, regions, N_ENSEMBLE, seed=1)
        return calibrate(case.x_outs, case.x_gt, n_bins=N_BINS, min_predicted_std=1e-6)

    def test_n_pixels_calibrated_positive(self, bad_result):
        assert bad_result.n_pixels_calibrated > 0

    def test_pearson_r_negative(self, bad_result):
        r = bad_result.pearson_r
        assert r < 0.0, (
            f"Miscalibrated oracle must have r < 0 (uncertainty anti-predicts error); "
            f"got r = {r:.4f}"
        )

    def test_slope_negative(self, bad_result):
        s = bad_result.slope
        assert s < 0.0, (
            f"Miscalibrated oracle must have slope < 0; got slope = {s:.4f}"
        )

    def test_ece_high(self, bad_result):
        e = bad_result.ece
        assert e > 0.3, (
            f"Miscalibrated oracle must have ECE > 0.3; got ECE = {e:.4f}"
        )

    def test_is_calibrated_returns_false(self, bad_result):
        """THE critical assertion: the metric flags the bad case as bad."""
        verdict = is_calibrated(
            bad_result,
            min_pearson_r=MIN_R,
            max_ece=MAX_ECE,
            min_slope=MIN_SLOP,
            max_slope=MAX_SLOP,
        )
        assert not verdict, (
            "is_calibrated() returned True for the deliberately miscalibrated oracle. "
            "A metric that cannot detect this miscalibration is useless (R6). "
            f"r={bad_result.pearson_r:.4f}, ECE={bad_result.ece:.4f}, "
            f"slope={bad_result.slope:.4f}"
        )

    def test_bin_actual_errors_decreasing_while_predicted_increasing(self, bad_result):
        """The inversion must be visible directly in the curve."""
        bp = bad_result.bin_predicted_std
        ba = bad_result.bin_actual_error
        # bins sorted by predicted_std (ascending)
        assert np.all(np.diff(bp) > 0), f"bins not sorted: {bp}"
        # actual errors must be DECREASING (inverted relationship)
        assert np.all(np.diff(ba) < 0), (
            f"Expected actual errors to decrease as predicted_std increases "
            f"(inverted oracle), got: {ba}"
        )


# ---------------------------------------------------------------------------
class TestEndToEnd:
    """Full pipeline: SyntheticCase → calibrate() → is_calibrated() verdict."""

    def test_well_calibrated_verdict(self):
        op, _, x_shape = _mask_op()
        regions = _well_calibrated_regions()
        case = make_null_calibration_case(op, x_shape, regions, N_ENSEMBLE, seed=42)
        result = calibrate(case.x_outs, case.x_gt, n_bins=N_BINS, min_predicted_std=1e-6)
        assert is_calibrated(result, MIN_R, MAX_ECE, MIN_SLOP, MAX_SLOP)

    def test_miscalibrated_verdict(self):
        op, _, x_shape = _mask_op()
        regions = _miscalibrated_regions()
        case = make_null_calibration_case(op, x_shape, regions, N_ENSEMBLE, seed=99)
        result = calibrate(case.x_outs, case.x_gt, n_bins=N_BINS, min_predicted_std=1e-6)
        assert not is_calibrated(result, MIN_R, MAX_ECE, MIN_SLOP, MAX_SLOP)

    def test_different_seeds_same_verdict(self):
        """Verdict is robust to seed choice (statistical stability check)."""
        op, _, x_shape = _mask_op()
        regions = _well_calibrated_regions()
        for seed in (0, 7, 123):
            case = make_null_calibration_case(op, x_shape, regions, N_ENSEMBLE, seed=seed)
            result = calibrate(case.x_outs, case.x_gt, n_bins=N_BINS, min_predicted_std=1e-6)
            assert is_calibrated(result, MIN_R, MAX_ECE, MIN_SLOP, MAX_SLOP), (
                f"Well-calibrated oracle failed is_calibrated at seed={seed}"
            )

    def test_miscalibrated_different_seeds_always_detected(self):
        op, _, x_shape = _mask_op()
        regions = _miscalibrated_regions()
        for seed in (1, 8, 200):
            case = make_null_calibration_case(op, x_shape, regions, N_ENSEMBLE, seed=seed)
            result = calibrate(case.x_outs, case.x_gt, n_bins=N_BINS, min_predicted_std=1e-6)
            assert not is_calibrated(result, MIN_R, MAX_ECE, MIN_SLOP, MAX_SLOP), (
                f"Miscalibrated oracle passed is_calibrated at seed={seed}"
            )

    def test_x_gt_not_modified(self):
        """x_gt passed to calibrate() must be read-only; it must not be altered."""
        op, _, x_shape = _mask_op()
        regions = _well_calibrated_regions()
        case = make_null_calibration_case(op, x_shape, regions, N_ENSEMBLE, seed=5)
        x_gt_copy = case.x_gt.copy()
        calibrate(case.x_outs, case.x_gt, n_bins=N_BINS, min_predicted_std=1e-6)
        assert np.array_equal(case.x_gt, x_gt_copy), "calibrate() must not modify x_gt"

    def test_calibrate_requires_at_least_2_members(self):
        op, _, x_shape = _mask_op()
        x_gt = np.zeros(x_shape)
        x_outs = [np.zeros(x_shape)]
        with pytest.raises(ValueError, match="at least 2"):
            calibrate(x_outs, x_gt)
