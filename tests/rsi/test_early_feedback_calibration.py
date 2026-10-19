"""Protocol and three actual capture comparisons, not G1 skill evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.physics.g1_early_feedback_forecast import G1EarlyFeedbackForecast
from scripts.rsi_early_feedback_calibration import preserve_and_compare, require_scope
from tests.rsi.test_feedback_branch_calibration_scope import scope as original_scope


def scope():
    cfg = original_scope()
    cfg.update(
        profile="six-zero-early-feedback80ms-v1",
        frame_selection="RETROSPECTIVE_CONTACT_MINUS_TWO_DIAGNOSTIC_ONLY",
        horizon_substeps=40,
        feedback_forecast_substep_budget=480,
        combined_substep_budget=18480,
        branch_encoder_decoder_pairs=36,
        contact_frame_lookback=2,
        offset_duration_substeps=30,
    )
    return cfg


def test_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_no_missing_or_widened_protocol(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    v = cfg[key]
    cfg[key] = int(v) if type(v) is bool else float(v) if type(v) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)


def raw():
    records, neural = [], []
    names = dict(
        navigation_target_rad="foundation_navigation_joint_target_rad",
        taskspace_increment_rad="taskspace_joint_increment_rad",
        motor_delta_rad="motor_delta_rad",
        joint_target_rad="joint_target_rad",
        canonical_qpos="canonical_qpos",
        canonical_qvel="canonical_qvel",
    )
    sizes = (29, 29, 12, 29, 43, 41)
    trace = {k: np.zeros((34, 1, n)) for k, n in zip(names.values(), sizes, strict=True)}
    trace["foundation_neural_a"] = np.zeros((34, 1, 2), dtype=np.float32)
    for j in range(1, 4):
        r = {k: np.full(n, j * 0.01) for k, n in zip(names, sizes, strict=True)}
        for k, source in names.items():
            trace[source][30 + j, 0] = r[k]
        records.append(r)
        neural.append(dict(a=np.zeros((1, 2), dtype=np.float32)))
    out = G1EarlyFeedbackForecast(
        "sha256:" + "0" * 64,
        np.zeros(100),
        {"requested_offset_nm": np.zeros((40, 29))},
        tuple(records),
        tuple(neural),
    )
    return out, trace, {"a": 2}


def test_three_updated_targets_and_nn_captures_are_saved_and_compared(tmp_path):
    out, trace, widths = raw()
    row = preserve_and_compare(tmp_path, 0, 0, out, None, trace, 30, widths)
    assert row["sha256"].startswith("sha256:")
    preserve_and_compare(tmp_path, 0, 1, copy.deepcopy(out), out, trace, 30, widths)
    with np.load(row["path"], allow_pickle=False) as a:
        for j in range(1, 4):
            np.testing.assert_array_equal(a[f"next_{j}_joint_target_rad"], j * 0.01)
            assert a[f"neural_next_{j}_a"].dtype == np.float32


@pytest.mark.parametrize("index", range(3))
def test_no_skipped_intermediate_neural_bits_and_save_before_rejection(tmp_path, index):
    out, trace, widths = raw()
    out.next_neural_records[index]["a"][0, 0] = -0.0
    with pytest.raises(ValueError, match="neural bits"):
        preserve_and_compare(tmp_path, 0, 0, out, None, trace, 30, widths)
    assert (tmp_path / "case-0-branch-0.npz").is_file()


@pytest.mark.parametrize("index", range(3))
def test_repeated_zero_matches_all_three_records_not_just_final(tmp_path, index):
    out, trace, widths = raw()
    original = copy.deepcopy(out)
    out.next_target_records[index]["motor_delta_rad"][0] += 0.001
    with pytest.raises(ValueError, match="zero branches differ"):
        preserve_and_compare(tmp_path, 0, 0, out, original, trace, 30, widths)


def test_zero_calibration_rejects_nonzero_torque(tmp_path):
    out, trace, widths = raw()
    out.arrays["requested_offset_nm"][0, 0] = 0.001
    with pytest.raises(ValueError, match="never request nonzero"):
        preserve_and_compare(tmp_path, 0, 0, out, None, trace, 30, widths)
