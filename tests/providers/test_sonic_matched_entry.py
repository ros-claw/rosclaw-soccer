"""Reference interpolation fixtures, never dynamic G1 skill evidence."""

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_matched_entry import (
    InfeasibleMatchedBridge,
    prepare_matched_entry_pair,
)
from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference


def inputs():
    entry = np.zeros(36, dtype=np.float64)
    entry[2], entry[3] = 0.793, 1
    source = np.repeat(entry[None], 60, axis=0)
    source[:, 7:] = 0.1 + np.arange(60)[:, None] / 120 * 0.01
    v = np.zeros(41, dtype=np.float64)
    v[6:35] = 0.2
    limits = np.tile([-3.0, 3.0], (29, 1))
    return source, entry, v, limits


def prepare(args, evidence="sha256:" + "1" * 64):
    return prepare_matched_entry_pair(*args, source_evidence_hash=evidence)


def test_same_time_exact_velocity_untouched_root_wrist_and_source():
    args = inputs()
    copies = [a.copy() for a in args]
    pair = prepare(args)
    n = pair.entry_intervals
    assert n == 10
    p, v = np.asarray(pair.position.poses), np.asarray(pair.velocity.poses)
    ids = np.array([i for i, name in enumerate(G1_DDS_JOINT_NAMES) if "wrist" not in name])
    untouched = [i for i in range(36) if i not in 7 + ids]
    np.testing.assert_array_equal(p[:, untouched], v[:, untouched])
    np.testing.assert_array_equal(v[n:], p[n:])
    np.testing.assert_array_equal(v[0], args[1])
    np.testing.assert_allclose(
        (v[1, 7 + ids] - v[0, 7 + ids]) / 0.02, args[2][6:35][ids], atol=1e-12
    )
    np.testing.assert_allclose(
        (v[n, 7 + ids] - v[n - 1, 7 + ids]) / 0.02,
        (v[n + 1, 7 + ids] - v[n, 7 + ids]) / 0.02,
        atol=1e-12,
    )
    assert pair.maximum_joint_correction_rad <= 0.12
    assert (
        pair.velocity_initial_velocity_gap_rad_s < 1e-10 < pair.position_initial_velocity_gap_rad_s
    )
    for a, old in zip(args, copies, strict=True):
        np.testing.assert_array_equal(a, old)
    assert not pair.report("matched_velocity")["dynamics_qualified"]
    assert pair.report("matched_velocity")["matched_joint_count"] == 23


def test_original_preparation_is_the_same_time_position_control():
    source, q, _, _ = inputs()
    pair = prepare(inputs())
    old = prepare_sonic_reference(
        source,
        q,
        source_frequency_hz=120,
        entry_intervals=pair.entry_intervals,
        source_evidence_hash="sha256:" + "1" * 64,
    )
    assert pair.position == old.reference
    assert pair.source_preparation_hash == old.contract_hash


def test_wrist_velocity_not_silently_applied_to_quarantined_channels():
    args = inputs()
    first = prepare(args)
    wrist = [i for i, n in enumerate(G1_DDS_JOINT_NAMES) if "wrist" in n]
    args[2][6 + np.array(wrist)] = 9
    second = prepare(args)
    assert first.entry_intervals == second.entry_intervals
    np.testing.assert_array_equal(first.velocity.poses, second.velocity.poses)


def test_unavailable_finite_proposal_never_clips_pose_or_velocity():
    args = inputs()
    args[2][6:35] = 3000
    old = args[2].copy()
    with pytest.raises(InfeasibleMatchedBridge, match="NO_FEASIBLE"):
        prepare(args)
    np.testing.assert_array_equal(args[2], old)


@pytest.mark.parametrize("kind", ["nan", "dtype", "shape", "limits", "source"])
def test_invalid_inputs_are_fatal_not_an_unavailable_candidate(kind):
    args = list(inputs())
    if kind == "nan":
        args[2][0] = np.nan
    elif kind == "dtype":
        args[2] = args[2].astype(np.float32)
    elif kind == "shape":
        args[2] = args[2][:35]
    elif kind == "limits":
        args[3][0] = [1, -1]
    else:
        args[0][0, 3] = 0
    with pytest.raises(ValueError) as error:
        prepare(args)
    assert not isinstance(error.value, InfeasibleMatchedBridge)


def test_invalid_mode_and_nonzero_contact_rejected_before_body_access():
    from pathlib import Path

    from rosclaw_soccer.rsi.full_entry_forecast import FullEntryForecaster, run_entry_suffix

    common = dict(
        model_root=Path("unused"),
        source_poses=np.empty((0, 36)),
        source_evidence_hash="unused",
        affordance=None,
        law=None,
    )
    for mode, offset in [("unknown", (0.0, 0.0, 0.0)), ("matched_velocity", (0.0, -0.025, -0.025))]:
        with pytest.raises(ValueError, match="mode|zero C"):
            run_entry_suffix(
                None, None, entry_reference_mode=mode, contact_reference_offset_m=offset, **common
            )
        f = FullEntryForecaster(world_hash="unused")
        with pytest.raises(ValueError, match="mode|zero C"):
            f.forecast(
                None, None, entry_reference_mode=mode, contact_reference_offset_m=offset, **common
            )
        assert f.integrated_native_steps == f.attempted_native_steps == 0
