"""Synthetic byte/role checks only; these tests do not qualify physical skill."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.sim.contracts import hash_bytes
from scripts.rsi_read_mesh_contact_native_pilot import bounded_profile, verify_release_prefix
from scripts.rsi_swing_release_ablation_pilot import declared_jobs


def fixture():
    original = dict(
        force_n=np.zeros((300, 1, 6)),
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
        ball_position_after_step_m=np.zeros((300, 1, 3)),
        taskspace_joint_increment_rad=np.zeros((300, 1, 29)),
    )
    original["force_n"][74, 0, 0] = 20
    original["force_n"][99, 0, 4] = 10
    original["taskspace_joint_increment_rad"][74:95] = 0.1
    new = {k: v.copy() for k, v in original.items()}
    new["taskspace_joint_increment_rad"][75:] = 0
    new["force_n"][99] = 0
    new["ball_position_after_step_m"][134, 0, 0] = 2
    baseline_hash = "sha256:" + "a" * 64
    prefix = dict(
        schema="soccer.rsi.first_contact_prefix_parity.v1",
        baseline_receipt_hash=baseline_hash,
        prefix_frames=75,
        prefix_array_hashes={k: hash_bytes(v[:75].tobytes()) for k, v in new.items()},
        all_original_array_prefixes_bit_equal=True,
    )
    return (
        new,
        original,
        label_first_touch(original).to_dict(),
        label_first_touch(new),
        prefix,
        baseline_hash,
    )


def test_same_fixed_four_jobs_independently_declared():
    spec = bounded_profile("release-four")
    assert spec["jobs"] == declared_jobs([SimpleNamespace(context_index=i) for i in range(8)])
    assert spec["count"] == 4 and spec["groups"] == [8, 56, 64, 65]


def test_unsafe_baseline_and_postcontact_change_are_preserved():
    args = fixture()
    assert args[2]["safe"] is False and args[3].task_pass
    verify_release_prefix(*args)


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "baseline_receipt_hash",
        "prefix_frames",
        "prefix_array_hashes",
        "all_original_array_prefixes_bit_equal",
    ],
)
def test_resealed_prefix_assertion_not_enough(field):
    args = list(fixture())
    args[4] = copy.deepcopy(args[4])
    args[4][field] = None
    with pytest.raises(ValueError):
        verify_release_prefix(*args)


@pytest.mark.parametrize(
    "tamper",
    [
        "first_contact",
        "other_body_first",
        "dtype",
        "array_missing",
        "prefix_value",
        "baseline_outcome",
        "no_contact",
    ],
)
def test_actual_original_bytes_and_contact_clock_required(tamper):
    args = list(fixture())
    if tamper == "first_contact":
        args[0]["force_n"][73, 0, 0] = 1.1
    elif tamper == "other_body_first":
        args[1]["force_n"][73, 0, 4] = 2
    elif tamper == "dtype":
        args[0]["taskspace_joint_increment_rad"] = args[0]["taskspace_joint_increment_rad"].astype(
            np.float32
        )
    elif tamper == "array_missing":
        del args[0]["taskspace_joint_increment_rad"]
    elif tamper == "prefix_value":
        args[0]["taskspace_joint_increment_rad"][74] += 0.01
    elif tamper == "no_contact":
        args[2]["first_foot_contact_frame"] = None
    else:
        args[2]["safe"] = True
    with pytest.raises(ValueError):
        verify_release_prefix(*args)
