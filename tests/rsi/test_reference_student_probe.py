"""Synthetic bounded probe contracts; no real dynamics, Foundation or training."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.reference_student_execution import PREFIX
from rosclaw_soccer.rsi.reference_student_probe import (
    JOBS,
    SEED,
    measured_authority,
    probe_models,
    require_parity,
)
from scripts.rsi_reference_student_three_pilot import require_scope


def config():
    return dict(
        reference_probe_profile="three-body-interface-authority",
        groups=[76],
        probe_seed=SEED,
        probe_joint_index=4,
        probe_head_bias=0.2,
        max_attempts=3,
        new_physical_budget=3,
        new_independent_cpu_budget=3,
        combined_body_substep_budget=18000,
        max_own_full_nn_workers=2,
        mode="SIM_ONLY",
        hardware_authorized=False,
        promotion_authorized=False,
        private_fresh_accessed=False,
        neural_training_authorized=False,
        teacher_auto_admission=False,
        generated_seed_episode_binding=False,
        use_mesh_surface_swing=False,
        suppress_postcontact_swing_release=False,
        use_contact_exit_torque_mpc=False,
        use_directional_contact_compliance=False,
        required_receipt_witnesses=["intent_witness", "parity_witness"],
    )


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "extra",
        "seed",
        "joint",
        "bias",
        "budget",
        "workers",
        "truthy",
        "training",
        "generated",
        "witness",
    ],
)
def test_fixed_single_predeclared_numeric_probe_is_not_parameter_search(tamper):
    cfg = config()
    if tamper == "extra":
        cfg["groups"].append(77)
    elif tamper == "seed":
        cfg["probe_seed"] += 1
    elif tamper == "joint":
        cfg["probe_joint_index"] = 10
    elif tamper == "bias":
        cfg["probe_head_bias"] = -0.2
    elif tamper == "budget":
        cfg["max_attempts"] = 4
    elif tamper == "workers":
        cfg["max_own_full_nn_workers"] = 3
    elif tamper == "truthy":
        cfg["new_independent_cpu_budget"] = True
    elif tamper == "training":
        cfg["neural_training_authorized"] = True
    elif tamper == "generated":
        cfg["generated_seed_episode_binding"] = True
    elif tamper == "witness":
        cfg["required_receipt_witnesses"].pop()
    if tamper:
        with pytest.raises(ValueError):
            require_scope(cfg)
    else:
        require_scope(cfg)
        assert [j["model_index"] for j in JOBS] == [None, 0, 1]


def test_both_models_share_every_numeric_parameter_except_one_fixed_ankle_bias():
    zero, nonzero = probe_models("sha256:" + "a" * 64, "sha256:" + "b" * 64)
    assert zero["parameters"]["head_bias"] == [0.0] * 12
    expected = copy.deepcopy(zero["parameters"])
    expected["head_bias"][4] = 0.2
    assert nonzero["parameters"] == expected
    assert not zero["training_provenance_authenticated"] and not nonzero["physics_qualified"]


def trace():
    value = {f"field{i}": np.zeros((300, 1, 1), dtype=np.float64) for i in range(30)}
    value.update(
        {
            k: np.zeros((300, 1, 29), dtype=np.float64)
            for k in ("pre_motor_joint_target_rad", "joint_target_rad", "joint_position_rad")
        }
    )
    return value


@pytest.mark.parametrize("tamper", [None, "missing", "dtype", "tail", "prefix", "frames"])
def test_off_full_arrays_and_pre_action_prefix_are_distinct_claims(tamper):
    original, current = trace(), trace()
    if tamper == "missing":
        current.pop("field0")
    elif tamper == "dtype":
        current["field0"] = current["field0"].astype(np.float32)
    elif tamper == "tail":
        current["field0"][100, 0, 0] = 1e-12
    elif tamper == "prefix":
        current["field0"][29, 0, 0] = 1e-12
    elif tamper == "frames":
        current["field0"] = current["field0"][:299]
    if tamper:
        with pytest.raises(ValueError):
            require_parity(current, original, 300)
        if tamper == "tail":
            assert len(require_parity(current, original, 30)) == 33
    else:
        assert len(require_parity(current, original, 300)) == 33


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_reference_only_without_parent_or_body_response_is_not_control_authority(level):
    zero, nonzero = trace(), trace()
    for t in (zero, nonzero):
        t[PREFIX + "applied_reference_increment_rad"] = np.zeros((300, 1, 12))
    if level >= 1:
        nonzero[PREFIX + "applied_reference_increment_rad"][30:, 0, 4] = 0.05
    if level >= 2:
        nonzero["pre_motor_joint_target_rad"][30:, 0, 4] = 0.05
    if level >= 3:
        nonzero["joint_target_rad"][30:, 0, 4] = 0.04
    if level >= 4:
        nonzero["joint_position_rad"][31:, 0, 4] = 0.03
    result = measured_authority(zero, nonzero)
    assert result["reference_control_authority_observed"] is (level == 4)
    assert not result["probe_is_trained_policy"] and not result["success_improvement_claimed"]
    assert not result["teacher_qualified"]
