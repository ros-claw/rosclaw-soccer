"""Synthetic numeric causality/guards; no Foundation, physics or training."""

import argparse
import ast
import copy
from pathlib import Path

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi import native_first_touch_episode as native
from rosclaw_soccer.rsi import reference_student_execution as execution
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from tests.rsi.test_first_touch_qualification import fixture as qualification_fixture
from tests.rsi.test_pre_parent_reference_features import measured_trace
from tests.rsi.test_swing_release_default_arithmetic import OffPath

WORLD, PARENT = "sha256:" + "a" * 64, "sha256:" + "b" * 64


def model():
    parameters = initial_parameters(135, 12, hidden_dimension=64, seed=23)
    parameters["head_bias"] = [0.05] * 12  # Numeric probe, not learned weights.
    return execution.make_model(
        parameters,
        mean=np.zeros(135),
        scale=np.ones(135),
        world_hash=WORLD,
        parent_model_hash=PARENT,
    )


def captured():
    value, trace = model(), measured_trace()
    decoder = execution.decoder_from_model(value, world_hash=WORLD, parent_model_hash=PARENT)
    commitment = dict(
        compiled_model_hash=WORLD,
        reference_student_model=value,
        reference_student_contract=decoder.contract(),
        bounded_control_profile=BoundedFirstTouchControl().contract(),
        foundation_observation_capture={},
    )
    trace["pre_motor_joint_target_rad"] = trace["foundation_navigation_joint_target_rad"].copy()
    trace["taskspace_joint_increment_rad"] = np.zeros((300, 1, 29))
    limits = np.tile([-3.0, 3.0], (29, 1))
    records = {k: [] for k in execution.FIELDS}
    for frame in range(300):
        base = trace["foundation_navigation_joint_target_rad"][frame, 0]
        record = decoder.advance(trace, frame=frame, navigation_target=base, joint_limits=limits)
        trace["pre_motor_joint_target_rad"][frame, 0] = record["pre_parent_joint_target_rad"]
        trace["taskspace_joint_increment_rad"][frame, 0] = (
            record["pre_parent_joint_target_rad"] - base
        )
        for key in records:
            records[key].append(record[key][None])
    trace.update({execution.PREFIX + k: np.asarray(v) for k, v in records.items()})
    return commitment, trace, limits


def test_owned_numeric_model_and_reconstruction_do_not_claim_training_or_physics():
    value = model()
    owned = execution.own_model(value)
    value["parameters"]["head_bias"][0] = 8
    assert owned["parameters"]["head_bias"][0] == 0.05
    commitment, trace, limits = captured()
    review = execution.reconstruct(commitment, trace, joint_limits=limits, parent_model_hash=PARENT)
    assert review["frames_reconstructed"] == 300
    assert review["own_hidden_and_applied_reference_reconstructed"]
    assert not review["independent_neural_implementation"]
    assert not review["training_provenance_authenticated"] and not review["teacher_qualified"]
    assert not commitment["reference_student_model"]["physics_qualified"]
    assert np.any(trace[execution.PREFIX + "applied_reference_increment_rad"][30:])


@pytest.mark.parametrize(
    "tamper", ["weights", "source", "features", "permission", "clock", "world", "parent"]
)
def test_model_identity_cannot_be_retagged_or_silently_reused(tamper):
    value = model()
    if tamper == "weights":
        value["parameters"]["head_bias"][0] = 0.06
    elif tamper == "source":
        value["source_hashes"]["decoder"] = "other"
    elif tamper == "features":
        value["feature_names"][-1] = "teacher_future_contact_phase"
    elif tamper == "permission":
        value["hardware_authorized"] = True
    elif tamper == "clock":
        value["decoder_contract"]["control_dt_s"] = 0.002
    with pytest.raises(ValueError):
        execution.decoder_from_model(
            value,
            world_hash=PARENT if tamper == "world" else WORLD,
            parent_model_hash=WORLD if tamper == "parent" else PARENT,
        )


@pytest.mark.parametrize(
    "tamper", ["missing", "dtype", "nan", "hidden", "previous", "target", "mixed", "report"]
)
def test_actual_captured_reference_clock_and_causal_parent_input_must_reconstruct(tamper):
    commitment, trace, limits = captured()
    report = copy.deepcopy(commitment)
    if tamper == "missing":
        trace.pop(execution.PREFIX + "input_features")
    elif tamper == "dtype":
        key = execution.PREFIX + "recurrent_hidden_state"
        trace[key] = trace[key].astype(np.float32)
    elif tamper == "nan":
        trace[execution.PREFIX + "normalized_input_features"][40, 0, 0] = np.nan
    elif tamper == "hidden":
        trace[execution.PREFIX + "recurrent_hidden_state"][100, 0, 0] += 0.001
    elif tamper == "previous":
        trace[execution.PREFIX + "applied_reference_increment_rad"][40, 0, 0] = 0
    elif tamper == "target":
        trace["pre_motor_joint_target_rad"][40, 0, 0] += 0.001
        trace["taskspace_joint_increment_rad"] = (
            trace["pre_motor_joint_target_rad"] - trace["foundation_navigation_joint_target_rad"]
        )
    elif tamper == "mixed":
        commitment["mesh_surface_swing"] = {}
    elif tamper == "report":
        report["reference_student_model"] = {}
    with pytest.raises(ValueError):
        execution.require_capture(report, commitment, trace)
        execution.reconstruct(commitment, trace, joint_limits=limits, parent_model_hash=PARENT)


def test_kernel_rejects_unmixed_request_before_body_output(tmp_path):
    with pytest.raises(ValueError, match="unmixed"):
        native.run_native_first_touch_episode(
            argparse.Namespace(output_root=tmp_path / "absent"),
            course=(2.65, 0.1, 0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            reference_student_model=model(),
        )
    assert not (tmp_path / "absent").exists()


def test_off_path_projection_cannot_hide_a_default_enabled_student():
    function = next(
        n
        for n in ast.parse(Path(native.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_native_first_touch_episode"
    )
    index = next(
        i for i, a in enumerate(function.args.kwonlyargs) if a.arg == "reference_student_model"
    )
    function.args.kw_defaults[index] = ast.Dict(keys=[], values=[])
    with pytest.raises(AssertionError):
        OffPath().visit(function)


def test_old_qualification_cannot_silently_admit_reference_student():
    commitment, trace, limits = captured()
    reference = execution.reconstruct(
        commitment, trace, joint_limits=limits, parent_model_hash=PARENT
    )
    review, foundation, outcome = qualification_fixture()
    review.update(
        reference_student_review=reference,
        actual_mujoco_dynamics_replayed=True,
        actual_pd_torque_reconstructed=True,
        neural_target_reconstructed=True,
    )
    with pytest.raises(ValueError, match="unexpected reference"):
        require_first_touch_qualification(review, foundation, outcome)
    result = require_first_touch_qualification(
        review,
        foundation,
        outcome,
        reference_student_model_hash=commitment["reference_student_model"]["model_hash"],
    )
    assert result["m0_task_pass"] and not result["promotion_authorized"]
    review["reference_student_review"]["frames_reconstructed"] = 299
    with pytest.raises(ValueError, match="whole own-state"):
        require_first_touch_qualification(
            review,
            foundation,
            outcome,
            reference_student_model_hash=commitment["reference_student_model"]["model_hash"],
        )


def test_null_reference_tag_is_not_an_ordinary_qualification():
    review, foundation, outcome = qualification_fixture()
    review["reference_student_review"] = None
    with pytest.raises(ValueError, match="unexpected reference"):
        require_first_touch_qualification(review, foundation, outcome)
