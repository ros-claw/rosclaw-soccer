"""Synthetic command provenance, not a physical authority or success result."""

import argparse
import ast
import copy
from pathlib import Path

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi import native_first_touch_episode as native
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.planner_navigation_pulse import PlannerNavigationPulse
from rosclaw_soccer.rsi.planner_navigation_pulse_evidence import (
    FIELDS,
    PREFIX,
    TAG,
    require_navigation_pulse_capture,
    require_navigation_pulse_request,
    require_pulse_physical_parity,
)
from rosclaw_soccer.rsi.reference_student_execution import make_model
from tests.rsi.test_first_touch_qualification import fixture as qualification_fixture
from tests.rsi.test_swing_release_default_arithmetic import OffPath


def zero_model():
    return make_model(
        initial_parameters(135, 12, hidden_dimension=64, seed=2026101902),
        mean=np.zeros(135),
        scale=np.ones(135),
        world_hash="sha256:" + "a" * 64,
        parent_model_hash="sha256:" + "b" * 64,
    )


def captured(delta=-0.08, first_contact=None):
    pulse, profile = PlannerNavigationPulse(delta), BoundedFirstTouchControl()
    commitment = {
        TAG: pulse.contract(),
        "bounded_control_profile": profile.contract(),
        "reference_student_model": zero_model(),
    }
    trace = {PREFIX + k: np.zeros((300, 1, 3)) for k in FIELDS}
    for name, width in (
        ("navigation_command", 3),
        ("approach_requested_navigation_command", 3),
        ("root_pose_xyzw_m", 7),
        ("ball_position_before_step_m", 3),
        ("ball_linear_velocity_before_step_m_s", 3),
        ("force_n", 6),
        ("reference_student_requested_reference_increment_rad", 12),
        ("reference_student_applied_reference_increment_rad", 12),
    ):
        trace[name] = np.zeros((300, 1, width))
    trace["ball_position_before_step_m"][:, 0, :2] = [1.2, 0.1]
    trace["ball_linear_velocity_before_step_m_s"][:, 0, 0] = 0.5
    if first_contact is not None:
        # Any robot contact changes native navigation; unsafe cases are kept.
        trace["force_n"][first_contact, 0, 3] = 2.0
    for f in range(300):
        raw, base = profile.navigation(
            gap_x=1.2,
            gap_y=0.1,
            ball_vx=0.5,
            before_contact=first_contact is None or f <= first_contact,
        )
        residual, actual = pulse.compose(f, base)
        trace["approach_requested_navigation_command"][f, 0] = raw
        trace["navigation_command"][f, 0] = actual
        for name, value in zip(FIELDS, (base, residual, actual), strict=True):
            trace[PREFIX + name][f, 0] = value
    return commitment, trace


@pytest.mark.parametrize("delta", [0.0, -0.08, 0.08])
@pytest.mark.parametrize("contact", [None, 110, 151])
def test_reconstructs_original_b2_plus_fixed_pulse_without_calling_producer(
    delta, contact, monkeypatch
):
    c, trace = captured(delta, contact)
    before = copy.deepcopy(trace)
    monkeypatch.setattr(
        PlannerNavigationPulse, "compose", lambda *a: pytest.fail("producer composition reused")
    )
    result = require_navigation_pulse_capture(c, c, trace)
    assert result["frames_reconstructed"] == 300
    assert result["entire_leg_reference_strictly_zero"]
    assert not result["planner_and_encoder_effect_verified_here"]
    for k in trace:
        np.testing.assert_array_equal(trace[k], before[k])


@pytest.mark.parametrize(
    "tamper",
    [
        "missing",
        "extra",
        "partial",
        "nan",
        "raw",
        "baseline",
        "residual",
        "nav",
        "yaw",
        "leg",
        "tag",
        "report",
        "profile",
    ],
)
def test_orphan_or_unreconstructed_interventions_are_rejected(tamper):
    c, t = captured()
    report = copy.deepcopy(c)
    if tamper == "missing":
        t.pop(PREFIX + FIELDS[0])
    elif tamper == "extra":
        t[PREFIX + "unknown"] = np.zeros((300, 1, 1))
    elif tamper == "partial":
        t["force_n"] = t["force_n"][:-1]
    elif tamper == "nan":
        t[PREFIX + FIELDS[0]][20, 0, 0] = np.nan
    elif tamper in ("raw", "baseline", "residual", "nav", "yaw"):
        key = {
            "raw": "approach_requested_navigation_command",
            "baseline": PREFIX + FIELDS[0],
            "residual": PREFIX + FIELDS[1],
            "nav": "navigation_command",
            "yaw": "navigation_command",
        }[tamper]
        t[key][110, 0, 2 if tamper == "yaw" else 1] += 0.01
    elif tamper == "leg":
        t["reference_student_applied_reference_increment_rad"][200, 0, 0] = 1e-12
    elif tamper == "tag":
        c.pop(TAG)
    elif tamper == "report":
        report[TAG] = {}
    else:
        c["bounded_control_profile"] = BoundedFirstTouchControl(forward_cap_m_s=1.48).contract()
    with pytest.raises(ValueError):
        require_navigation_pulse_capture(report, c, t)


@pytest.mark.parametrize("tamper", ["type", "model", "learned", "recording", "truthy", "profile"])
def test_only_fixed_original_zero_reference_request_is_admitted(tamper):
    pulse, model, profile, recording = (
        PlannerNavigationPulse(-0.08),
        zero_model(),
        BoundedFirstTouchControl(),
        True,
    )
    if tamper == "type":
        pulse = {}
    elif tamper == "model":
        model = None
    elif tamper == "learned":
        p = initial_parameters(135, 12, hidden_dimension=64, seed=2026101902)
        p["head_bias"][0] = 0.01
        model = make_model(
            p,
            mean=np.zeros(135),
            scale=np.ones(135),
            world_hash="sha256:" + "a" * 64,
            parent_model_hash="sha256:" + "b" * 64,
        )
    elif tamper == "recording":
        recording = False
    elif tamper == "truthy":
        recording = 1
    else:
        profile = BoundedFirstTouchControl(closing_gain_s_inv=3.0)
    with pytest.raises(ValueError):
        require_navigation_pulse_request(
            pulse, model, profile=profile, record_planner_events=recording
        )


def test_native_refuses_incomplete_pulse_before_any_output(tmp_path):
    with pytest.raises(ValueError, match="zero model"):
        native.run_native_first_touch_episode(
            argparse.Namespace(output_root=tmp_path / "unused"),
            course=(2.65, 0.1, 0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            planner_navigation_pulse=PlannerNavigationPulse(-0.08),
        )
    assert not (tmp_path / "unused").exists()


def test_default_on_pulse_cannot_be_hidden_by_off_path_ast_projection():
    f = next(
        n
        for n in ast.parse(Path(native.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_native_first_touch_episode"
    )
    index = next(i for i, a in enumerate(f.args.kwonlyargs) if a.arg == "planner_navigation_pulse")
    f.args.kw_defaults[index] = ast.Dict(keys=[], values=[])
    with pytest.raises(AssertionError):
        OffPath().visit(f)


@pytest.mark.parametrize("tamper", [None, "implicit", "hash", "short", "truthy", "leg", "claim"])
def test_old_qualification_cannot_silently_admit_new_command_channel(tamper):
    c, trace = captured()
    pulse_review = require_navigation_pulse_capture(c, c, trace)
    review, foundation, outcome = qualification_fixture()
    model_hash = c["reference_student_model"]["model_hash"]
    review.update(
        planner_navigation_pulse_review=pulse_review,
        actual_mujoco_dynamics_replayed=True,
        actual_pd_torque_reconstructed=True,
        neural_target_reconstructed=True,
        reference_student_review=dict(
            schema="soccer.rsi.pre_parent_reference_student_reconstruction.v1",
            model_hash=model_hash,
            frames_reconstructed=300,
            own_hidden_and_applied_reference_reconstructed=True,
            raw_foundation_navigation_targets_bit_exact=True,
            reconstruction_uses_same_pinned_decoder=True,
            parent_target_composition_reconstructed_separately=True,
            independent_neural_implementation=False,
            training_provenance_authenticated=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    kw = dict(
        reference_student_model_hash=model_hash,
        planner_navigation_pulse_contract_hash=c[TAG]["contract_hash"],
    )
    if tamper == "implicit":
        kw.pop("planner_navigation_pulse_contract_hash")
    elif tamper == "hash":
        kw["planner_navigation_pulse_contract_hash"] = "sha256:" + "c" * 64
    elif tamper == "short":
        pulse_review["frames_reconstructed"] = 299
    elif tamper == "truthy":
        pulse_review["pulse_composition_reconstructed"] = 1
    elif tamper == "leg":
        pulse_review["entire_leg_reference_strictly_zero"] = False
    elif tamper == "claim":
        pulse_review["planner_and_encoder_effect_verified_here"] = True
    if tamper is None:
        result = require_first_touch_qualification(review, foundation, outcome, **kw)
        assert result["m0_task_pass"]
        assert result["planner_navigation_pulse_contract_hash"] == c[TAG]["contract_hash"]
        assert not result["promotion_authorized"]
    else:
        with pytest.raises(ValueError):
            require_first_touch_qualification(review, foundation, outcome, **kw)


def test_prefix_compare_keeps_exact_original33_and_rejects_unknown_array_hiding():
    # Synthetic fields only. Counts/identity checks never claim physics here.
    baseline = {f"physical_{i}": np.zeros((300, 1, 1)) for i in range(33)}
    candidate = copy.deepcopy(baseline)
    candidate.update({PREFIX + k: np.zeros((300, 1, 3)) for k in FIELDS})
    candidate["physical_0"][100, 0, 0] = 1.0
    assert require_pulse_physical_parity(candidate, baseline, frames=100)[
        "original_array_bits_equal"
    ]
    with pytest.raises(ValueError):
        require_pulse_physical_parity(candidate, baseline, frames=300)
    candidate[PREFIX + "unexpected"] = np.zeros((300, 1, 1))
    with pytest.raises(ValueError):
        require_pulse_physical_parity(candidate, baseline, frames=100)
