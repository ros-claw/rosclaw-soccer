"""Scalar capture and causal IK checks; not a qualified physical experiment."""

import argparse
import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi import native_first_touch_episode as native
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.original_ik_planner_evidence import (
    FIELDS,
    PREFIX,
    TAG,
    OriginalIkCpuReconstruction,
    require_original_ik_capture,
    require_original_ik_parity,
)
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
from rosclaw_soccer.rsi.taskspace_swing_probe import (
    continuous_swing_joint_delta,
    release_joint_delta,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_first_touch_qualification import fixture as qualification_fixture
from tests.rsi.test_swing_release_default_arithmetic import OffPath


def captured(delta=0.08, contact=None):
    pulse, profile = OriginalIkPlannerPulse(delta), BoundedFirstTouchControl()
    c = {
        TAG: pulse.contract(),
        "bounded_control_profile": profile.contract(),
        "execution_profile": "taskspace_plus_motor",
        "swing_mode": "bounded_contact_diagnostic",
        "taskspace_contract": dict(
            forward_cap_m=0.08,
            lateral_cap_m=0.15,
            acquisition_max_gap_m=0.95,
            revalidate_swing_side=True,
            vertical_offset_m=0.04,
        ),
    }
    t = {PREFIX + k: np.zeros((300, 1, 3)) for k in FIELDS}
    for key, width in (
        ("navigation_command", 3),
        ("approach_requested_navigation_command", 3),
        ("root_pose_xyzw_m", 7),
        ("ball_position_before_step_m", 3),
        ("ball_linear_velocity_before_step_m_s", 3),
        ("force_n", 6),
    ):
        t[key] = np.zeros((300, 1, width))
    # Exercise fading, both saturation bounds and unsaturated composition.
    for f in range(300):
        gap_x = (0.2, 0.6, 1.2)[f % 3]
        gap_y = (-0.3, 0.1, 0.4)[f % 3]
        t["ball_position_before_step_m"][f, 0, :2] = [gap_x, gap_y]
        t["ball_linear_velocity_before_step_m_s"][f, 0, 0] = 0.5
        raw, base = profile.navigation(
            gap_x=gap_x,
            gap_y=gap_y,
            ball_vx=0.5,
            before_contact=contact is None or f <= contact,
        )
        residual, actual = pulse.compose(f, base)
        t["navigation_command"][f, 0] = actual
        t["approach_requested_navigation_command"][f, 0] = raw
        for key, value in zip(FIELDS, (base, residual, actual), strict=True):
            t[PREFIX + key][f, 0] = value
    if contact is not None:
        t["force_n"][contact, 0, 4] = 2.0
    return c, t


@pytest.mark.parametrize("delta", [0.0, 0.08])
@pytest.mark.parametrize("contact", [None, 0, 110, 159, 299])
def test_scalar_reconstruction_is_independent_and_not_ik_proof(delta, contact, monkeypatch):
    c, t = captured(delta, contact)
    before = copy.deepcopy(t)
    monkeypatch.setattr(
        OriginalIkPlannerPulse, "compose", lambda *a: pytest.fail("producer reused")
    )
    monkeypatch.setattr(
        BoundedFirstTouchControl, "navigation", lambda *a, **k: pytest.fail("B2 reused")
    )
    r = require_original_ik_capture(c, c, t)
    assert r["frames_reconstructed"] == 300
    assert r["original_ik_intent_checked"] is True
    assert r["original_ik_targets_reconstructed_here"] is False
    assert r["report_hash"] == hash_json({k: v for k, v in r.items() if k != "report_hash"})
    for key in t:
        assert t[key].tobytes() == before[key].tobytes()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "old",
        "reference",
        "model",
        "swing",
        "taskspace",
        "profile",
        "report",
        "nav",
        "raw",
        "residual",
        "nan",
        "short",
        "dtype",
        "negative_force",
    ],
)
def test_unmixed_whole_capture_required(mutation):
    c, t = captured()
    r = copy.deepcopy(c)
    if mutation == "missing":
        t.pop(PREFIX + FIELDS[0])
    elif mutation == "extra":
        t[PREFIX + "unknown"] = np.zeros((300, 1, 1))
    elif mutation in ("old", "reference"):
        t[("planner_pulse_" if mutation == "old" else "reference_student_") + "x"] = np.zeros(300)
    elif mutation == "model":
        c["reference_student_model"] = None
    elif mutation == "swing":
        c["swing_mode"] = "legacy"
    elif mutation == "taskspace":
        c["taskspace_contract"]["lateral_cap_m"] = 0.2
    elif mutation == "profile":
        c["bounded_control_profile"] = BoundedFirstTouchControl(forward_cap_m_s=1.2).contract()
    elif mutation == "report":
        r[TAG] = {}
    elif mutation in ("nav", "raw", "residual"):
        key = {
            "nav": "navigation_command",
            "raw": "approach_requested_navigation_command",
            "residual": PREFIX + FIELDS[1],
        }[mutation]
        t[key][110, 0, 1] += 0.001
    elif mutation == "nan":
        t["force_n"][0, 0, 0] = np.nan
    elif mutation == "short":
        t["force_n"] = t["force_n"][:-1]
    elif mutation == "dtype":
        t["force_n"] = t["force_n"].astype(np.float32)
    else:
        t["force_n"][0, 0, 0] = -1
    with pytest.raises(ValueError):
        require_original_ik_capture(r, c, t)


@pytest.mark.parametrize("frames", [100, 300])
def test_parity_preserves_every_original_array_bit(frames):
    base = {f"field_{i}": np.zeros((300, 1, i + 1)) for i in range(33)}
    trace = copy.deepcopy(base)
    trace.update({PREFIX + k: np.zeros((300, 1, 3)) for k in FIELDS})
    assert require_original_ik_parity(trace, base, frames=frames)["physical_array_count"] == 33
    trace["field_0"][frames - 1, 0, 0] = -0.0
    with pytest.raises(ValueError, match="bits"):
        require_original_ik_parity(trace, base, frames=frames)


def tiny_model():
    import mujoco

    bodies = []
    for side, names in enumerate((G1_DDS_JOINT_NAMES[:6], G1_DDS_JOINT_NAMES[6:12])):
        nested = '<geom type="sphere" size="0.02" mass="0.1"/>'
        for i, name in reversed(list(enumerate(names))):
            body_name = (
                ("left" if side == 0 else "right") + "_ankle_roll_link" if i == 5 else "b_" + name
            )
            nested = (
                f'<body name="{body_name}" pos="0 0 .03">'
                f'<joint name="{name}" type="hinge" axis="0 1 0" range="-2 2"/>'
                f'{nested}<geom type="sphere" size=".01" mass=".01"/></body>'
            )
        bodies.append(nested)
    for name in G1_DDS_JOINT_NAMES[12:]:
        bodies.append(
            f'<body><joint name="{name}" type="hinge" range="-2 2"/>'
            '<geom type="sphere" size=".01" mass=".01"/></body>'
        )
    xml = (
        '<mujoco><compiler angle="radian"/><worldbody>'
        + "".join(bodies)
        + '<body name="ball"><geom type="sphere" size=".11"/></body></worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_actual_cpu_jacobian_and_causal_contact_release_are_reconstructed():
    import mujoco

    model, data = tiny_model()
    recon = OriginalIkCpuReconstruction(model)
    positions = data.xpos.copy()
    positions[recon.feet[0]] = [0.0, 0.0, 0.15]
    positions[recon.feet[1]] = [0.0, 0.0, 0.1]
    positions[recon.ball] = [0.5, 0.03, 0.11]
    target, forces = np.zeros(29), np.zeros(6)
    with pytest.raises(ValueError):
        recon.review()
    contact_delta = None
    for f in range(300):
        full = recon.before_step(data, f, positions, target)
        if f < 30:
            assert not full.any()
        elif f <= 45:
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jac, None, recon.feet[0])
            expected = continuous_swing_joint_delta(
                positions[recon.feet[0]],
                positions[recon.ball],
                jac[:, recon.vi[:6]],
                target[:6],
                recon.limits[:6],
                forward_cap_m=0.08,
                lateral_cap_m=0.15,
                vertical_offset_m=0.04,
            )
            np.testing.assert_array_equal(full[:6], expected)
            assert full[:6].any()
            contact_delta = expected.copy()
        else:
            np.testing.assert_array_equal(full[:6], release_joint_delta(contact_delta, f - 45))
        assert not full[6:].any()
        forces[4] = 2.0 if f == 45 else 0.0
        recon.after_step(f, forces)
    result = recon.review()
    assert result["original_ik_from_actual_cpu_state_reconstructed"] is True
    assert result["independent_ik_implementation"] is False
    assert result["new_physical_steps"] == 0
    with pytest.raises(ValueError):
        recon.before_step(data, 300, positions, target)


@pytest.mark.parametrize(
    "mutation",
    ["skip", "repeat", "after_first", "double_before", "nan_positions", "bad_target", "bad_forces"],
)
def test_causal_clock_and_finite_cpu_inputs_fail_closed(mutation):
    model, data = tiny_model()
    r = OriginalIkCpuReconstruction(model)
    pos, target, forces = data.xpos.copy(), np.zeros(29), np.zeros(6)
    with pytest.raises(ValueError):
        if mutation == "after_first":
            r.after_step(0, forces)
        elif mutation == "skip":
            r.before_step(data, 1, pos, target)
        elif mutation == "nan_positions":
            pos[0, 0] = np.nan
            r.before_step(data, 0, pos, target)
        elif mutation == "bad_target":
            r.before_step(data, 0, pos, target.astype(np.float32))
        else:
            r.before_step(data, 0, pos, target)
            if mutation == "double_before":
                r.before_step(data, 0, pos, target)
            elif mutation == "bad_forces":
                forces[0] = -1
                r.after_step(0, forces)
            else:
                r.after_step(0, forces)
                r.before_step(data, 0, pos, target)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "implicit",
        "missing_cpu",
        "short",
        "future_claim",
        "source",
        "hash",
        "truthy",
        "unsealed",
        "reference",
        "old_pulse",
        "no_dynamics",
        "false_zero",
    ],
)
def test_qualification_requires_explicit_whole_scalar_and_actual_cpu_reconstruction(mutation):
    c, t = captured()
    scalar = require_original_ik_capture(c, c, t)
    model, data = tiny_model()
    recon = OriginalIkCpuReconstruction(model)
    for frame in range(300):
        recon.before_step(data, frame, data.xpos, np.zeros(29))
        recon.after_step(frame, np.zeros(6))
    cpu = recon.review()
    review, foundation, outcome = qualification_fixture()
    review.update(
        original_ik_planner_pulse_review=scalar,
        original_ik_cpu_reconstruction=cpu,
        actual_mujoco_dynamics_replayed=True,
        actual_pd_torque_reconstructed=True,
        neural_target_reconstructed=True,
    )
    kw = dict(original_ik_planner_pulse_contract_hash=c[TAG]["contract_hash"])
    if mutation == "implicit":
        kw.clear()
    elif mutation == "missing_cpu":
        review.pop("original_ik_cpu_reconstruction")
    elif mutation == "short":
        cpu["frames_reconstructed"] = 299
    elif mutation == "future_claim":
        scalar["original_ik_targets_reconstructed_here"] = True
    elif mutation == "source":
        cpu["source_hash"] = "sha256:" + "b" * 64
    elif mutation == "hash":
        kw["original_ik_planner_pulse_contract_hash"] = "sha256:" + "c" * 64
    elif mutation == "truthy":
        cpu["original_ik_from_actual_cpu_state_reconstructed"] = 1
    elif mutation == "unsealed":
        cpu["report_hash"] = "sha256:" + "d" * 64
    elif mutation == "reference":
        kw["reference_student_model_hash"] = "sha256:" + "e" * 64
    elif mutation == "old_pulse":
        kw["planner_navigation_pulse_contract_hash"] = "sha256:" + "e" * 64
    elif mutation == "no_dynamics":
        review["actual_mujoco_dynamics_replayed"] = False
    elif mutation == "false_zero":
        cpu["new_physical_steps"] = False
    # Mutation guards must hold even if a caller reseals the altered review.
    if mutation != "unsealed":
        for d in (scalar, cpu):
            d["report_hash"] = hash_json({k: v for k, v in d.items() if k != "report_hash"})
    if mutation is None:
        result = require_first_touch_qualification(review, foundation, outcome, **kw)
        assert result["original_ik_planner_pulse_contract_hash"] == c[TAG]["contract_hash"]
        assert result["promotion_authorized"] is False
    else:
        with pytest.raises(ValueError):
            require_first_touch_qualification(review, foundation, outcome, **kw)


@pytest.mark.parametrize("delta", [0.0, 0.08])
def test_native_refuses_unqualified_ik_request_before_output_or_physics(tmp_path, delta):
    with pytest.raises(ValueError, match="original-IK planner SIM request"):
        native.run_native_first_touch_episode(
            argparse.Namespace(output_root=tmp_path / "unused"),
            course=(2.65, 0.1, 0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            original_ik_planner_pulse=OriginalIkPlannerPulse(delta),
            control_profile=BoundedFirstTouchControl(),
            record_default_planner_events=True,
        )
    assert not (tmp_path / "unused").exists()


def test_default_on_ik_cannot_be_hidden_by_default_off_arithmetic_projection():
    f = next(
        n
        for n in ast.parse(Path(native.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_native_first_touch_episode"
    )
    i = next(i for i, a in enumerate(f.args.kwonlyargs) if a.arg == "original_ik_planner_pulse")
    f.args.kw_defaults[i] = ast.Dict(keys=[], values=[])
    with pytest.raises(AssertionError):
        OffPath().visit(f)
