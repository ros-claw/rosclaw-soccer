import numpy as np
import pytest

from rosclaw_soccer.rsi.native_foot_contact_teacher import (
    FootContactTeacherConfig,
    NativeFootContactTeacher,
    bounded_contact_target,
)


def context(foot=0):
    j = np.zeros((3, 29))
    j[:, foot * 6 : foot * 6 + 3] = np.eye(3)
    return dict(
        nominal=np.zeros(29, np.float64),
        previous_delta=np.zeros(29),
        joint_ranges=np.tile([-2.0, 2.0], (29, 1)),
        jacobian_hg=j,
        relative_velocity=np.array([0.8, 0.2, 0.0]),
        direction_to_ball=np.array([1.0, 0.0, 0.0]),
        surface_gap=0.1,
        foot=foot,
        enabled=True,
        config=FootContactTeacherConfig(),
    )


@pytest.mark.parametrize("foot", [0, 1])
def test_foot_dls_target_respects_amplitude_slew_and_selected_leg(foot):
    args = context(foot)
    before = {k: v.copy() for k, v in args.items() if isinstance(v, np.ndarray)}
    for _ in range(20):
        result = bounded_contact_target(**args)
        assert result["active"] and not result["fallback"]
        delta = result["applied_delta"]
        assert max(abs(delta)) <= 0.08
        assert max(abs(delta - args["previous_delta"])) <= 0.015 + 1e-12
        assert np.any(delta[foot * 6 : foot * 6 + 6])
        assert not np.any(np.delete(delta, np.arange(foot * 6, foot * 6 + 6)))
        args["previous_delta"] = delta.copy()
    args["enabled"] = False
    for _ in range(20):
        result = bounded_contact_target(**args)
        assert max(abs(result["applied_delta"] - args["previous_delta"])) <= 0.015 + 1e-12
        args["previous_delta"] = result["applied_delta"].copy()
    assert not np.any(args["previous_delta"])
    for key in before.keys() - {"previous_delta"}:
        np.testing.assert_array_equal(before[key], args[key])


def test_zero_strength_is_exact_parent_even_outside_nominal_joint_range():
    args = context()
    args["config"] = FootContactTeacherConfig(strength=0)
    args["nominal"][0] = 3.0
    args["nominal"][10] = -0.0
    result = bounded_contact_target(**args)
    np.testing.assert_array_equal(result["target"].view(np.uint64), args["nominal"].view(np.uint64))
    assert not result["active"] and not result["fallback"]


def test_infeasible_target_interval_is_explicit_parent_fallback():
    args = context()
    args["nominal"][0] = 3.0
    result = bounded_contact_target(**args)
    assert result["fallback"]
    np.testing.assert_array_equal(result["target"], args["nominal"])


def test_dls_world_frame_half_turn_does_not_change_hg_joint_result():
    args = context()
    expected = bounded_contact_target(**args)["target"]
    turn = np.diag([-1.0, -1.0, 1.0])
    for key in ("relative_velocity", "direction_to_ball", "jacobian_hg"):
        args[key] = turn @ args[key]
    np.testing.assert_allclose(bounded_contact_target(**args)["target"], expected, atol=1e-14)


@pytest.mark.parametrize("fault", ["nonfinite", "cross_leg", "nonleg", "invalid_foot"])
def test_bad_geometry_or_unowned_history_rejected(fault):
    args = context()
    if fault == "nonfinite":
        args["jacobian_hg"][0, 0] = np.nan
    elif fault == "cross_leg":
        args["previous_delta"][6] = 0.01
    elif fault == "nonleg":
        args["previous_delta"][12] = 0.01
    else:
        args["foot"] = 2
    with pytest.raises(ValueError):
        bounded_contact_target(**args)


@pytest.mark.parametrize(
    "change", [{"strength": 1.1}, {"maximum_delta_rad": 1.0}, {"damping_m_per_rad": 0}]
)
def test_teacher_parameters_fail_closed(change):
    with pytest.raises(ValueError):
        FootContactTeacherConfig(**change)


def test_native_geometry_adapter_does_not_step_or_change_simulator_state():
    import mujoco

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

    prefix = "red_finisher/"
    chain = ""
    for i, name in enumerate(G1_DDS_JOINT_NAMES):
        chain += (
            f'<body name="{prefix}link{i}" pos="0.01 0 0">'
            f'<joint name="{prefix}{name}" axis="0 1 0" range="-2 2"/>'
            '<geom type="sphere" size=".01" mass=".01" contype="0" conaffinity="0"/>'
        )
        if i in (0, 6):
            side, y = ("left", 0.05) if i == 0 else ("right", -0.05)
            chain += (
                f'<body name="{prefix}{side}_ankle_roll_link" pos=".02 {y} 0">'
                '<geom type="box" size=".04 .03 .02" mass=".1"/></body>'
            )
    chain += "</body>" * 29
    motors = "".join(f'<motor joint="{prefix}{name}"/>' for name in G1_DDS_JOINT_NAMES)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>'
        f'<body name="{prefix}pelvis" pos="0 0 .5"><freejoint/>'
        '<geom type="sphere" size=".05" mass="1" contype="0" conaffinity="0"/>'
        + chain
        + '</body><body name="soccer_ball" pos=".2 .05 .5"><freejoint/>'
        '<geom name="ball_geom" type="sphere" size=".11" mass=".43"/></body></worldbody>'
        f"<actuator>{motors}</actuator></mujoco>"
    )
    layout = NativeVelocityLayout(model, robot_prefix=prefix, ball_body_name="soccer_ball")
    data = mujoco.MjData(model)
    data.qvel[layout.bv] = -0.3
    mujoco.mj_forward(model, data)
    before = [a.copy() for a in (data.qpos, data.qvel, data.ctrl, data.xfrc_applied)]
    teacher = NativeFootContactTeacher(FootContactTeacherConfig())
    result = teacher.propose(
        model=model,
        snapshot=data,
        layout=layout,
        nominal=np.zeros(29),
        ball_geom=model.geom("ball_geom").id,
        foot_bodies=[model.body(prefix + s + "_ankle_roll_link").id for s in ("left", "right")],
        enabled=True,
        frame=0,
    )
    assert result["active"] and result["foot_index"] in (0, 1)
    assert result["surface_gap"] < 0.35
    assert result["jacobian_hg"].shape == (3, 29)
    assert data.time == 0
    for original, current in zip(
        before, (data.qpos, data.qvel, data.ctrl, data.xfrc_applied), strict=True
    ):
        np.testing.assert_array_equal(original, current)
