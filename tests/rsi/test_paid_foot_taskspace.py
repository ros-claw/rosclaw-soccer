from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent, phase_intent
from scripts.rsi_search_paid_foot_contact import contact_rank


def test_bounded_smooth_phase_intent_and_no_reference_extension():
    latent = np.ones((2, 6))
    assert not phase_intent(0, 120, latent).any()
    assert not phase_intent(300, 120, latent).any()
    for frame in range(300):
        assert np.isfinite(phase_intent(frame, 120, latent)).all()
    with pytest.raises(ValueError):
        phase_intent(100, 120, latent * 1.01)
    with pytest.raises(ValueError):
        phase_intent(100, 120, np.zeros((12,)))


def test_teacher_rank_never_trades_body_failure_for_a_clean_highlight():
    import copy

    row = dict(
        body_failed=False,
        forbidden=True,
        terminal_reward=3,
        postcontact=dict(
            correct_foot_first=True,
            post_contact_quality_observed=True,
            goal_progress_1p2s_m=4,
            goal_lateral_1p2s_m=0,
            other_foot_contact_in_window=False,
            maximum_pelvis_retreat_after_contact_m=0,
        ),
    )
    clean = copy.deepcopy(row)
    clean["forbidden"] = False
    failed = copy.deepcopy(row)
    failed["body_failed"] = True
    assert contact_rank([row, row]) > contact_rank([clean, failed])
    assert contact_rank([clean, clean]) > contact_rank([row, row])
    grazing = copy.deepcopy(clean)
    grazing["postcontact"]["goal_progress_1p2s_m"] = 0.01
    assert contact_rank([grazing, grazing])[1] == 0


def synthetic_episode(with_ball=False):
    names = list(G1_DDS_JOINT_NAMES)
    leg = [
        "left_" + s + "_joint"
        for s in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    inertial = '<inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>'
    chain = ""
    for name, axis in zip(leg, ("0 1 0", "1 0 0", "0 0 1", "0 1 0", "0 1 0", "1 0 0"), strict=True):
        body = name.removesuffix("_joint") + "_link"
        chain += (
            f'<body name="{body}" pos="0 0 -.05">{inertial}<joint name="{name}" axis="{axis}"/>'
        )
        if with_ball and body == "left_knee_link":
            chain += '<geom name="left_shin" type="capsule" size=".04" fromto=".02 0 0 .02 0 -.1"/>'
    chain += "</body>" * 6
    others = "".join(
        f'<body name="other{i}">{inertial}<joint name="{name}"/></body>'
        for i, name in enumerate(names)
        if name not in leg
    )
    ball = (
        '<body name="ball" pos="-.1 0 .55">'
        '<geom name="soccer_ball_geom" type="sphere" size=".11"/></body>'
        if with_ball
        else ""
    )
    xml = (
        '<mujoco><worldbody><body name="pelvis" pos="0 0 .8"><freejoint/>'
        f"{inertial}{chain}{others}</body>{ball}</worldbody></mujoco>"
    )
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    qi = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
    return SimpleNamespace(
        frame=108,
        env=SimpleNamespace(model=m, data=d, pelvis_body_id=m.body("pelvis").id),
        layout=SimpleNamespace(qi=qi, ball=m.body("ball").id if with_ball else -1),
        offset=np.zeros(29, np.float32),
        scale=np.ones(29, np.float32),
        order=np.arange(29),
    )


@pytest.mark.parametrize("observed", [False, True])
def test_shin_clearance_uses_only_scratch_kinematics_and_original_joint_bound(observed):
    ep = synthetic_episode(with_ball=True)
    before = (ep.env.data.qpos.copy(), ep.env.data.qvel.copy(), ep.env.data.ctrl.copy())
    intent = FootTaskSpaceIntent(ep.env.model, foot="left", center=104, latent=np.zeros((2, 6)))
    ep.layout.canonical = lambda data: (np.zeros(43), np.zeros(41))
    action, plan = intent.solve_offset(
        ep,
        np.zeros(29, np.float32),
        np.zeros(6),
        shin_clearance=True,
        observed_shin_clearance=observed,
    )
    assert np.isfinite(action).all() and abs(action).max() <= 0.200001
    assert plan["requested_shin_surface_margin_m"] == 0.015
    assert np.isfinite(plan["shin_clearance_after_scratch_m"])
    assert plan["ik_iterations"] == 4
    if observed:
        assert plan["clearance_from_observed_pose_plus_correction"] is True
        assert plan["closing_velocity_horizon_s"] == 0.04
    for actual, old in zip(
        (ep.env.data.qpos, ep.env.data.qvel, ep.env.data.ctrl), before, strict=True
    ):
        np.testing.assert_array_equal(actual, old)
    assert ep.env.data.time == 0


def test_zero_intent_is_bit_exact_and_ik_never_mutates_live_state_or_model():
    ep = synthetic_episode()
    m, d = ep.env.model, ep.env.data
    action = np.zeros(29, np.float32)
    zero = FootTaskSpaceIntent(m, foot="left", center=120, latent=np.zeros((2, 6)))
    returned, trace = zero(ep, action)
    np.testing.assert_array_equal(returned, action)
    assert trace["ik_iterations"] == 0
    before = {
        k: getattr(d, k).copy()
        for k in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "xpos", "xmat")
    }
    dimensions = m.geom_size.copy()
    latent = np.zeros((2, 6))
    latent[0, 0] = 0.25
    latent[0, 4] = 0.1
    intent = FootTaskSpaceIntent(m, foot="left", center=120, latent=latent)
    transformed, trace = intent(ep, action)
    assert transformed.dtype == np.float32 and transformed.shape == (29,)
    assert np.isfinite(transformed).all() and np.abs(trace["joint_delta_HG"]).max() <= 0.2 + 1e-12
    assert trace["ik_iterations"] == 4
    assert trace["position_residual_m"] < 0.01
    for k, value in before.items():
        np.testing.assert_array_equal(getattr(d, k), value)
    np.testing.assert_array_equal(m.geom_size, dimensions)
