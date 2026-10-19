"""Synthetic current-state forks; not qualified G1 feedback prediction."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.feedback_branch_control import fork_ik, fork_motor
from rosclaw_soccer.rsi.fixed_recurrent_reference_audit import _numeric_graph_hash
from rosclaw_soccer.rsi.foot_heading_evidence import FootHeadingCpuReconstruction
from rosclaw_soccer.rsi.recurrent_sampling_motor import (
    CompiledRecurrentSamplingMotor,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.rsi.recurrent_success_motor import make_model
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_foot_heading_evidence import fixture as ik_fixture
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_sampling_motor import long_body
from tests.rsi.test_recurrent_success_motor import boundary
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_original_ik_pending_snapshot_owns_side_contact_release():
    world, data, _, _, probe = ik_fixture()
    original = FootHeadingCpuReconstruction(world, probe)
    for frame in range(31):
        original.before_step(data, frame, data.xpos, np.zeros(29))
        if frame < 30:
            original.after_step(frame, np.zeros(6))
    branch = fork_ik(original, frame=30)
    assert branch.model is original.model
    assert branch.pending[1] is not original.pending[1]
    forces = np.array([10.0, 0, 0, 0, 0, 0])
    branch.after_step(30, forces)
    expected = branch.before_step(data, 31, data.xpos, np.zeros(29))
    assert original.contact is None and original.next_frame == 30
    original.after_step(30, forces)
    np.testing.assert_array_equal(expected, original.before_step(data, 31, data.xpos, np.zeros(29)))
    branch.pending[1][:] = 9
    assert not np.any(original.pending[1] == 9)


def test_recurrent_current_state_deeply_owned_and_same_draw(imitation_parent):  # noqa: F811
    policy = make_preview(make_sampling_view(make_model(imitation_parent), seed=901))
    original = CompiledRecurrentSamplingMotor(policy)
    observed, previous = long_body(), np.zeros(12)
    for frame in range(31):
        previous = original.delta_at_frame(policy, observed, **boundary(frame, previous))
    before = _numeric_graph_hash(original)
    branch = fork_motor(original, next_frame=31)
    assert branch._recurrent is not original._recurrent
    assert branch._memory is not original._memory
    assert branch._parent is not original._parent
    predicted = branch.delta_at_frame(policy, observed, **boundary(31, previous))
    assert _numeric_graph_hash(original) == before
    actual = original.delta_at_frame(policy, observed, **boundary(31, previous))
    np.testing.assert_array_equal(predicted, actual)
    np.testing.assert_array_equal(branch.hidden_state, original.hidden_state)
    assert _numeric_graph_hash(branch) == _numeric_graph_hash(original)


@pytest.mark.parametrize("next_frame", [30, 32, True, 31.0])
def test_wrong_recurrent_boundary_rejected(imitation_parent, next_frame):  # noqa: F811
    policy = make_preview(make_sampling_view(make_model(imitation_parent), seed=901))
    decoder = CompiledRecurrentSamplingMotor(policy)
    for frame in range(31):
        decoder.delta_at_frame(policy, long_body(), **boundary(frame, np.zeros(12)))
    with pytest.raises(ValueError):
        fork_motor(decoder, next_frame=next_frame)


@pytest.mark.parametrize("early", [False, True])
def test_composed_feedback_branch_real_adapters_on_toy_assets(imitation_parent, early):  # noqa: F811
    """Exercises the full adapter composition, NOT an official G1/SONIC result."""
    import mujoco

    from rosclaw_soccer.physics.g1_early_feedback_forecast import G1EarlyFeedbackForecaster
    from rosclaw_soccer.physics.g1_feedback_forecast import G1FeedbackForecaster
    from rosclaw_soccer.rsi.feedback_branch_control import G1FeedbackBranchControl
    from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
    from rosclaw_soccer.sim.contracts import hash_bytes
    from rosclaw_soccer.sim.root_velocity_reference import root_velocity_world
    from scripts.rsi_calibrate_feedback_branches import ik_signature, tracker_signature
    from tests.physics.test_g1_feedback_forecast import world
    from tests.providers.test_sonic_branch_state import tracker as toy_tracker

    sim_model, data, _ = world()
    # Declare toy limits before creating the new model-hash-bound forecaster.
    sim_model.jnt_range[:] = [-2.0, 2.0]
    # Toy spheres have near-zero hinge inertia; give this synthetic integration
    # fixture explicit motor inertia before binding its new compiled-model hash.
    sim_model.dof_armature[6:35] = 0.02
    binary = np.zeros(mujoco.mj_sizeModel(sim_model), dtype=np.uint8)
    mujoco.mj_saveModel(sim_model, None, binary)
    forecaster_class = G1EarlyFeedbackForecaster if early else G1FeedbackForecaster
    f = forecaster_class(
        sim_model,
        world_hash=hash_bytes(binary.tobytes()),
        kp=np.full(29, 10.0),
        kd=np.full(29, 0.2),
        max_forecast_steps=80 if early else 40,
    )
    tracker, _, _ = toy_tracker()
    tracker.reference = tracker.reference[:, :1].repeat(1, 256, 1)
    q, v = data.qpos[None].copy(), data.qvel[None].copy()
    tracker.reset(q, v)
    policy = make_preview(make_sampling_view(make_model(imitation_parent), seed=901))
    decoder = CompiledRecurrentSamplingMotor(policy)
    ik = FootHeadingCpuReconstruction(sim_model, FootHeadingContactProbe(False))
    pelvis, ball = sim_model.body("pelvis").id, sim_model.body("ball").id
    geometry = [
        sim_model.body(n).id
        for n in (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_knee_link",
            "right_knee_link",
        )
    ]
    observed_body = dict(
        joint_position_rad=data.qpos[7:36],
        joint_velocity_rad_s=data.qvel[6:35],
        root_pose_xyzw_m=np.concatenate((data.xpos[pelvis], data.xquat[pelvis][[1, 2, 3, 0]])),
        root_velocity_world=root_velocity_world(sim_model, data, int(pelvis), "body-com"),
        ball_position_before_step_m=data.xpos[ball],
        ball_linear_velocity_before_step_m_s=np.zeros(3),
        foot_geometry_position_before_step_m=data.xpos[geometry],
    )
    body = {k: np.repeat(a[None, None], 31, axis=0) for k, a in observed_body.items()}
    previous = np.zeros(12)
    for frame in range(31):
        if frame:
            tracker.observe(q, v)
        nav = tracker.update(frame, q, v)[0].numpy().astype(np.float64)
        delta = ik.before_step(data, frame, data.xpos, nav)
        nominal = nav + delta
        previous = decoder.delta_at_frame(
            policy,
            body,
            frame=frame,
            nominal_target=nominal,
            baseline=nominal[:12],
            limits=np.tile([-2.0, 2.0], (12, 1)),
            previous=previous,
            previous_contact_forces=np.zeros(6),
        )
        if frame < 30:
            ik.after_step(frame, np.zeros(6))
    target = nominal.copy()
    target[:12] += previous
    before = (_numeric_graph_hash(decoder), tracker_signature(tracker), ik_signature(ik))
    outputs = []
    for _ in range(2):
        controller = G1FeedbackBranchControl(
            sim_model,
            frame=30,
            tracker=tracker,
            decoder=decoder,
            ik=ik,
            policy=policy,
            body_prefix=body,
            previous_motor_delta_rad=previous,
        )
        outputs.append(
            f.forecast(data, target, control=controller, early_offset_nm=np.zeros(29))
            if early
            else f.forecast(data, target, control=controller, first_offset_nm=np.zeros(29))
        )
    assert before == (_numeric_graph_hash(decoder), tracker_signature(tracker), ik_signature(ik))
    assert outputs[0].end_integration_state.tobytes() == outputs[1].end_integration_state.tobytes()
    if early:
        assert len(outputs[0].next_target_records) == len(outputs[0].next_neural_records) == 3
        for row in outputs[0].next_target_records:
            assert row["joint_target_rad"].shape == (29,)
        assert f.next_target_completed == 6 and f.integrated_steps == 80
    else:
        assert outputs[0].next_target_record["joint_target_rad"].shape == (29,)
        assert f.next_target_completed == 2 and f.integrated_steps == 40
    assert data.time == 0.6
