import numpy as np

from rosclaw_soccer.rsi.step_motor_execution import delta_at_frame, make_preview
from tests.rsi.test_step_motor_network import model  # noqa: F401


def body():
    result = dict(
        joint_position_rad=np.zeros((40, 1, 29)),
        joint_velocity_rad_s=np.zeros((40, 1, 29)),
        root_pose_xyzw_m=np.zeros((40, 1, 7)),
        root_velocity_world=np.zeros((40, 1, 6)),
        ball_position_before_step_m=np.zeros((40, 1, 3)),
        ball_linear_velocity_before_step_m_s=np.zeros((40, 1, 3)),
        foot_geometry_position_before_step_m=np.zeros((40, 1, 4, 3)),
    )
    result["root_pose_xyzw_m"][:, :, 6] = 1
    return result


def test_per_frame_inference_never_reads_future_state(model):  # noqa: F811
    policy = make_preview(model)
    observation = body()
    boundary = dict(
        frame=30,
        nominal_target=np.zeros(29),
        baseline=np.zeros(12),
        limits=np.tile([-1.0, 1.0], (12, 1)),
        previous=np.zeros(12),
        previous_contact_forces=np.zeros(6),
    )
    first = delta_at_frame(policy, observation, **boundary)
    observation["ball_position_before_step_m"][31:] = 1000
    observation["joint_position_rad"][31:] = 1000
    assert np.array_equal(first, delta_at_frame(policy, observation, **boundary))
    assert policy["step_motor_proof"]["qualification"] == "UNQUALIFIED_SIM_COUNTERFACTUAL"
    assert policy["step_motor_proof"]["promotion_authorized"] is False


def test_per_frame_proposal_is_zero_before_decision_boundary(model):  # noqa: F811
    policy = make_preview(model)
    result = delta_at_frame(
        policy,
        body(),
        frame=20,
        nominal_target=np.zeros(29),
        baseline=np.zeros(12),
        limits=np.tile([-1.0, 1.0], (12, 1)),
        previous=np.zeros(12),
        previous_contact_forces=np.zeros(6),
    )
    assert np.array_equal(result, np.zeros(12))
