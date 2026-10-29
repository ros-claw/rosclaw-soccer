"""Source-qualified cold CFtrack SIM context; no integration or model training.

Call ONLY after externally pinning all files and in an isolated process. The
initial state is a new declared episode, never a reset of a running world.
"""

import sys


def build_cold_cftrack(scene_text, external, *, direct_approach):
    if any(name in sys.modules for name in ("common", "policy", "FSM")):
        raise ValueError("isolated source-qualified CFtrack import namespace required")
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(shared_native_scene(scene_text, prefixes=("a/",)))
    commitment = compiled_model_hash(model)
    if commitment != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d":
        raise ValueError("exact standard native body required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ
    from policy.omnicontact.OmniContact import OmniContact

    policy = OmniContact(
        StateAndCmd(29),
        PolicyOutput(29),
        onnx_path=str(external / "policy/omnicontact/model/kick_50k.onnx"),
    )
    if tuple(policy.kinematics.joint_names) != tuple(G1_DDS_JOINT_NAMES):
        raise ValueError("qualified canonical external FK joint order required")
    policy.task, policy.reference_source, policy.active_object_name = "kickball", "CFgen", "ball"
    policy.box_dims = policy.ball_dims = np.full(3, 0.11, np.float32)
    policy.bbox_scale = policy.box_dims * 2
    policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)
    data = mujoco.MjData(model)
    data.qpos[lo.rq : lo.rq + 7] = [0, 0, 0.77, 1, 0, 0, 0]
    data.qpos[lo.qi] = DEFAULT_JOINT_POS_MJ
    data.qpos[lo.bq : lo.bq + 7] = [1, 0, 0.11, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    order = policy.lab2mj
    pd = NativePDParameters(
        policy.kps_lab[order], policy.kds_lab[order], 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    )
    world = SharedNativeMotorWorld(model, data, players={"a/": pd}, ball_body_name="soccer_ball")
    player = SharedOmniContactPlayer(
        world, "a/", policy, expected_policy_type=OmniContact, goal=np.array([5.0, 0.0, 0.11])
    )
    receipt = None
    if direct_approach:
        from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach

        receipt = condition_direct_kick_approach(policy)
        world._require_current_boundary()
    return player, commitment, receipt


def build_reference_cftrack(scene_text, external, reference_path, *, ground_zero_spin):
    """Construct one cold reference-initialized private SIM world after source admission."""
    if any(name in sys.modules for name in ("common", "policy", "FSM")):
        raise ValueError("isolated source-qualified CFtrack import namespace required")
    if type(ground_zero_spin) is not bool:
        raise ValueError("explicit initial ball factor required")
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.cftrack_motion_reference import load_cftrack_motion
    from rosclaw_soccer.rsi.initial_ball_factorial import initial_ball_factorial
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.omnicontact_initial_velocity import reference_initial_velocities
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.training.omnicontact_reference import LAB_TO_MUJOCO

    motion = load_cftrack_motion(reference_path)
    initial = reference_initial_velocities(motion, np.array(LAB_TO_MUJOCO))
    model = mujoco.MjModel.from_xml_string(shared_native_scene(scene_text, prefixes=("a/",)))
    commitment = compiled_model_hash(model)
    if commitment != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d":
        raise ValueError("exact standard native body required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.OmniContact import OmniContact

    policy = OmniContact(
        StateAndCmd(29),
        PolicyOutput(29),
        onnx_path=str(external / "policy/omnicontact/model/kick_50k.onnx"),
    )
    if tuple(policy.kinematics.joint_names) != tuple(G1_DDS_JOINT_NAMES) or not np.array_equal(
        policy.lab2mj, LAB_TO_MUJOCO
    ):
        raise ValueError("reference and actual HG joint order mismatch")
    policy.task, policy.reference_source, policy.active_object_name = (
        "kickball",
        "NPZmotion",
        "ball",
    )
    policy.npz_dir = str(reference_path)
    policy.tracking_start_frame, policy.tracking_end_frame = 0, -1
    policy.box_dims = policy.ball_dims = np.full(3, 0.11, np.float32)
    policy.bbox_scale = policy.box_dims * 2
    policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)
    data = mujoco.MjData(model)
    data.qpos[lo.rq : lo.rq + 3] = motion["body_pos_w"][0, 0]
    data.qpos[lo.rq + 3 : lo.rq + 7] = motion["body_quat_w"][0, 0]
    data.qpos[lo.qi] = motion["joint_pos"][0, policy.lab2mj]
    data.qpos[lo.bq : lo.bq + 3] = motion["object_pos_w"][0]
    data.qpos[lo.bq + 2] = max(0.11, float(data.qpos[lo.bq + 2]))
    data.qpos[lo.bq + 3 : lo.bq + 7] = motion["object_quat_w"][0]
    spin = initial["ball_angular_local"]
    if ground_zero_spin:
        position, spin = initial_ball_factorial(
            data.qpos[lo.bq : lo.bq + 3], spin, ground_ball=True, zero_spin=True
        )
        data.qpos[lo.bq : lo.bq + 3] = position
    data.qvel[lo.rv : lo.rv + 3] = initial["root_linear"]
    data.qvel[lo.rv + 3 : lo.rv + 6] = initial["root_angular_local"]
    data.qvel[lo.vi] = initial["joints"]
    data.qvel[lo.bv : lo.bv + 3] = initial["ball_linear"]
    data.qvel[lo.bv + 3 : lo.bv + 6] = spin
    mujoco.mj_forward(model, data)
    order = policy.lab2mj
    pd = NativePDParameters(
        policy.kps_lab[order], policy.kds_lab[order], 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    )
    world = SharedNativeMotorWorld(model, data, players={"a/": pd}, ball_body_name="soccer_ball")
    player = SharedOmniContactPlayer(
        world,
        "a/",
        policy,
        expected_policy_type=OmniContact,
        goal=np.array([5.0, 0.0, 0.11]),
        reference_source="NPZmotion",
    )
    return player, commitment
