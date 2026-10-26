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
