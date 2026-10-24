"""Read-only measured transition extraction; never integrate or apply a command.

Actor observes121 columns; critic additionally observes the gait's preceding
29 actions and unmasked2-phase clock. The command-bearing98-vector is NOT fed
to either learner. Duration truncations retain a reconstructed actual next
state and controller memory. Teacher queries are not executed actions.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np


def build_role_residual_episode(
    root: Path, row: dict[str, Any], pins: dict[str, str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import mujoco
    from rosclaw.control.parent_residual import parent_context

    from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy, phase_delta, target29
    from rosclaw_soccer.rsi.native_role_command import (
        NativeRoleCommandPolicy,
        role_command_observation,
    )
    from rosclaw_soccer.rsi.native_role_residual import ROLE_COMMAND_LIMITS
    from rosclaw_soccer.rsi.native_role_residual_task import native_role_residual_reward
    from rosclaw_soccer.rsi.native_role_teacher import coached_native_role_label
    from rosclaw_soccer.rsi.native_team_intent import ROLES
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.sim.contracts import hash_json

    if row["arm"] != "role-command-residual" or row["repeat"] != 0:
        raise ValueError("one real source trajectory, never a duplicated re-execution, required")
    scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(k) for k in pins if k.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(k) for k in pins if k.endswith("/velocity/v0/params/deploy.yaml")]
    heads = [Path(k) for k in pins if k.endswith("/residual-head.json")]
    if any(len(v) != 1 for v in (scenes, models, configs, heads)):
        raise ValueError("unique bound scene/gait/initial residual head required")
    import json

    metadata = json.loads(heads[0].read_text())
    if metadata["update_index"] != 0:
        raise ValueError("this exploration bank requires the qualified zero-head parent")
    encoders = [
        Path(k)
        for k, v in pins.items()
        if v == metadata["encoder_hash"] and k.endswith("/command-policy.json")
    ]
    if len(encoders) != 1:
        raise ValueError("exact unique retained feature encoder required")
    encoder = NativeRoleCommandPolicy(
        encoders[0],
        expected_hash=metadata["encoder_hash"],
        expected_binding=metadata["binding_hash"],
    )
    agents = [(t, r, t + "_" + r + "/") for t in ("red", "blue") for r in ROLES]
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=tuple(x[2] for x in agents))
    )
    data = mujoco.MjData(model)
    layouts = {
        prefix: NativeVelocityLayout(
            model, robot_prefix=prefix, ball_body_name="soccer_ball", player_count=8
        )
        for _, _, prefix in agents
    }
    if [x["body_binding_hash"] for x in row["players"]] != [
        hash_json(layouts[p].profile()) for _, _, p in agents
    ]:
        raise ValueError("actual model binding differs from measured source")
    with np.load(root / row["artifact"]["path"]) as saved:
        raw = {key: saved[key] for key in saved.files}
    count = len(raw["commands"])
    if (
        not 50 < count <= 600
        or len(raw["qpos"]) != count * 10
        or raw["residual_context121"].shape != (count, 8, 121)
        or raw["residual_features131"].shape != (count, 8, 131)
        or not raw["residual_model_valid"][50:].all()
    ):
        raise ValueError("complete aligned active controller snapshots required")
    action = raw["residual_normalized_actions"]
    if not np.array_equal(action[50:], np.tanh(raw["residual_latent_noise"][50:])):
        raise ValueError("recorded zero-head exploration action differs from declared latent noise")
    if not np.allclose(
        raw["residual_deltas"],
        action * ROLE_COMMAND_LIMITS * metadata["maximum_ratio"],
        rtol=0,
        atol=1e-14,
    ):
        raise ValueError("raw residual action was not composed according to its contract")
    if not np.allclose(
        raw["residual_raw_commands"],
        raw["residual_parent_commands"].astype(np.float64) + raw["residual_deltas"],
        rtol=0,
        atol=1e-14,
    ):
        raise ValueError("raw proposal/parent link differs")
    deviation = abs(raw["commands"].astype(np.float64) - raw["residual_parent_commands"])
    if (deviation > ROLE_COMMAND_LIMITS * metadata["maximum_ratio"] + 1e-6).any():
        raise ValueError("applied command escaped residual authority after guard")
    commands = raw["commands"]
    if (
        (commands[:, :, 0] < -0.5 - 1e-6).any()
        or (commands[:, :, 0] > 1 + 1e-6).any()
        or (abs(commands[:, :, 1]) > 0.5 + 1e-6).any()
        or (abs(commands[:, :, 2]) > 1 + 1e-6).any()
    ):
        raise ValueError("applied command exceeds original gait domain")
    # Own prior neural memory is causal even though stored alongside the gait
    # command. Do not use98[6:9] (current label) or9:11 (command-masked phase).
    phase = phase_delta()
    gait_memory = []
    for frame in range(count):
        clock = np.array(
            [math.sin(float(phase) * 2 * math.pi), math.cos(float(phase) * 2 * math.pi)], np.float32
        )
        gait_memory.append(
            np.concatenate((raw["observations98"][frame, :, 69:98], np.tile(clock, (8, 1))), axis=1)
        )
        phase = np.float32(np.fmod(np.float32(phase + phase_delta()), np.float32(1)))
    # Actual next controller memory: recompute just the final frozen ONNX
    # inference and require its HG29 targets to equal the executed targets.
    gait = G1VelocityPolicy(
        models[0],
        configs[0],
        expected_model_hash=pins[str(models[0])],
        expected_config_hash=pins[str(configs[0])],
    )
    final_outputs = []
    for i in range(8):
        output = gait.session.run(["actions"], {"obs": raw["observations98"][-1, i][None]})[0][0]
        if not np.array_equal(target29(output, gait.params), raw["targets29"][-1, i]):
            raise ValueError("final controller memory does not reproduce the executed target")
        final_outputs.append(output.copy())
    final_clock = np.array(
        [math.sin(float(phase) * 2 * math.pi), math.cos(float(phase) * 2 * math.pi)], np.float32
    )
    final_memory = np.concatenate((np.stack(final_outputs), np.tile(final_clock, (8, 1))), axis=1)
    data.qpos[:] = raw["qpos"][-1]
    data.qvel[:] = raw["qvel"][-1]
    mujoco.mj_kinematics(model, data)
    final_context, final_features = [], []
    for i, (team, role, prefix) in enumerate(agents):
        lo = layouts[prefix]
        quat = raw["qpos"][(count - 1) * 10 - 1, lo.rq + 3 : lo.rq + 7]
        yaw = math.atan2(
            2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)
        )
        c, s = math.cos(yaw), math.sin(yaw)
        previous = raw["commands"][-1, i]
        world_previous = np.array(
            [c * previous[0] - s * previous[1], s * previous[0] + c * previous[1]], np.float64
        )
        observation = role_command_observation(
            data, data, layouts, team=team, role=role, previous_body_command=previous
        )
        parent = coached_native_role_label(
            data, data, layouts, team=team, role=role, previous_world_command=world_previous
        )
        final_context.append(parent_context(observation, parent, ROLE_COMMAND_LIMITS))
        final_features.append(
            parent_context(encoder.encode(observation), parent, ROLE_COMMAND_LIMITS)
        )
    context = np.concatenate((raw["residual_context121"], np.stack(gait_memory)), axis=2)[50:]
    next_context = np.concatenate(
        (context[1:], np.concatenate((np.stack(final_context), final_memory), axis=1)[None])
    )
    features = raw["residual_features131"][50:]
    next_features = np.concatenate((features[1:], np.stack(final_features)[None]))
    reward, components, terminations, comotion = [], [], [], []
    age = np.full(8, 999, np.int32)
    for frame in range(50, count):
        before, after = raw["qpos"][frame * 10 - 1], raw["qpos"][(frame + 1) * 10 - 1]
        contact = raw["ball_forces"][frame * 10 : (frame + 1) * 10, :, :2].max(axis=(0, 2)) > 1
        age = np.where(contact, 0, age + 1).astype(np.int32)
        collision = (
            raw["interplayer_contact_forces"][frame * 10 : (frame + 1) * 10].max(axis=(0, 2)) > 1
        )
        failed = []
        for _, _, p in agents:
            lo = layouts[p]
            q = raw["qpos"][frame * 10 : (frame + 1) * 10, lo.rq : lo.rq + 7]
            failed.append(
                bool(((q[:, 2] < 0.65) | (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2) < 0.9)).any())
            )
        r, reward_parts = native_role_residual_reward(
            before_roots=np.stack(
                [before[layouts[p].rq : layouts[p].rq + 2] for _, _, p in agents]
            ),
            after_roots=np.stack([after[layouts[p].rq : layouts[p].rq + 2] for _, _, p in agents]),
            before_ball=before[lo.bq : lo.bq + 2],
            after_ball=after[lo.bq : lo.bq + 2],
            foot_gap=raw["foot_gap_after_m"][frame],
            foot_relative_speed=raw["nearest_foot_relative_speed_mps"][frame],
            contact=contact,
            contact_age=age,
            collision=collision,
            body_failed=np.array(failed),
            raw_action=action[frame],
        )
        reward.append(r)
        components.append(reward_parts)
        terminations.append(np.full(8, any(failed), np.bool_))
        comotion.append(
            (age <= 25)
            & (raw["foot_gap_after_m"][frame] < 0.35)
            & (raw["nearest_foot_relative_speed_mps"][frame] < 0.5)
        )
    if bool(terminations[-1].any()) != any(row["body_failed"].values()):
        raise ValueError("derived native body termination differs from source")
    data_bank = dict(
        context=context,
        next_context=next_context,
        features=features,
        next_features=next_features,
        actions=action[50:],
        rewards=np.array(reward),
        reward_components=np.array(components),
        terminated=np.array(terminations),
        truncated=np.zeros((count - 50, 8), np.bool_),
        applied_commands=raw["commands"][50:],
        parent_commands=raw["residual_parent_commands"][50:],
        close_recent_comotion=np.array(comotion),
    )
    if not terminations[-1].any():
        data_bank["truncated"][-1] = True
    return data_bank, dict(
        source_controls=count,
        transition_rows=(count - 50) * 8,
        role_return=np.array(reward).sum(axis=0).tolist(),
        actual_foot_contact_steps=[p["foot_contact_steps"] for p in row["players"]],
        close_recent_comotion_controls=np.array(comotion).sum(axis=0).tolist(),
        final_onnx_target_recomputation_exact=True,
        sampler_truncated=not bool(terminations[-1].any()),
        next_state_and_controller_memory_reconstructed=True,
        applied_authority_audit_pass=True,
        critic_context_dim=152,
        actor_context_dim=121,
        actor_feature_dim=131,
        current_gait_command_not_a_critic_input=True,
        physical_steps_added=0,
    )
