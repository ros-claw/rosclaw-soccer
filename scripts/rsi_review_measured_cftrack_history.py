"""Independently rebuild scratch CFtrack history from immutable SIM trajectory."""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.source / "complete.json"
    if (
        a.output.exists()
        or complete.is_symlink()
        or digest(complete) != a.source_hash
        or any(n in sys.modules for n in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused output, exact source and isolated import required")
    source = json.loads(complete.read_text())
    if (
        source["status"] != "MEASURED_CFTRACK_HISTORY_CANDIDATES_NOT_INSTALLED"
        or source["network_calls"] != 0
        or source["native_steps"] != 0
        or [x["entry_control"] for x in source["records"]] != [57, 82]
    ):
        raise ValueError("exact two scratch candidates with no prior inference required")
    candidate = a.source / source["artifact"]["path"]
    if (
        candidate.is_symlink()
        or candidate.parent != a.source
        or digest(candidate) != source["artifact"]["hash"]
    ):
        raise ValueError("exact confined candidate artifact required")
    pins = dict(source["input_file_hashes"])
    pins.update({str(complete): a.source_hash, str(candidate): source["artifact"]["hash"]})
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    external_files = [Path(x) for x in pins if x.endswith("/policy/omnicontact/OmniContact.py")]
    traces = [Path(x) for x in pins if x.endswith("/rsi-warm-left-settle-v3-20261027/live.npz")]
    if not (len(scenes) == len(external_files) == len(traces) == 1):
        raise ValueError("one pinned scene, policy and source physical trace required")
    external = external_files[0].parents[2]
    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if compiled_model_hash(model) != (
        "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("canonical SIM model required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from common.utils import matrix_from_quat, subtract_frame_transforms, yaw_quat
    from policy.omnicontact.OmniContact import OmniContact

    policy = OmniContact(
        StateAndCmd(29),
        PolicyOutput(29),
        onnx_path=str(external / "policy/omnicontact/model/kick_50k.onnx"),
    )
    if tuple(policy.kinematics.joint_names) != tuple(G1_DDS_JOINT_NAMES):
        raise ValueError("canonical external FK joint order required")
    policy.task, policy.reference_source, policy.active_object_name = "kickball", "CFgen", "ball"
    policy.box_dims = policy.ball_dims = np.full(3, 0.11, np.float32)
    policy.bbox_scale = policy.box_dims * 2
    policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)
    with np.load(traces[0], allow_pickle=False) as f:
        raw = {k: f[k].copy() for k in f.files}
    with np.load(candidate, allow_pickle=False) as f:
        saved = {k: f[k].copy() for k in f.files}
    if raw["qpos"].shape != (1570, model.nq) or raw["joint_targets"].shape != (157, 29):
        raise ValueError("complete physical and executed target trace required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.source_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            independent_state_target_and_history_rebuild=True,
            source_fk_and_quaternion_helpers_shared=True,
            scratch_history_only=True,
            network_calls=0,
            native_steps=0,
            hardware_authorized=False,
        ),
    )
    records = []
    for entry in (57, 82):
        data = mujoco.MjData(model)
        rows = []
        for frame in range(entry - 4, entry + 1):
            data.qpos[:] = raw["qpos"][frame * 10 - 2]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            ball_pos = data.xpos[lo.ball].copy()
            ball_quat = data.xquat[lo.ball].copy()
            data.qpos[:] = raw["qpos"][frame * 10 - 1]
            data.qvel[:] = raw["qvel"][frame * 10 - 1]
            q, v = lo.canonical(data)
            s = policy.state_cmd
            s.q, s.dq = q[7:36].copy(), v[6:35].copy()
            s.base_pos, s.base_quat = q[:3].copy(), q[3:7].copy()
            s.ang_vel = v[3:6].copy()
            w, x, y, z = s.base_quat
            s.gravity_ori = np.array(
                [2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)], np.float32
            )
            s.obj_pos, s.obj_quat = ball_pos, ball_quat
            policy._get_fk_info()
            heading = yaw_quat(policy.torso_quat)
            obj_pos, _ = subtract_frame_transforms(policy.torso_pos, heading, ball_pos, ball_quat)
            obj_pos = policy._clip_norm(obj_pos)
            obj_rot = matrix_from_quat(np.array([1, 0, 0, 0], np.float32))[:, :2].reshape(-1)
            previous_target = raw["joint_targets"][frame - 1]
            action = (
                (previous_target[policy.mj2lab] - policy.default_angles_lab)
                / policy.action_scale_lab
            ).astype(np.float32)
            obs = np.concatenate(
                [
                    policy.ee_pos,
                    s.ang_vel.astype(np.float32),
                    s.gravity_ori,
                    (s.q[policy.mj2lab] - policy.default_angles_lab).astype(np.float32),
                    s.dq[policy.mj2lab].astype(np.float32),
                    action,
                    obj_pos.astype(np.float32),
                    obj_rot.astype(np.float32),
                    policy._build_bbox_rel(heading),
                ]
            )
            if not np.array_equal(obs, saved[f"entry{entry}_observations"][frame - (entry - 4)]):
                raise ValueError(f"reconstructed source observation mismatch at {frame}")
            rows.append(obs)
        policy.obs_history_buffer[:] = np.stack(rows)
        if not np.array_equal(policy._flatten_obs_history(), saved[f"entry{entry}_flattened"]):
            raise ValueError("source flatten mismatch")
        policy.obs_history_buffer.fill(0)
        if data.time != 0 or policy.counter_step != 0:
            raise ValueError("scratch analysis must not integrate or infer")
        records.append(
            dict(
                entry_control=entry,
                frames=list(range(entry - 4, entry + 1)),
                all_current_vectors_exact=True,
                source_history_flatten_exact=True,
            )
        )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="MEASURED_CFTRACK_HISTORY_REBUILT_EXACT_NOT_INSTALLED",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            records=records,
            source_fk_and_quaternion_helpers_shared=True,
            scratch_history_only=True,
            network_calls=0,
            native_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
