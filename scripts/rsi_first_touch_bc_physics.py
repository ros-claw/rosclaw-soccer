"""SIM-only own-feedback screening for an offline 29-joint student."""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_mjlab_stage1_training_smoke import file_digest
from scripts.rsi_native_fresh_common import clean


def student_observation(q, v, previous):
    q, v, previous = np.asarray(q), np.asarray(v), np.asarray(previous)
    if q.shape != (43,) or v.shape != (41,) or previous.shape != (29,):
        raise ValueError("actual own canonical state and previous target required")
    result = np.concatenate((q[2:36], v[:35], q[36:39] - q[:3], v[35:38] - v[:3], previous))
    if result.shape != (104,) or not np.isfinite(result).all():
        raise ValueError("finite own-state observation required")
    return result


def phase_entry(frame, delta):
    delta = np.asarray(delta)
    if type(frame) is not int or delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("actual finite ball-relative entry state required")
    return bool(50 <= frame <= 239 and 0 < delta[0] <= 0.9 and abs(delta[1]) <= 0.25)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "training-receipt", "protocol", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("model-hash", "training-hash", "protocol-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--source-hash", action="append", required=True)
    p.add_argument("--declaration-hash", action="append", required=True)
    p.add_argument("--student-reference", type=Path)
    p.add_argument("--student-reference-hash")
    p.add_argument("--phase-policy", action="store_true")
    p.add_argument("--kick-teacher-shadow", action="store_true")
    p.add_argument("--shadow-reference", type=Path)
    p.add_argument("--shadow-reference-hash")
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.source) != 4 or len(a.source_hash) != 4 or len(a.declaration_hash) != 4:
        raise ValueError("exact four source receipts and declarations required")
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    for path, digest in (
        (a.model, a.model_hash),
        (a.training_receipt, a.training_hash),
        (a.protocol, a.protocol_hash),
    ):
        if path.is_symlink() or file_digest(path) != digest:
            raise ValueError("pinned input mismatch")
    trained = json.loads(a.training_receipt.read_text())
    if trained["model_hash"] != a.model_hash or trained["source_hashes"] != a.source_hash:
        raise ValueError("student training provenance mismatch")
    if (a.student_reference is None) != (a.student_reference_hash is None):
        raise ValueError("reference and exact receipt hash must be supplied together")
    if a.phase_policy and (a.student_reference is not None or not trained.get("kick_phase_only")):
        raise ValueError("phase-only trained model and no takeover reference required")
    if a.kick_teacher_shadow != bool(a.shadow_reference) or a.kick_teacher_shadow != bool(
        a.shadow_reference_hash
    ):
        raise ValueError("shadow requires an exact prior student reference")
    shadow_report = None
    if a.kick_teacher_shadow:
        if (
            not a.phase_policy
            or file_digest(a.shadow_reference / "complete.json") != a.shadow_reference_hash
        ):
            raise ValueError("source-bound phase student reference required")
        shadow_report = json.loads((a.shadow_reference / "complete.json").read_text())
        if shadow_report["model_hash"] != a.model_hash or [
            r["course"] for r in shadow_report["rows"]
        ] != list(range(0, 40, 5)):
            raise ValueError("same student and complete eight courses required")
    reference_report = None
    if a.student_reference is not None:
        reference_receipt = a.student_reference / "complete.json"
        if file_digest(reference_receipt) != a.student_reference_hash:
            raise ValueError("exact prior student receipt required")
        reference_report = json.loads(reference_receipt.read_text())
        if (
            reference_report["model_hash"] != a.model_hash
            or reference_report["source_head"] != "8090e7a704fa4ad236bddd377c2bf5313822260f"
            or [row["course"] for row in reference_report["rows"]] != list(range(0, 40, 5))
        ):
            raise ValueError("same student model required")
    sources = []
    for root, receipt_hash, declaration_hash in zip(
        a.source, a.source_hash, a.declaration_hash, strict=True
    ):
        receipt, declaration = root / "complete.json", root / "declaration.json"
        if file_digest(receipt) != receipt_hash or file_digest(declaration) != declaration_hash:
            raise ValueError("source or declaration mismatch")
        sources.append((json.loads(receipt.read_text()), json.loads(declaration.read_text())))
    pins = sources[0][1]["input_file_hashes"]

    def asset(suffix):
        matches = [(Path(path), digest) for path, digest in pins.items() if path.endswith(suffix)]
        if len(matches) != 1 or file_digest(matches[0][0]) != matches[0][1]:
            raise ValueError("unique source-bound simulator asset required")
        return matches[0]

    scene, scene_hash = asset("/cached-standard/declared-scene.xml")
    velocity, velocity_hash = asset("/velocity/v0/exported/policy.onnx")
    config, config_hash = asset("/velocity/v0/params/deploy.yaml")
    checkpoint = torch.load(a.model, map_location="cpu", weights_only=True)
    dimension = 105 if a.phase_policy else 104
    if checkpoint["feature_dim"] != dimension:
        raise ValueError("qualified observation dimension required")
    net = torch.nn.Sequential(
        torch.nn.Linear(dimension, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 29),
    )
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    has_bounds = "target_lower" in checkpoint or "target_upper" in checkpoint
    if has_bounds and (
        not a.phase_policy
        or not {"target_lower", "target_upper", "codec_hash"}.issubset(checkpoint)
    ):
        raise ValueError("complete phase policy codec bounds required")
    mean, scale = checkpoint["mean"].numpy(), checkpoint["scale"].numpy()
    if (
        mean.shape != (dimension,)
        or scale.shape != (dimension,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("valid frozen training normalization required")
    import mujoco
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.rsi.kick_reference_phase import nearest_kick_reference_phase
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.observed_velocity_continuation import ObservedVelocityContinuation
    from rosclaw_soccer.rsi.shared_kick_history import measured_kick_sample
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
    from rosclaw_soccer.rsi.walk_kick_approach_command import approach_command

    model = mujoco.MjModel.from_xml_string(shared_native_scene(scene.read_text(), prefixes=("a/",)))
    kick_types = None
    kick_path = None
    kick_hash = None
    kick_dependencies = []
    if a.phase_policy:
        from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo

        kick_path, kick_hash = asset("/policy/robonaldo/FreeKick.py")
        for suffix in (
            "/FSM/FSMState.py",
            "/common/ctrlcomp.py",
            "/common/utils.py",
            "/common/path_config.py",
            "/policy/robonaldo/config/freekick.yaml",
            "/policy/robonaldo/model/freekick_motion.npz",
            "/policy/robonaldo/model/policy-obs-aic.onnx",
        ):
            kick_dependencies.append(asset(suffix))
        if has_bounds:
            import yaml

            from rosclaw_soccer.rsi.bounded_kick_decoder import kick_target_bounds

            codec_path, codec_hash = asset("/policy/robonaldo/config/freekick.yaml")
            expected_bounds = kick_target_bounds(yaml.safe_load(codec_path.read_text()))
            if (
                checkpoint["codec_hash"] != codec_hash
                or trained["bounded_codec_hash"] != codec_hash
                or not all(
                    np.array_equal(checkpoint[key].numpy(), expected)
                    for key, expected in zip(
                        ("target_lower", "target_upper"), expected_bounds, strict=True
                    )
                )
            ):
                raise ValueError("student bounds differ from actual source kick codec")
        kick_types = _load_robonaldo(kick_path.parents[2])
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            model_hash=a.model_hash,
            training_hash=a.training_hash,
            protocol_hash=a.protocol_hash,
            source_hashes=a.source_hash,
            scene_hash=scene_hash,
            velocity_hash=velocity_hash,
            config_hash=config_hash,
            student_reference_hash=a.student_reference_hash,
            student_prefix_controls=20 if reference_report is not None else None,
            phase_policy=a.phase_policy,
            kick_source_hash=kick_hash,
            kick_dependency_hashes={str(path): digest for path, digest in kick_dependencies},
            shadow_reference_hash=a.shadow_reference_hash,
            courses=list(range(0, 40, 5)),
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    rows = []
    parent_pins = []
    reference_pins = []
    try:
        for i in range(0, 40, 5):
            shard = i % 4
            source, _ = sources[shard]
            matches = [
                row for row in source["rows"] if row["course_index"] == i and row["arm"] == "parent"
            ]
            if len(matches) != 1:
                raise ValueError("one original parent required per course")
            parent = matches[0]
            parent_path = a.source[shard] / f"course-{i:02d}-parent.npz"
            if file_digest(parent_path) != parent["raw_hash"]:
                raise ValueError("original parent raw hash mismatch")
            parent_pins.append((parent_path, parent["raw_hash"]))
            stream = ContinuousVelocityStream(
                velocity,
                config,
                expected_model_hash=velocity_hash,
                expected_config_hash=config_hash,
                horizon_controls=360,
            )
            shadow = ObservedVelocityContinuation(stream) if reference_report is not None else None
            data = mujoco.MjData(model)
            lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
            x, y, vx = parent["result"]["course"]
            data.qpos[lo.rq : lo.rq + 7] = [0, 0, 0.793, 1, 0, 0, 0]
            data.qpos[lo.qi] = np.asarray(stream.params.defaults)
            data.qpos[lo.bq : lo.bq + 7] = [x, y, 0.11, 1, 0, 0, 0]
            data.qvel[lo.bv] = vx
            data.qvel[lo.bv + 4] = vx / 0.11
            mujoco.mj_forward(model, data)
            pd = NativePDParameters(
                np.array(stream.params.stiffness),
                np.array(stream.params.damping),
                0.85 * np.asarray(G1_HARD_TORQUE_LIMITS),
            )
            world = SharedNativeMotorWorld(
                model, data, players={"a/": pd}, ball_body_name="soccer_ball"
            )
            q, v = lo.canonical(data)
            command = approach_command(
                q=q,
                v=v,
                rotation=data.xmat[lo.pelvis].reshape(3, 3),
                cached_ball_position=data.xpos[lo.ball],
                previous=np.zeros(3, np.float32),
                parameters=parent["result"]["parameters"],
                fast=False,
                feedback=False,
                halt=False,
            )
            target = (
                shadow.propose(q, v, lo.angular_velocity_body(data), command, frame=0)
                if shadow is not None
                else stream.propose(q, v, lo.angular_velocity_body(data), command, frame=0)
            )
            with np.load(parent_path, allow_pickle=False) as reference:
                if not np.array_equal(target, reference["joint_targets"][0]):
                    raise ValueError("initial foundation action does not match source parent")
                first = world.advance({"a/": target})
                for key in ("qpos", "qvel", "torque", "ball_forces"):
                    if not np.array_equal(first[key], reference[key][:10]):
                        raise ValueError("initial native physical prefix differs from parent")
            if shadow is not None:
                shadow.observe_executed(target.astype(np.float64), completed_control=1)
                ref_rows = [r for r in reference_report["rows"] if r["course"] == i]
                if len(ref_rows) != 1:
                    raise ValueError("one original student course required")
                ref_path = a.student_reference / f"course-{i:02d}-student.npz"
                if file_digest(ref_path) != ref_rows[0]["artifact_hash"]:
                    raise ValueError("exact prior student trajectory required")
                reference_pins.append((ref_path, ref_rows[0]["artifact_hash"]))
                with np.load(ref_path, allow_pickle=False) as old:
                    prefix = {k: old[k].copy() for k in old.files}
                if len(prefix["joint_targets"]) < 20 or not np.array_equal(
                    target, prefix["joint_targets"][0]
                ):
                    raise ValueError("original student initial action required")
            previous_command = command.copy()
            shadow_targets = [target.copy()] if shadow is not None else []
            shadow_commands = [command.copy()] if shadow is not None else []
            switch_frame = None
            recorded = {k: [first[k]] for k in ("qpos", "qvel", "torque", "ball_forces")}
            targets = [target.copy()]
            inputs = []
            history = deque(maxlen=5)
            if a.kick_teacher_shadow and not world.failed:
                history.append(measured_kick_sample(world, "a/"))
            kick_shadow = None
            teacher_proposals = []
            shadow_receipt = None
            selected_reference_phase = None
            for frame in range(1, 360):
                if world.failed:
                    break
                q, v = lo.canonical(data)
                if a.phase_policy and switch_frame is None:
                    delta = data.xmat[lo.pelvis].reshape(3, 3).T @ (data.xpos[lo.ball] - q[:3])
                    if phase_entry(frame, delta):
                        assert kick_types is not None
                        state_type, output_type, policy_type, _ = kick_types
                        kick = policy_type(state_type(29), output_type(29))
                        kp = NativePDParameters(np.array(kick.kps), np.array(kick.kds), pd.effort)
                        if not world.switch_pd_profile("a/", kp)["physics_unchanged"]:
                            raise ValueError("kick gain transition changed physical state")
                        switch_frame = frame
                        if a.kick_teacher_shadow:
                            _, _, _, mapping = kick_types
                            selected_reference_phase = nearest_kick_reference_phase(
                                q[7:36],
                                v[6:35],
                                kick.motion_joint_pos[:, mapping],
                                kick.motion_joint_vel[:, mapping],
                            )["reference_frame"]
                            kick_shadow = SharedRoboNaldoPlayer(
                                world, "a/", kick, goal=np.array([8.0, 0.0, 0.11])
                            )
                            shadow_receipt = kick_shadow.install_measured_entry(
                                tuple(history), reference_frame=selected_reference_phase
                            )
                if shadow is not None:
                    command = approach_command(
                        q=q,
                        v=v,
                        rotation=data.xmat[lo.pelvis].reshape(3, 3),
                        cached_ball_position=data.xpos[lo.ball],
                        previous=previous_command,
                        parameters=parent["result"]["parameters"],
                        fast=False,
                        feedback=False,
                        halt=False,
                    )
                    previous_command = command.copy()
                    proposal = shadow.propose(
                        q, v, lo.angular_velocity_body(data), command, frame=frame
                    )
                    shadow_targets.append(proposal.copy())
                    shadow_commands.append(command.copy())
                observation = student_observation(q, v, world.last_applied_target("a/"))
                if a.phase_policy and switch_frame is None:
                    command = approach_command(
                        q=q,
                        v=v,
                        rotation=data.xmat[lo.pelvis].reshape(3, 3),
                        cached_ball_position=data.xpos[lo.ball],
                        previous=previous_command,
                        parameters=parent["result"]["parameters"],
                        fast=False,
                        feedback=False,
                        halt=False,
                    )
                    previous_command = command.copy()
                    target = stream.propose(
                        q, v, lo.angular_velocity_body(data), command, frame=frame
                    )
                else:
                    if a.phase_policy:
                        observation = np.r_[observation, (frame - switch_frame) * 0.02]
                    with torch.no_grad():
                        logits = net(
                            torch.tensor((observation - mean) / scale, dtype=torch.float32)
                        )
                        if has_bounds:
                            from rosclaw_soccer.rsi.bounded_kick_decoder import decode_bounded

                            logits = decode_bounded(
                                logits, checkpoint["target_lower"], checkpoint["target_upper"]
                            )
                        target = logits.numpy().astype(np.float64)
                if shadow is not None and frame >= 20:
                    target = proposal
                if target.shape != (29,) or not np.isfinite(target).all():
                    raise ValueError("finite student target required")
                if not a.phase_policy or switch_frame is not None:
                    inputs.append(observation)
                if kick_shadow is not None:
                    teacher_target, _ = kick_shadow.propose()
                    teacher_proposals.append(teacher_target.copy())
                result = world.advance({"a/": target})
                if kick_shadow is not None:
                    kick_shadow.commit_observed_target()
                if a.kick_teacher_shadow and not world.failed:
                    history.append(measured_kick_sample(world, "a/"))
                if a.phase_policy and switch_frame is None:
                    with np.load(parent_path, allow_pickle=False) as original:
                        if not np.array_equal(target, original["joint_targets"][frame]):
                            raise ValueError("original approach target mismatch")
                        for key in ("qpos", "qvel", "torque", "ball_forces"):
                            if not np.array_equal(
                                result[key], original[key][frame * 10 : (frame + 1) * 10]
                            ):
                                raise ValueError("original approach physics mismatch")
                if shadow is not None:
                    shadow.observe_executed(target.astype(np.float64), completed_control=frame + 1)
                    if frame < 20:
                        if not np.array_equal(target, prefix["joint_targets"][frame]):
                            raise ValueError("student action prefix mismatch")
                        for key in ("qpos", "qvel", "torque", "ball_forces"):
                            if not np.array_equal(
                                result[key], prefix[key][frame * 10 : (frame + 1) * 10]
                            ):
                                raise ValueError("student native prefix mismatch")
                for key in recorded:
                    recorded[key].append(result[key])
                targets.append(target.copy())
            arrays = {key: np.concatenate(value) for key, value in recorded.items()}
            n = len(targets)
            arrays.update(
                joint_targets=np.stack(targets),
                commands=np.zeros((n, 3)),
                ball_position_after_step_m=arrays["qpos"].reshape(n, 10, 43)[
                    :, -1, lo.bq : lo.bq + 3
                ],
                pelvis_z_per_substep_m=arrays["qpos"].reshape(n, 10, 43)[:, :, lo.rq + 2],
                student_observations=np.asarray(inputs).reshape(-1, dimension),
            )
            if shadow is not None:
                arrays["shadow_targets"] = np.stack(shadow_targets)
                arrays["shadow_commands"] = np.stack(shadow_commands)
            if a.kick_teacher_shadow:
                arrays["kick_teacher_targets"] = np.asarray(teacher_proposals).reshape(-1, 29)
                ref_rows = [r for r in shadow_report["rows"] if r["course"] == i]
                if len(ref_rows) != 1:
                    raise ValueError("one original bounded student trajectory required")
                ref_path = a.shadow_reference / f"course-{i:02d}-student.npz"
                if file_digest(ref_path) != ref_rows[0]["artifact_hash"]:
                    raise ValueError("original bounded student artifact mismatch")
                reference_pins.append((ref_path, ref_rows[0]["artifact_hash"]))
                with np.load(ref_path, allow_pickle=False) as ref:
                    if any(
                        not np.array_equal(arrays[k], ref[k], equal_nan=True) for k in ref.files
                    ):
                        raise ValueError("teacher shadow altered actual student trajectory")
            label = relabel_arrays(arrays, controls=n, ball_address=lo.bq, root_address=lo.rq)
            path = a.output / f"course-{i:02d}-student.npz"
            np.savez_compressed(path, **arrays)
            rows.append(
                dict(
                    course=i,
                    source_parent_hash=parent["raw_hash"],
                    parent_outcome=parent["result"]["outcome"],
                    controls=n,
                    phase_switch_frame=switch_frame,
                    teacher_reference_phase=selected_reference_phase,
                    teacher_history_receipt=shadow_receipt,
                    healthy_full_horizon=not world.failed and n == 360,
                    student_outcome=label,
                    artifact_hash=file_digest(path),
                )
            )
        clean(a.expected_head)
        if (
            file_digest(a.model) != a.model_hash
            or file_digest(a.training_receipt) != a.training_hash
            or file_digest(a.protocol) != a.protocol_hash
            or any(file_digest(path) != digest for path, digest in parent_pins)
            or any(file_digest(path) != digest for path, digest in reference_pins)
            or (kick_path is not None and file_digest(kick_path) != kick_hash)
            or any(file_digest(path) != digest for path, digest in kick_dependencies)
            or (
                a.shadow_reference is not None
                and file_digest(a.shadow_reference / "complete.json") != a.shadow_reference_hash
            )
            or (
                a.student_reference is not None
                and file_digest(a.student_reference / "complete.json") != a.student_reference_hash
            )
            or any(
                file_digest(root / "complete.json") != receipt_hash
                or file_digest(root / "declaration.json") != declaration_hash
                for root, receipt_hash, declaration_hash in zip(
                    a.source, a.source_hash, a.declaration_hash, strict=True
                )
            )
            or any(
                file_digest(path) != digest
                for path, digest in (
                    (scene, scene_hash),
                    (velocity, velocity_hash),
                    (config, config_hash),
                )
            )
        ):
            raise ValueError("student input changed during physics")
        write_once(
            a.output / "complete.json",
            dict(
                status="OWN_FEEDBACK_SINGLE_PLAYER_DEVELOPMENT_ONLY",
                rows=rows,
                source_head=a.expected_head,
                protocol_hash=a.protocol_hash,
                model_hash=a.model_hash,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error_type=type(error).__name__, error=str(error), completed_rows=rows),
        )
        raise


if __name__ == "__main__":
    main()
