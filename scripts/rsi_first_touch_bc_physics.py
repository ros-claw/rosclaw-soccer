"""SIM-only own-feedback screening for an offline 29-joint student."""

import argparse
import json
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
    if checkpoint["feature_dim"] != 104:
        raise ValueError("qualified observation dimension required")
    net = torch.nn.Sequential(
        torch.nn.Linear(104, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 29),
    )
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    mean, scale = checkpoint["mean"].numpy(), checkpoint["scale"].numpy()
    if (
        mean.shape != (104,)
        or scale.shape != (104,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("valid frozen training normalization required")
    import mujoco
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.observed_velocity_continuation import ObservedVelocityContinuation
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_approach_command import approach_command

    model = mujoco.MjModel.from_xml_string(shared_native_scene(scene.read_text(), prefixes=("a/",)))
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
            recorded = {k: [first[k]] for k in ("qpos", "qvel", "torque", "ball_forces")}
            targets = [target.copy()]
            inputs = []
            for frame in range(1, 360):
                if world.failed:
                    break
                q, v = lo.canonical(data)
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
                with torch.no_grad():
                    target = (
                        net(torch.tensor((observation - mean) / scale, dtype=torch.float32))
                        .numpy()
                        .astype(np.float64)
                    )
                if shadow is not None and frame >= 20:
                    target = proposal
                if target.shape != (29,) or not np.isfinite(target).all():
                    raise ValueError("finite student target required")
                inputs.append(observation)
                result = world.advance({"a/": target})
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
                student_observations=np.asarray(inputs).reshape(-1, 104),
            )
            if shadow is not None:
                arrays["shadow_targets"] = np.stack(shadow_targets)
                arrays["shadow_commands"] = np.stack(shadow_commands)
            label = relabel_arrays(arrays, controls=n, ball_address=lo.bq, root_address=lo.rq)
            path = a.output / f"course-{i:02d}-student.npz"
            np.savez_compressed(path, **arrays)
            rows.append(
                dict(
                    course=i,
                    source_parent_hash=parent["raw_hash"],
                    parent_outcome=parent["result"]["outcome"],
                    controls=n,
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
