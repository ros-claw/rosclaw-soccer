"""Query frozen velocity foundation on recorded failed student states, SIM-only."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_mjlab_stage1_training_smoke import file_digest
from scripts.rsi_native_fresh_common import clean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("student", "protocol", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--previous-substep-kinematics", action="store_true")
    for key in (
        "student-hash",
        "student-declaration-hash",
        "declaration-hash",
        "protocol-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    if len(a.source) != 4:
        raise ValueError("four original course sources required")
    receipt, declaration = a.student / "complete.json", a.source[0] / "declaration.json"
    student_declaration_path = a.student / "declaration.json"
    for path, digest in (
        (receipt, a.student_hash),
        (student_declaration_path, a.student_declaration_hash),
        (declaration, a.declaration_hash),
        (a.protocol, a.protocol_hash),
    ):
        if path.is_symlink() or file_digest(path) != digest:
            raise ValueError("exact pinned provenance required")
    report = json.loads(receipt.read_text())
    if report["source_head"] != "8090e7a704fa4ad236bddd377c2bf5313822260f" or [
        r["course"] for r in report["rows"]
    ] != list(range(0, 40, 5)):
        raise ValueError("all eight declared physical student courses required")
    student_declaration = json.loads(student_declaration_path.read_text())
    sources = []
    for root, digest in zip(a.source, student_declaration["source_hashes"], strict=True):
        source_path = root / "complete.json"
        if file_digest(source_path) != digest:
            raise ValueError("original paired source mismatch")
        sources.append(json.loads(source_path.read_text()))
    pins = json.loads(declaration.read_text())["input_file_hashes"]

    def asset(suffix):
        found = [(Path(path), digest) for path, digest in pins.items() if path.endswith(suffix)]
        if len(found) != 1 or file_digest(found[0][0]) != found[0][1]:
            raise ValueError("unique qualified asset required")
        return found[0]

    scene, scene_hash = asset("/cached-standard/declared-scene.xml")
    velocity, velocity_hash = asset("/velocity/v0/exported/policy.onnx")
    config, config_hash = asset("/velocity/v0/params/deploy.yaml")
    import mujoco

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.observed_velocity_continuation import ObservedVelocityContinuation
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_approach_command import approach_command

    model = mujoco.MjModel.from_xml_string(shared_native_scene(scene.read_text(), prefixes=("a/",)))
    a.output.mkdir(parents=True)
    rows = []
    raw_pins = []
    try:
        for row in report["rows"]:
            i = row["course"]
            path = a.student / f"course-{i:02d}-student.npz"
            if file_digest(path) != row["artifact_hash"]:
                raise ValueError("student physical trajectory mismatch")
            raw_pins.append((path, row["artifact_hash"]))
            with np.load(path, allow_pickle=False) as archive:
                raw = {k: archive[k] for k in archive.files}
            matches = [
                r for r in sources[i % 4]["rows"] if r["course_index"] == i and r["arm"] == "parent"
            ]
            if len(matches) != 1 or matches[0]["raw_hash"] != row["source_parent_hash"]:
                raise ValueError("course metadata does not bind original parent")
            info = matches[0]["result"]
            stream = ContinuousVelocityStream(
                velocity,
                config,
                expected_model_hash=velocity_hash,
                expected_config_hash=config_hash,
                horizon_controls=360,
            )
            shadow = ObservedVelocityContinuation(stream)
            data = mujoco.MjData(model)
            lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
            x, y, vx = info["course"]
            data.qpos[lo.rq : lo.rq + 7] = [0, 0, 0.793, 1, 0, 0, 0]
            data.qpos[lo.qi] = np.asarray(stream.params.defaults)
            data.qpos[lo.bq : lo.bq + 7] = [x, y, 0.11, 1, 0, 0, 0]
            data.qvel[lo.bv] = vx
            data.qvel[lo.bv + 4] = vx / 0.11
            previous_command = np.zeros(3, np.float32)
            proposals, commands, errors = [], [], []
            n = len(raw["joint_targets"])
            for frame in range(n):
                if frame and a.previous_substep_kinematics:
                    data.qpos[:] = raw["qpos"][10 * frame - 2]
                    data.qvel[:] = raw["qvel"][10 * frame - 2]
                    mujoco.mj_forward(model, data)
                    data.qpos[:] = raw["qpos"][10 * frame - 1]
                    data.qvel[:] = raw["qvel"][10 * frame - 1]
                elif frame:
                    data.qpos[:] = raw["qpos"][10 * frame - 1]
                    data.qvel[:] = raw["qvel"][10 * frame - 1]
                    mujoco.mj_forward(model, data)
                else:
                    mujoco.mj_forward(model, data)
                q, v = lo.canonical(data)
                command = approach_command(
                    q=q,
                    v=v,
                    rotation=data.xmat[lo.pelvis].reshape(3, 3),
                    cached_ball_position=data.xpos[lo.ball],
                    previous=previous_command,
                    parameters=info["parameters"],
                    fast=False,
                    feedback=False,
                    halt=False,
                )
                previous_command = command.copy()
                proposal = shadow.propose(
                    q, v, lo.angular_velocity_body(data), command, frame=frame
                )
                actual = raw["joint_targets"][frame].astype(np.float64)
                if frame == 0 and not np.array_equal(proposal, actual):
                    raise ValueError("initial foundation target mismatch")
                note = shadow.observe_executed(actual, completed_control=frame + 1)
                proposals.append(proposal)
                commands.append(command)
                errors.append(note["maximum_target_decode_error_rad"])
            proposals = np.stack(proposals)
            path_out = a.output / f"course-{i:02d}-teacher-query.npz"
            np.savez_compressed(
                path_out,
                proposed=proposals,
                actual=raw["joint_targets"],
                commands=np.stack(commands),
                codec_error=np.asarray(errors),
            )
            delta = proposals - raw["joint_targets"]
            rows.append(
                dict(
                    course=i,
                    frames=n,
                    artifact_hash=file_digest(path_out),
                    mean_l2_rad=float(np.linalg.norm(delta, axis=1).mean()),
                    max_coordinate_rad=float(np.abs(delta).max()),
                    max_codec_error_rad=float(np.max(errors)),
                    physical_teacher_recovery_verified=False,
                )
            )
        clean(a.expected_head)
        if (
            file_digest(receipt) != a.student_hash
            or file_digest(student_declaration_path) != a.student_declaration_hash
            or file_digest(declaration) != a.declaration_hash
            or file_digest(a.protocol) != a.protocol_hash
            or any(file_digest(path) != digest for path, digest in raw_pins)
            or any(
                file_digest(root / "complete.json") != digest
                for root, digest in zip(a.source, student_declaration["source_hashes"], strict=True)
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
            raise ValueError("input changed during query")
        write_once(
            a.output / "complete.json",
            dict(
                status="OFFLINE_TEACHER_QUERIES_NOT_PHYSICS",
                source_head=a.expected_head,
                student_hash=a.student_hash,
                protocol_hash=a.protocol_hash,
                previous_substep_kinematics=a.previous_substep_kinematics,
                rows=rows,
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
