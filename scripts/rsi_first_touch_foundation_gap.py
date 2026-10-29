"""Audit successful teacher trajectories against a causally warmed foundation."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_first_touch_bc_physics import student_observation
from scripts.rsi_mjlab_stage1_training_smoke import file_digest
from scripts.rsi_native_fresh_common import clean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("training-receipt", "protocol", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("training-hash", "protocol-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--declaration-hash", required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.source) != 4 or a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("four sources and unused external output required")
    declaration = a.source[0] / "declaration.json"
    for path, expected in (
        (a.training_receipt, a.training_hash),
        (a.protocol, a.protocol_hash),
        (declaration, a.declaration_hash),
    ):
        if path.is_symlink() or file_digest(path) != expected:
            raise ValueError("source provenance mismatch")
    trained = json.loads(a.training_receipt.read_text())
    sources = []
    for root, expected in zip(a.source, trained["source_hashes"], strict=True):
        path = root / "complete.json"
        if file_digest(path) != expected:
            raise ValueError("original source mismatch")
        sources.append(json.loads(path.read_text()))
    pins = json.loads(declaration.read_text())["input_file_hashes"]

    def asset(suffix):
        found = [(Path(path), digest) for path, digest in pins.items() if path.endswith(suffix)]
        if len(found) != 1 or file_digest(found[0][0]) != found[0][1]:
            raise ValueError("unique frozen asset required")
        return found[0]

    scene, scene_hash = asset("/cached-standard/declared-scene.xml")
    velocity, velocity_hash = asset("/velocity/v0/exported/policy.onnx")
    config, config_hash = asset("/velocity/v0/params/deploy.yaml")
    import mujoco

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.observed_velocity_continuation import ObservedVelocityContinuation
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene

    model = mujoco.MjModel.from_xml_string(shared_native_scene(scene.read_text(), prefixes=("a/",)))
    a.output.mkdir(parents=True)
    rows, raw_pins = [], []
    for shard, source in enumerate(sources):
        for meta in source["rows"]:
            i, arm, info = meta["course_index"], meta["arm"], meta["result"]
            if i not in range(shard, 40, 4) or arm not in ("parent", "retrieval"):
                raise ValueError("unexpected course membership")
            raw_path = a.source[shard] / f"course-{i:02d}-{arm}.npz"
            if raw_path.is_symlink() or file_digest(raw_path) != meta["raw_hash"]:
                raise ValueError("original physics hash mismatch")
            raw_pins.append((raw_path, meta["raw_hash"]))
            with np.load(raw_path, allow_pickle=False) as archive:
                raw = {k: archive[k] for k in archive.files}
            label = relabel_arrays(
                raw,
                controls=info["controls"],
                ball_address=info["ball_qpos_address"],
                root_address=info["robot_qpos_address"],
            )
            if any(
                label[k] != info["outcome"][k]
                for k in ("safe", "task_pass", "first_foot_contact_frame")
            ):
                raise ValueError("original label mismatch")
            admitted = bool(label["safe"] and label["task_pass"] and info["healthy_full_horizon"])
            row = dict(
                course=i,
                arm=arm,
                admitted=admitted,
                split="validation" if i % 5 == 0 else "train",
                original_label=label,
                healthy_full_horizon=info["healthy_full_horizon"],
            )
            rows.append(row)
            if not admitted:
                continue
            try:
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
                proposals, observations, errors = [], [], []
                for frame, target in enumerate(raw["joint_targets"]):
                    if frame:
                        data.qpos[:] = raw["qpos"][10 * frame - 2]
                        data.qvel[:] = raw["qvel"][10 * frame - 2]
                        mujoco.mj_forward(model, data)
                        data.qpos[:] = raw["qpos"][10 * frame - 1]
                        data.qvel[:] = raw["qvel"][10 * frame - 1]
                    else:
                        mujoco.mj_forward(model, data)
                    q, v = lo.canonical(data)
                    command = raw["commands"][frame]
                    proposal = shadow.propose(
                        q, v, lo.angular_velocity_body(data), command, frame=frame
                    )
                    target = target.astype(np.float64)
                    if frame == 0 and not np.array_equal(target, proposal):
                        raise ValueError("original initial foundation target mismatch")
                    observations.append(
                        student_observation(q, v, raw["joint_targets"][frame - 1])
                        if frame
                        else np.zeros(104)
                    )
                    note = shadow.observe_executed(target, completed_control=frame + 1)
                    proposals.append(proposal)
                    errors.append(note["maximum_target_decode_error_rad"])
                proposal = np.stack(proposals)
                output = a.output / f"course-{i:02d}-{arm}.npz"
                np.savez_compressed(
                    output,
                    observation=np.stack(observations),
                    observation_valid=np.arange(len(observations)) > 0,
                    foundation_target=proposal,
                    teacher_target=raw["joint_targets"],
                    commands=raw["commands"],
                    velocity_control_active=raw["velocity_control_active"],
                    codec_error=np.asarray(errors),
                )
                delta = np.abs(raw["joint_targets"] - proposal)
                row.update(
                    query_status="COMPLETE",
                    artifact_hash=file_digest(output),
                    frame_count=len(proposal),
                    first_foot_contact_frame=label["first_foot_contact_frame"],
                    max_abs_delta_rad=float(delta.max()),
                    mean_joint_abs_delta_rad=float(delta.mean()),
                    full_vector_within_0_1_fraction=float(np.mean(np.max(delta, axis=1) <= 0.1)),
                    full_vector_within_0_2_fraction=float(np.mean(np.max(delta, axis=1) <= 0.2)),
                    max_codec_error_rad=float(np.max(errors)),
                )
            except Exception as error:
                row.update(query_status="FAILED", error_type=type(error).__name__, error=str(error))
    if len(rows) != 80 or len({(r["course"], r["arm"]) for r in rows}) != 80:
        raise ValueError("all 80 unique source episodes required")
    clean(a.expected_head)
    if (
        file_digest(a.training_receipt) != a.training_hash
        or file_digest(a.protocol) != a.protocol_hash
        or file_digest(declaration) != a.declaration_hash
        or any(file_digest(path) != h for path, h in raw_pins)
        or any(
            file_digest(root / "complete.json") != h
            for root, h in zip(a.source, trained["source_hashes"], strict=True)
        )
        or any(
            file_digest(path) != h
            for path, h in ((scene, scene_hash), (velocity, velocity_hash), (config, config_hash))
        )
    ):
        raise ValueError("source changed during audit")
    write_once(
        a.output / "complete.json",
        dict(
            status="OFFLINE_TEACHER_FOUNDATION_GAP_NOT_PHYSICS",
            source_head=a.expected_head,
            training_hash=a.training_hash,
            protocol_hash=a.protocol_hash,
            rows=rows,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
