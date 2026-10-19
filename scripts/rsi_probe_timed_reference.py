"""Two-arm H300 native tracking feasibility, not a football exam or training.

Requires training+diagnostics extras and external, separately licensed data.
Pins sources before importing them. All failed trajectories remain evidence.
"""

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("motion-csv", "license", "world", "model-root", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-motion-hash", required=True)
    parser.add_argument("--expected-world-hash", required=True)
    args = parser.parse_args()
    source = Path.cwd().resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists():
        raise ValueError("sealed source HEAD and unused output required")
    paths = [
        p
        for folder in (source / "src", source / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ] + [
        args.motion_csv,
        args.license,
        args.world,
        args.model_root / "planner_sonic.onnx",
        args.model_root / "low_latency/model_encoder.onnx",
        args.model_root / "low_latency/model_decoder.onnx",
        args.model_root / "low_latency/observation_config.yaml",
    ]
    pins = {str(p): digest(p) for p in paths}
    if (
        pins[str(args.motion_csv)] != args.expected_motion_hash
        or pins[str(args.world)] != args.expected_world_hash
    ):
        raise ValueError("source motion or world hash mismatch")

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
    from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.timed_reference_feasibility.v1",
        source_head=head,
        input_file_hashes=pins,
        source_frequency_hz=120,
        source_interval_inclusive=[0, 480],
        source_crop_reason="first4seconds separate from rejected full-source discontinuities",
        entry_intervals=50,
        frames_per_arm=300,
        arms=["measured_pose_hold", "source_first4seconds"],
        course=[10.0, 0.0, 0.0],
        ball_is_task_target=False,
        planned_native_steps=6000,
        optimizer_steps=0,
        model_variant="low_latency",
        onnx_graph_encoder_layout=False,
        activation_ceiling="SIM_ONLY",
        licensed_source_requires_separate_publicity_review=True,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)
    try:
        expected = (
            "root_pos_x(m)",
            "root_pos_y(m)",
            "root_pos_z(m)",
            "root_rot_w",
            "root_rot_x",
            "root_rot_y",
            "root_rot_z",
        ) + tuple(f"dof_{n}(rad)" for n in G1_DDS_JOINT_NAMES)
        with args.motion_csv.open(newline="") as stream:
            if tuple(next(csv.reader(stream))) != expected:
                raise ValueError("exact named meter/wxyz/radian canonical29 source required")
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        defaults = G1SonicRunupController.default_angles.copy()
        entry = np.r_[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0, defaults]
        options = dict(
            source_frequency_hz=120,
            entry_intervals=50,
            source_evidence_hash=hash_json(declaration),
        )
        try:
            prepare_sonic_reference(poses, entry, **options)
        except ValueError as error:
            write_once(args.output / "full-source-rejected.json", {"error": str(error)})
        else:
            raise ValueError("expected rejected whole-source premise changed; review protocol")
        prepared = prepare_sonic_reference(poses[:481], entry, **options)
        hold = SonicPoseReference((tuple(float(x) for x in entry),) * 10, hash_json(declaration))
        write_once(
            args.output / "prepared-reference.json",
            dict(
                contract_hash=prepared.contract_hash,
                source_pose_hash=prepared.source_pose_hash,
                poses=prepared.reference.poses,
                root_xyz_is_tracking_authority=False,
                physical_body_teleported=False,
            ),
        )
        results = []
        for name, reference in zip(declaration["arms"], (hold, prepared.reference), strict=True):
            nav = G1SonicNavigation(
                args.model_root,
                "tracker",
                SonicNavigationConfig(
                    maximum_frames=300,
                    model_variant="low_latency",
                    pose_reference=reference,
                    inference_threads=1,
                ),
            )
            model = mujoco.MjModel.from_binary_path(str(args.world))
            body = G1TargetEpisode(
                model,
                world_hash=args.expected_world_hash,
                kp=nav.backend.kp.copy(),
                kd=nav.backend.kd.copy(),
                course=(10.0, 0.0, 0.0),
                defaults=defaults,
            )
            if not np.array_equal(body.canonical()[0][:36], entry):
                raise ValueError("measured native entry mismatch")
            target_reference = reference.initialize(entry, frames=410)
            traces = []
            actions = []
            for frame in range(300):
                q, v = body.canonical()
                observation = TeamMotorObservation(
                    agent_id="tracker",
                    frame=frame,
                    time_sec=float(body.data.time),
                    intent="other",
                    prospective_owner=False,
                    qpos=tuple(float(x) for x in q),
                    qvel=tuple(float(x) for x in v),
                    target_position_m=(0.0, 0.0, 0.793),
                    navigation_command=(0.0, 0.0, 0.0),
                )
                proposal = nav.propose(observation)
                if not (
                    np.array_equal(proposal.kp, body.kp) and np.array_equal(proposal.kd, body.kd)
                ):
                    raise ValueError("PD gains changed")
                traces.append(body.step(frame, np.asarray(proposal.target_rad, dtype=np.float64)))
                actions.append(nav.backend.action.copy())
                if frame % 50 == 49:
                    print(name, frame + 1, "native", body.integrated_steps, flush=True)
            arrays = {key: np.concatenate([t[key] for t in traces]) for key in traces[0]}
            arrays["neural_action_isaac"] = np.asarray(actions)
            arrays["planned_reference"] = target_reference[:300]
            np.savez_compressed(args.output / f"{name}.npz", **arrays)
            actual = arrays["canonical_qpos"]
            results.append(
                dict(
                    arm=name,
                    body=body.review(),
                    artifact_hash=digest(args.output / f"{name}.npz"),
                    navigation_contract=nav.contract_hash,
                    foundation=nav.backend.qualification.qualification_hash,
                    planner_calls=nav.backend.planner_calls,
                    joint_reference_rmse_rad=float(
                        np.sqrt(np.mean((actual[:, 7:36] - target_reference[:300, 7:]) ** 2))
                    ),
                    minimum_native_pelvis_z_m=float(arrays["pelvis_z_per_substep_m"].min()),
                    final_root_position_m=body.canonical()[0][:3].tolist(),
                    source_reference_not_a_goal_label=True,
                )
            )
            write_once(args.output / f"{name}-complete.json", results[-1])
        if any(digest(Path(p)) != h for p, h in pins.items()):
            raise ValueError("pinned source/assets changed during physics")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                actual_native_steps=6000,
                neural_policy_recomputed_on_own_state=True,
                optimizer_steps=0,
                promotion_authorized=False,
                hardware_authorized=False,
                result="TRACKING_FEASIBILITY_DIAGNOSTIC_NOT_FOOTBALL_EXAM",
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json", dict(error=repr(error), promotion_authorized=False)
        )
        raise


if __name__ == "__main__":
    main()
