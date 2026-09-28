"""Compare batched GPU SONIC targets with individual ONNX on saved Isaac body states.

Read-only offline replay. No physics stepping, actuation, or policy promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.sim.isaac_root_bridge import isaac_root_to_mujoco
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation


def compare_saved_states(
    model_root: Path, probe_path: Path, *, frames: int, layout: str = "graph"
) -> dict:
    if not 2 <= frames <= 200:
        raise ValueError("bounded saved-body parity horizon required")
    if layout not in {"graph", "legacy"}:
        raise ValueError("explicit SONIC encoder layout required")
    with np.load(probe_path, allow_pickle=False) as archive:
        root = archive["root_before"].copy()
        root_velocity = archive["root_velocity_before"].copy()
        joint = archive["joint_before"].copy()
        joint_velocity = archive["joint_velocity_before"].copy()
    count = root.shape[1]
    if (
        root.ndim != 3
        or root.shape[0] < frames + 1
        or root.shape[2] != 7
        or root_velocity.shape != (root.shape[0], count, 6)
        or joint.shape != (root.shape[0], count, 29)
        or joint_velocity.shape != joint.shape
        or not 2 <= count <= 16
        or not all(np.isfinite(x).all() for x in (root, root_velocity, joint, joint_velocity))
    ):
        raise ValueError("invalid authenticated Isaac body state shape")
    config = SonicNavigationConfig(
        maximum_frames=max(50, frames + 1),
        model_variant="low_latency",
        experimental_maximum_speed_mps=1.5,
        inference_threads=1,
        onnx_graph_encoder_layout=layout == "graph",
    )
    navigations = [
        G1SonicNavigation(model_root, f"vector.first_touch.{i}", config) for i in range(count)
    ]

    def observations(frame: int) -> tuple[list[TeamMotorObservation], np.ndarray, np.ndarray]:
        rows = []
        qpos = []
        qvel = []
        for index, navigation in enumerate(navigations):
            qroot, vroot = isaac_root_to_mujoco(
                pose_xyzw=root[frame, index],
                velocity_world=root_velocity[frame, index],
                asset_quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
            )
            qroot[1] -= index * 8.0
            q = np.concatenate((qroot, joint[frame, index], (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)))
            v = np.concatenate((vroot, joint_velocity[frame, index], np.zeros(6)))
            qpos.append(q)
            qvel.append(v)
            rows.append(
                TeamMotorObservation(
                    agent_id=navigation.agent_id,
                    frame=frame,
                    time_sec=frame * 0.02,
                    intent="other",
                    prospective_owner=False,
                    qpos=tuple(float(x) for x in q),
                    qvel=tuple(float(x) for x in v),
                    target_position_m=(0.0, 0.0, 0.0),
                    navigation_command=(1.4, 0.0, 0.0),
                    navigation_envelope=navigation.navigation_envelope,
                )
            )
        return rows, np.asarray(qpos), np.asarray(qvel)

    first_obs, first_q, first_v = observations(0)
    for navigation, observation in zip(navigations, first_obs, strict=True):
        navigation.start_from_observation(observation)
    model = FrozenSonicG1Torch(model_root, variant="low_latency", device="cuda:0")
    tracker = BatchedSonicTracker(
        model,
        np.stack([navigation.backend.reference for navigation in navigations]),
        low_latency_legacy_encoder_layout=layout == "legacy",
    )
    tracker.reset(first_q, first_v)
    maxima = []
    for frame in range(frames):
        obs, q, v = observations(frame)
        reference = np.asarray(
            [
                navigation.propose(row).target_rad
                for navigation, row in zip(navigations, obs, strict=True)
            ]
        )
        if frame and frame % config.replan_frames == 0:
            tracker.refresh_unexecuted_reference(
                frame,
                np.stack([navigation.backend.reference for navigation in navigations]),
                unchanged_lookahead_frames=config.lookahead_frames,
            )
        actual = tracker.update(frame, q, v).detach().cpu().numpy()
        maxima.append(float(np.max(np.abs(reference - actual))))
        if frame + 1 < frames:
            _, q_next, v_next = observations(frame + 1)
            tracker.observe(q_next, v_next)
    maximum = max(maxima)
    report = {
        "schema": "rsi_sonic_torch_saved_body_replay_parity_v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "physical_episode_count": 0,
        "probe_hash": hash_bytes(probe_path.read_bytes()),
        "foundation_qualification_hash": model.qualification.qualification_hash,
        "frame_count": frames,
        "player_count": count,
        "encoder_layout": layout,
        "per_frame_max_target_difference_rad": maxima,
        "maximum_target_difference_rad": maximum,
        "parity_passed": maximum <= 1e-3,
    }
    report["report_hash"] = hash_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=19)
    parser.add_argument("--layout", choices=("graph", "legacy"), default="graph")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare_saved_states(
        args.model_root, args.probe, frames=args.frames, layout=args.layout
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    if not report["parity_passed"]:
        raise SystemExit("saved-body Torch/ONNX target parity failed")


if __name__ == "__main__":
    main()
