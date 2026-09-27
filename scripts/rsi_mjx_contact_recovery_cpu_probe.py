"""SIM_ONLY long-horizon CPU probe for the frozen MJX-trained contact actor.

This is a diagnostic, not a promotion exam: the probe reuses exposed courses
and asks whether contact-triggered residual withdrawal preserves balance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rsi_mjx_contact_replay_smoke import _contact_masks
from rsi_mjx_contact_residual_es import ACTION_LIMIT_RAD, FEATURES, JOINT_INDICES

from rosclaw_soccer.providers.g1.sonic_runup import _sonic_control_parameters
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model


def probe(
    *,
    asset_root: Path,
    parent_trace: Path,
    candidate: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY evidence directory required")
    payload: dict[str, Any] = json.loads(candidate.read_text(encoding="utf-8"))
    commitment = payload.pop("result_hash")
    if commitment != hash_json(payload) or payload.get("promotion_authorized") is not False:
        raise ValueError("sealed unpromoted contact actor required")
    parameters = np.asarray(payload["parameters"], dtype=np.float64)
    if parameters.shape != (64,) or not np.isfinite(parameters).all():
        raise ValueError("finite 64-parameter contact actor required")
    with np.load(parent_trace, allow_pickle=False) as trace:
        qpos = np.asarray(trace["qpos"], dtype=np.float64)
        qvel = np.asarray(trace["qvel"], dtype=np.float64)
        targets = np.asarray(trace["target"], dtype=np.float64)
    if qpos.shape != (300, 43) or qvel.shape != (300, 41) or targets.shape != (300, 29):
        raise ValueError("qualified 300-frame parent trajectory required")
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    )
    model.opt.timestep = 0.002
    ball_id, robot_mask, foot_mask = _contact_masks(model)
    kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
    torque_limit = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    index = np.asarray(JOINT_INDICES, dtype=np.int64)
    limits = model.jnt_range[model.actuator_trnid[index, 0]]
    matrix = parameters[: 4 * FEATURES].reshape(4, FEATURES)
    bias = parameters[4 * FEATURES :]
    courses = ((2.20, 0.10, 0.0), (2.10, 0.08, -0.4), (2.26, 0.08, -0.4))
    rows = []
    for course in courses:
        for mode in ("parent", "persistent", "fade_on_foot_5", "fade_on_foot_15"):
            data = mujoco.MjData(model)
            data.qpos[:] = qpos[80]
            data.qvel[:] = qvel[80]
            data.qpos[36:39] = (course[0], course[1], 0.11)
            data.qvel[35] = course[2]
            data.qvel[39] = course[2] / 0.11
            mujoco.mj_forward(model, data)
            foot_seen = False
            nonfoot_seen = False
            first_foot_frame: int | None = None
            first_fall_frame: int | None = None
            minimum = float(data.qpos[2])
            for frame, base in enumerate(targets[81:300]):
                ball_dx = float(data.qpos[36] - data.qpos[0])
                features = np.concatenate(
                    (
                        np.asarray(
                            (
                                ball_dx,
                                data.qpos[37] - data.qpos[1],
                                data.qvel[35] - data.qvel[0],
                                data.qvel[0],
                                data.qpos[2] - 0.75,
                            )
                        ),
                        data.qpos[7 + index],
                        data.qvel[6 + index] / 5.0,
                        np.asarray((foot_seen, nonfoot_seen), dtype=np.float64),
                    )
                )
                if not np.isfinite(features).all():
                    raise ValueError("nonfinite long-horizon observation")
                gate = float((0.05 < ball_dx < 1.2) or foot_seen)
                if mode == "parent":
                    gate = 0.0
                elif mode.startswith("fade_on_foot") and first_foot_frame is not None:
                    fade_frames = 5 if mode.endswith("_5") else 15
                    gate *= max(0.0, 1.0 - (frame - first_foot_frame) / fade_frames)
                residual = ACTION_LIMIT_RAD * gate * np.tanh(matrix @ features + bias)
                proposal = base.copy()
                proposal[index] = np.clip(base[index] + residual, limits[:, 0], limits[:, 1])
                for _ in range(10):
                    data.ctrl[:] = np.clip(
                        kp * (proposal - data.qpos[7:36]) - kd * data.qvel[6:35],
                        -torque_limit,
                        torque_limit,
                    )
                    mujoco.mj_step(model, data)
                    minimum = min(minimum, float(data.qpos[2]))
                    if first_fall_frame is None and data.qpos[2] < 0.65:
                        first_fall_frame = frame
                    for contact_id in range(data.ncon):
                        contact = data.contact[contact_id]
                        a, b = int(contact.geom1), int(contact.geom2)
                        if a != ball_id and b != ball_id:
                            continue
                        other = b if a == ball_id else a
                        if robot_mask[other]:
                            if foot_mask[other]:
                                foot_seen = True
                                if first_foot_frame is None:
                                    first_foot_frame = frame
                            else:
                                nonfoot_seen = True
            if not math.isfinite(minimum) or not np.isfinite(data.qpos).all():
                raise ValueError("nonfinite long-horizon physics")
            rows.append(
                {
                    "course": list(course),
                    "mode": mode,
                    "foot_seen": foot_seen,
                    "nonfoot_seen": nonfoot_seen,
                    "first_foot_frame": first_foot_frame,
                    "first_fall_frame": first_fall_frame,
                    "minimum_pelvis_height_m": minimum,
                    "final_pelvis_height_m": float(data.qpos[2]),
                    "final_ball_position_m": data.qpos[36:39].tolist(),
                    "final_ball_velocity_m_s": data.qvel[35:38].tolist(),
                }
            )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("probe source changed")
    result = {
        "schema": "rosclaw_soccer.rsi.mjx_contact_recovery_cpu_probe.v1",
        "activation_ceiling": "SIM_ONLY",
        "candidate_hash": commitment,
        "parent_trace_hash": hash_bytes(parent_trace.read_bytes()),
        "source_hash": source_hash,
        "promotion_authorized": False,
        "rows": rows,
    }
    result["report_hash"] = hash_json(result)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--parent-trace", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = probe(**vars(args))
    for row in result["rows"]:
        print(json.dumps(row, sort_keys=True))
    print(result["report_hash"])


if __name__ == "__main__":
    main()
