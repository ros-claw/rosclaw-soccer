"""SIM_ONLY search of phase-synced left-leg receiving contact in a faithful CPU proxy.

This is a development-only geometry probe. It never promotes a policy or scores
rendered pixels; the reserved Fresh8 incoming-ball suite remains unopened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from rsi_mjx_contact_replay_smoke import _contact_masks

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SNAPSHOT = 20
END_FRAME = 70
LEFT_JOINTS = (1, 3, 4, 5, 6)  # hip roll, hip pitch, knee, ankle pitch, ankle roll
LIMIT = 0.20


def _window(frame: int, onset: int, peak: int, end: int) -> float:
    if frame < onset or frame > end:
        return 0.0
    if frame <= peak:
        return (frame - onset) / max(1, peak - onset)
    return (end - frame) / max(1, end - peak)


def _evaluate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: NDArray[np.float64],
    qvel: NDArray[np.float64],
    target: NDArray[np.float64],
    kp: NDArray[np.float64],
    kd: NDArray[np.float64],
    params: NDArray[np.float64],
    timing: tuple[int, int, int],
    masks: tuple[int, NDArray[np.bool_], NDArray[np.bool_]],
) -> dict[str, Any]:
    data.qpos[:] = qpos[SNAPSHOT]
    data.qvel[:] = qvel[SNAPSHOT]
    mujoco.mj_forward(model, data)
    ball_geom, robot_mask, foot_mask = masks
    torque_limit = np.asarray(G1_HARD_TORQUE_LIMITS)
    joints = np.asarray(LEFT_JOINTS)
    limits = model.jnt_range[model.actuator_trnid[joints, 0]]
    contacts: list[tuple[int, int, str]] = []
    minimum = float(data.qpos[2])
    maximum_tilt = 0.0
    for frame in range(SNAPSHOT + 1, END_FRAME + 1):
        proposal = target[frame].copy()
        proposal[joints] = np.clip(
            proposal[joints] + _window(frame, *timing) * params,
            limits[:, 0],
            limits[:, 1],
        )
        for substep in range(10):
            data.ctrl[:] = np.clip(
                kp[frame] * (proposal - data.qpos[7:36]) - kd[frame] * data.qvel[6:35],
                -torque_limit,
                torque_limit,
            )
            mujoco.mj_step(model, data)
            minimum = min(minimum, float(data.qpos[2]))
            quat = data.qpos[3:7]
            maximum_tilt = max(
                maximum_tilt,
                float(np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1))),
            )
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                a, b = int(contact.geom1), int(contact.geom2)
                if a != ball_geom and b != ball_geom:
                    continue
                other = b if a == ball_geom else a
                if robot_mask[other]:
                    kind = "foot" if foot_mask[other] else "nonfoot"
                    contacts.append((frame, substep, kind))
    foot = any(kind == "foot" for _, _, kind in contacts)
    nonfoot = any(kind == "nonfoot" for _, _, kind in contacts)
    return {
        "clean_foot": foot and not nonfoot,
        "foot_seen": foot,
        "nonfoot_seen": nonfoot,
        "first_contact": list(contacts[0]) if contacts else None,
        "first_nonfoot": next((list(row) for row in contacts if row[2] == "nonfoot"), None),
        "minimum_pelvis_height_m": minimum,
        "maximum_tilt_rad": maximum_tilt,
        "ball_x_m": float(data.qpos[36]),
        "ball_vx_mps": float(data.qvel[35]),
        "safe": minimum >= 0.65 and maximum_tilt < 0.30,
    }


def probe(*, asset_root: Path, captured: Path, output_dir: Path, trials: int) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if (
        not 1 <= trials <= 500
        or output_dir.exists()
        or output_dir.resolve().is_relative_to(source.resolve().parents[1])
    ):
        raise ValueError("bounded new external SIM_ONLY output required")
    trace_path = captured / "motor-trace.npz"
    capture: dict[str, Any] = json.loads((captured / "report.json").read_text(encoding="utf-8"))
    committed = capture.pop("report_hash")
    if committed != hash_json(capture) or capture["trace_hash"] != hash_bytes(
        trace_path.read_bytes()
    ):
        raise ValueError("sealed receiving capture required")
    with np.load(trace_path, allow_pickle=False) as trace:
        arrays = [
            np.asarray(trace[key], dtype=np.float64)
            for key in (
                "sonic_recorded_qpos",
                "sonic_recorded_qvel",
                "sonic_recorded_target",
                "sonic_recorded_kp",
                "sonic_recorded_kd",
            )
        ]
    qpos, qvel, target, kp, kd = arrays
    if [array.shape for array in arrays] != [(300, 43), (300, 41), (300, 29), (300, 29), (300, 29)]:
        raise ValueError("complete measured motor trace required")
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    masks = _contact_masks(model)
    zero = np.zeros(5, dtype=np.float64)
    baseline = _evaluate(model, data, qpos, qvel, target, kp, kd, zero, (22, 33, 43), masks)
    rng = np.random.default_rng(20260928)
    rows = []
    timing_choices = ((21, 31, 43), (23, 34, 46), (25, 32, 40), (21, 35, 50))
    for index in range(trials):
        timing = timing_choices[index % len(timing_choices)]
        amplitude = rng.uniform(-LIMIT, LIMIT, size=5)
        result = _evaluate(model, data, qpos, qvel, target, kp, kd, amplitude, timing, masks)
        rows.append(
            {"trial": index, "timing": list(timing), "residual_rad": amplitude.tolist(), **result}
        )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("probe source changed during physics")
    rows.sort(
        key=lambda row: (
            row["safe"] and row["clean_foot"],
            row["safe"] and row["foot_seen"],
            not row["nonfoot_seen"],
            row["minimum_pelvis_height_m"],
        ),
        reverse=True,
    )
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_phase_contact_probe.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "capture_hash": committed,
        "compiled_model_hash": compiled_model_hash(model),
        "seed": 20260928,
        "trials": trials,
        "baseline": baseline,
        "safe_clean_foot_count": sum(bool(row["safe"] and row["clean_foot"]) for row in rows),
        "safe_foot_seen_count": sum(bool(row["safe"] and row["foot_seen"]) for row in rows),
        "best": rows[0],
        "fresh8_opened": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    (output_dir / "trials.json").write_text(
        json.dumps(rows, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trials", required=True, type=int)
    print(json.dumps(probe(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
