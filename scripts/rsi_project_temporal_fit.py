"""Project a failed reference fit and independently recheck sampled FK."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.reference_rate_projection import project_rate
from rosclaw_soccer.training.omnicontact_reference import LAB_TO_MUJOCO


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "reference", "fit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("scene-sha256", "reference-sha256", "fit-sha256"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    paths = ((a.scene, a.scene_sha256), (a.reference, a.reference_sha256), (a.fit, a.fit_sha256))
    if any(path.is_symlink() or digest(path) != expected for path, expected in paths):
        raise ValueError("exact input hashes required")
    fit = json.loads(a.fit.read_text())
    if (fit["scene_sha256"], fit["reference_sha256"], fit["status"]) != (
        a.scene_sha256,
        a.reference_sha256,
        "NOT_QUALIFIED",
    ) or fit["foot"] not in ("left", "right"):
        raise ValueError("unqualified same-input fit required")
    frames = np.array(fit["frames"], dtype=int)
    offsets = np.array(fit["active_leg_offsets_rad"], dtype=np.float64)
    if len(frames) != len(offsets) or len(frames) % 2 != 1 or np.any(np.diff(frames) != 1):
        raise ValueError("complete contiguous aligned reference required")
    projected, scale = project_rate(offsets, max_step=0.08)
    m = mujoco.MjModel.from_xml_path(str(a.scene))
    lo = NativeVelocityLayout(m, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2)
    d = mujoco.MjData(m)
    active = np.arange(0, 6) if fit["foot"] == "left" else np.arange(6, 12)
    rows = []
    with np.load(a.reference, allow_pickle=False) as ref:
        if frames[0] < 0 or frames[-1] >= len(ref["joint_pos"]):
            raise ValueError("valid source window required")
        for index, frame in enumerate(frames):
            d.qpos[lo.rq : lo.rq + 3] = ref["body_pos_w"][frame, 0]
            d.qpos[lo.rq + 3 : lo.rq + 7] = ref["body_quat_w"][frame, 0]
            d.qpos[lo.bq : lo.bq + 3] = ref["object_pos_w"][frame]
            d.qpos[lo.bq + 3 : lo.bq + 7] = ref["object_quat_w"][frame]
            d.qpos[lo.qi] = ref["joint_pos"][frame, list(LAB_TO_MUJOCO)]
            d.qpos[lo.qi[active]] += projected[index]
            mujoco.mj_forward(m, d)
            surface = measure_ball_surfaces(m, d, lo)["surfaces"]
            limits = m.jnt_range[np.array(lo.joints)[active]]
            joints = d.qpos[lo.qi[active]]
            rows.append(
                dict(
                    frame=int(frame),
                    foot_gap_m=float(surface[fit["foot"] + "_foot"]["gap_m"]),
                    left_shin_gap_m=float(surface["left_shin"]["gap_m"]),
                    right_shin_gap_m=float(surface["right_shin"]["gap_m"]),
                    limit_violation_rad=float(
                        max(0.0, np.max(limits[:, 0] - joints), np.max(joints - limits[:, 1]))
                    ),
                )
            )
    event = len(rows) // 2
    gates = dict(
        foot_gap=all(r["foot_gap_m"] >= -0.001 for r in rows),
        shin_gap=all(min(r["left_shin_gap_m"], r["right_shin_gap_m"]) >= -0.001 for r in rows),
        event_gap=-0.001 <= rows[event]["foot_gap_m"] <= 0.01,
        native_joint_limits=all(r["limit_violation_rad"] == 0 for r in rows),
        correction_step=float(np.max(np.abs(np.diff(projected, axis=0)))) <= 0.08,
    )
    if any(digest(path) != expected for path, expected in paths):
        raise ValueError("inputs changed during inspection")
    result = dict(
        status="SAMPLED_FK_GATES_PASS" if all(gates.values()) else "NOT_QUALIFIED",
        gates=gates,
        rows=rows,
        offsets_rad=projected.tolist(),
        scale=scale,
        source_optimizer_converged=bool(fit["gates"]["optimizer_converged"]),
        input_hashes=[expected for _, expected in paths],
        physics_steps=0,
        physically_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
        warning="sampled FK only, not trajectory feasibility, training or policy",
    )
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("rows", "offsets_rad")}, indent=2)
    )


if __name__ == "__main__":
    main()
