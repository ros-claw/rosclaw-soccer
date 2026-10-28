"""Inspect a smooth window around an unqualified pose without changing motion."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.reference_window_diagnostics import cosine_envelope
from rosclaw_soccer.training.omnicontact_reference import LAB_TO_MUJOCO


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("scene", "reference", "fit", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("scene-sha256", "reference-sha256", "fit-sha256"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--center", type=int, required=True)
    p.add_argument("--half-width", type=int, required=True)
    a = p.parse_args()
    paths = ((a.scene, a.scene_sha256), (a.reference, a.reference_sha256), (a.fit, a.fit_sha256))
    if any(path.is_symlink() or digest(path) != expected for path, expected in paths):
        raise ValueError("exact original inputs required")
    fit = json.loads(a.fit.read_text())
    if fit["reference_sha256"] != a.reference_sha256 or fit["scene_sha256"] != a.scene_sha256:
        raise ValueError("fit must bind the same scene and reference")
    delta = np.asarray(fit["delta_leg_rad"], dtype=np.float64)
    if delta.shape != (12,) or not np.isfinite(delta).all() or np.max(np.abs(delta)) > 0.35000001:
        raise ValueError("bounded 12-leg diagnostic offset required")
    m = mujoco.MjModel.from_xml_path(str(a.scene))
    lo = NativeVelocityLayout(m, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2)
    d = mujoco.MjData(m)
    frames = np.arange(a.center - a.half_width, a.center + a.half_width + 1)
    weights = cosine_envelope(frames, center=a.center, half_width=a.half_width)
    rows = []
    with np.load(a.reference, allow_pickle=False) as ref:
        if frames[0] < 0 or frames[-1] >= len(ref["joint_pos"]):
            raise ValueError("complete bounded reference window required")
        for frame, weight in zip(frames, weights, strict=True):
            d.qpos[lo.rq : lo.rq + 3] = ref["body_pos_w"][frame, 0]
            d.qpos[lo.rq + 3 : lo.rq + 7] = ref["body_quat_w"][frame, 0]
            d.qpos[lo.bq : lo.bq + 3] = ref["object_pos_w"][frame]
            d.qpos[lo.bq + 3 : lo.bq + 7] = ref["object_quat_w"][frame]
            source_joints = ref["joint_pos"][frame, list(LAB_TO_MUJOCO)]
            samples = {}
            for label, amount in (("source", 0.0), ("adjusted", float(weight))):
                d.qpos[lo.qi] = source_joints
                d.qpos[lo.qi[:12]] += amount * delta
                mujoco.mj_forward(m, d)
                surfaces = measure_ball_surfaces(m, d, lo)["surfaces"]
                if not all(
                    surfaces[key]["available"]
                    for key in (fit["foot"] + "_foot", "left_shin", "right_shin")
                ):
                    raise ValueError("actual foot and shin surfaces required")
                limits = m.jnt_range[lo.joints[:12]]
                leg = d.qpos[lo.qi[:12]]
                samples[label] = dict(
                    foot_gap_m=surfaces[fit["foot"] + "_foot"]["gap_m"],
                    shin_gap_m=min(surfaces["left_shin"]["gap_m"], surfaces["right_shin"]["gap_m"]),
                    leg_limit_violation_rad=float(
                        max(0.0, np.max(limits[:, 0] - leg), np.max(leg - limits[:, 1]))
                    ),
                )
            rows.append(dict(frame=int(frame), weight=float(weight), **samples))
    if any(digest(path) != expected for path, expected in paths):
        raise ValueError("input changed during inspection")
    result = dict(
        status="REFERENCE_WINDOW_FK_DIAGNOSTIC_ONLY",
        scene_sha256=a.scene_sha256,
        reference_sha256=a.reference_sha256,
        fit_sha256=a.fit_sha256,
        center=a.center,
        half_width=a.half_width,
        rows=rows,
        physics_steps=0,
        physically_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
        warning="FK only; no dynamics, policy, dataset labels or reference artifact modified",
    )
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
