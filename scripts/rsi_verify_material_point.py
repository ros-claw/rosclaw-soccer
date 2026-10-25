"""Verify material-point derivatives on saved actual G1 states, without integration."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("qualification", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--qualification-hash", required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(args.expected_head)
    if args.output.exists() or digest(args.qualification) != args.qualification_hash:
        raise ValueError("unused output and exact prior forecast qualification required")
    qualification = json.loads(args.qualification.read_text())
    if qualification["status"] != (
        "G1_SHARED_PD_FORECAST_ALL_NEURAL_SUBSTEPS_EXACT_AND_ORIGINAL_ROLLOUT_UNCHANGED"
    ):
        raise ValueError("actual G1 forecast qualification required")
    pins = dict(qualification["input_file_hashes"])
    pins[str(args.qualification)] = args.qualification_hash
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique pinned physical input required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.measured_material_point import measure_material_point
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    reference = one("/course-09/onnx.json")
    record = json.loads(reference.read_text())
    artifact = reference.parent / record["artifact"]["path"]
    if pins.get(str(artifact)) != record["artifact"]["hash"]:
        raise ValueError("qualified actual native trajectory required")
    with np.load(artifact, allow_pickle=False) as raw:
        qpos, qvel = raw["qpos"], raw["qvel"]
    witnesses = []
    # Fixed diagnostic samples, not a new independent course or skill examination.
    for native in (0, 500, 1000, 1500, 2000, 2500, 3000, 3599):
        donor = mujoco.MjData(model)
        donor.qpos[:] = qpos[native]
        donor.qvel[:] = qvel[native]
        donor.time = (native + 1) * 0.002
        for name in (
            "a/left_knee_link",
            "a/right_knee_link",
            "a/left_ankle_roll_link",
            "a/right_ankle_roll_link",
        ):
            local = np.array([0.03, -0.02, -0.04], np.float64)
            measured = measure_material_point(
                model, donor, layout, body_name=name, local_point=local
            )
            numerical = np.empty((3, 29))
            for j, qi in enumerate(layout.qi):
                points = []
                for sign in (-1, 1):
                    scratch = mujoco.MjData(model)
                    scratch.qpos[:] = donor.qpos
                    scratch.qpos[qi] += sign * 1e-6
                    mujoco.mj_kinematics(model, scratch)
                    body = int(model.body(name).id)
                    points.append(scratch.xpos[body] + scratch.xmat[body].reshape(3, 3) @ local)
                numerical[:, j] = (points[1] - points[0]) / 2e-6
            error = float(np.max(np.abs(numerical - measured["joint_jacobian_hg"])))
            if error > 1e-8 or not np.array_equal(donor.qpos, qpos[native]):
                raise ValueError("actual G1 derivative or donor immutability failed")
            witnesses.append(dict(native_index=native, body=name, max_absolute_error=error))
    check_pins(pins)
    clean(args.expected_head)
    args.output.mkdir(parents=True)
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_G1_MATERIAL_POINT_HG29_DERIVATIVES_VERIFIED",
            input_file_hashes=pins,
            source_head=args.expected_head,
            witnesses=witnesses,
            new_physical_steps=0,
            new_optimizer_updates=0,
            task_space_teacher_effectiveness_proven=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
