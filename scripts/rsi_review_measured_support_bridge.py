"""Independently read back anchored bridge geometry, without rerunning IK."""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.source / "complete.json"
    if (
        a.output.exists()
        or complete.is_symlink()
        or digest(complete) != a.source_hash
        or any(n in sys.modules for n in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused output, exact source and isolated import required")
    source = json.loads(complete.read_text())
    if (
        source["status"] != "MEASURED_SUPPORT_BRIDGE_STATIC_STUDY_NOT_PHYSICS"
        or [row["entry"] for row in source["records"]] != [57, 82]
        or source["native_steps"] != 0
        or source["neural_calls"] != 0
    ):
        raise ValueError("two closed declared static cases required")
    pins = dict(source["input_file_hashes"])
    pins[str(complete)] = a.source_hash
    for case in source["records"]:
        entry = case["entry"]
        for name, expected in [(case["artifact"]["path"], case["artifact"]["hash"])]:
            path = a.source / name
            if path.is_symlink() or path.parent != a.source or digest(path) != expected:
                raise ValueError("exact confined aggregate reference required")
            pins[str(path)] = expected
        for frame in range(21):
            report = a.source / f"entry{entry}-frame{frame:02d}.json"
            if report.is_symlink() or not report.is_file():
                raise ValueError("every recorded frame report required")
            row = json.loads(report.read_text())
            frame_path = a.source / row["artifact"]["path"]
            if (
                frame_path.is_symlink()
                or frame_path.parent != a.source
                or digest(frame_path) != row["artifact"]["hash"]
            ):
                raise ValueError("exact confined per-frame candidate required")
            pins[str(report)], pins[str(frame_path)] = digest(report), row["artifact"]["hash"]
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    traces = [Path(x) for x in pins if x.endswith("/rsi-warm-left-settle-v3-20261027/live.npz")]
    external_files = [Path(x) for x in pins if x.endswith("/policy/omnicontact/OmniContact.py")]
    if not (len(scenes) == len(traces) == len(external_files) == 1):
        raise ValueError("unique body, measured trace and source reference required")
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.measured_joint_bridge import sampled_hermite_joint_bridge
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact source body required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    sys.path.insert(0, str(external_files[0].parents[2]))
    from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ

    with np.load(traces[0], allow_pickle=False) as f:
        qpos, qvel = f["qpos"].copy(), f["qvel"].copy()
    floor = model.geom("floor").id
    feet = [
        [model.geom(f"a/{side}_foot{i}_collision").id for i in range(1, 8)]
        for side in ("left", "right")
    ]
    foot_set = {g for group in feet for g in group}
    ankles = [model.body(f"a/{side}_ankle_roll_link").id for side in ("left", "right")]
    data = mujoco.MjData(model)

    def measure(root, joints):
        data.qpos[lo.bq : lo.bq + 7] = [100, 100, 0.11, 1, 0, 0, 0]
        data.qpos[lo.rq : lo.rq + 7], data.qpos[lo.qi] = root, joints
        mujoco.mj_forward(model, data)
        gaps = np.array(
            [
                min(float(mujoco.mj_geomDistance(model, data, floor, g, 1.0, None)) for g in foot)
                for foot in feet
            ]
        )
        nonfoot = [
            float(c.dist)
            for c in data.contact
            if floor in (int(c.geom[0]), int(c.geom[1]))
            and not ({int(c.geom[0]), int(c.geom[1])} & foot_set)
        ]
        return gaps, data.xpos[ankles, :2].copy(), min(nonfoot, default=0.0)

    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            frames=42,
            independent_mujoco_geometry_and_anchor=True,
            bridge_polynomial_and_body_assets_shared=True,
            optimizer_not_rerun=True,
            optimizer_success_not_recorded_in_source=True,
            neural_calls=0,
            native_steps=0,
            hardware_authorized=False,
        ),
    )
    summaries = []
    for case in source["records"]:
        entry = case["entry"]
        initial_q, initial_v = qpos[entry * 10 - 1].copy(), qvel[entry * 10 - 1].copy()
        root0, q0, v0 = (
            initial_q[lo.rq : lo.rq + 7].copy(),
            initial_q[lo.qi].copy(),
            initial_v[lo.vi].copy(),
        )
        initial_gaps, anchor, _ = measure(root0, q0)
        support = (initial_gaps >= -0.002) & (initial_gaps <= 0.003)
        if support.tolist() != case["provisional_support"] or not np.allclose(
            initial_gaps, case["source_initial_foot_gaps_m"], rtol=0, atol=1e-12
        ):
            raise ValueError("initial measured support proxy mismatch")
        bridge = sampled_hermite_joint_bridge(
            q0,
            v0,
            DEFAULT_JOINT_POS_MJ.astype(np.float64),
            model.jnt_range[lo.joints].astype(np.float64),
            controls=20,
            control_interval_s=0.02,
            maximum_sampled_speed_rad_s=6.0,
        )
        with np.load(a.source / case["artifact"]["path"], allow_pickle=False) as f:
            aggregate = {k: f[k].copy() for k in f.files}
        if (
            not np.array_equal(aggregate["raw_joint"], bridge["position"])
            or not np.array_equal(aggregate["support"], support)
            or not np.allclose(aggregate["ankle_anchor_xy"], anchor, rtol=0, atol=1e-12)
        ):
            raise ValueError("source initial state, raw bridge or anchor mismatch")
        valid, xy_rows, uncertain = [], [], []
        for i in range(21):
            row = json.loads((a.source / f"entry{entry}-frame{i:02d}.json").read_text())
            with np.load(a.source / row["artifact"]["path"], allow_pickle=False) as f:
                saved = {k: f[k].copy() for k in f.files}
            raw_root = root0.copy()
            t = i * 0.02
            raw_root[:3] += initial_v[lo.rv : lo.rv + 3] * (t - t * t / 0.8)
            if (
                not np.array_equal(saved["raw_root"], raw_root)
                or not np.array_equal(saved["raw_joint"], bridge["position"][i])
                or not np.array_equal(saved["candidate_root"], aggregate["candidate_root"][i])
                or not np.array_equal(saved["candidate_joint"], aggregate["candidate_joint"][i])
            ):
                raise ValueError(f"raw/candidate provenance mismatch at {entry}/{i}")
            root, joint = saved["candidate_root"], saved["candidate_joint"]
            gaps, xy, nonfoot = measure(root, joint)
            anchor_error = float(np.max(np.linalg.norm((xy - anchor)[support], axis=1)))
            if not np.allclose(gaps, row["foot_gaps_m"], rtol=0, atol=1e-12) or not np.allclose(
                xy, aggregate["candidate_ankle_xy"][i], rtol=0, atol=1e-12
            ):
                raise ValueError(f"independent candidate geometry mismatch at {entry}/{i}")
            leg_delta = joint[:12] - saved["raw_joint"][:12]
            geometry_ok = bool(
                np.min(gaps) >= -0.002
                and np.max(np.abs(gaps[support])) <= 0.01
                and anchor_error <= 0.02
                and nonfoot >= -0.002
                and abs(root[2] - raw_root[2]) <= 0.02
                and root[2] >= 0.65
                and np.array_equal(root[[0, 1, 3, 4, 5, 6]], raw_root[[0, 1, 3, 4, 5, 6]])
                and np.array_equal(joint[12:], saved["raw_joint"][12:])
                and np.all(joint >= model.jnt_range[lo.joints, 0])
                and np.all(joint <= model.jnt_range[lo.joints, 1])
                and not np.any(np.abs(leg_delta) >= 0.2 - 1e-6)
            )
            if row["valid"] and not geometry_ok:
                raise ValueError(f"source accepted geometrically invalid frame {entry}/{i}")
            if not row["valid"] and geometry_ok:
                uncertain.append(i)
            valid.append(bool(row["valid"]))
            xy_rows.append(xy)
        roots, joints, xy = (
            aggregate["candidate_root"],
            aggregate["candidate_joint"],
            np.stack(xy_rows),
        )
        measured = dict(
            valid_frames=sum(valid),
            max_root_step_m=float(np.max(np.abs(np.diff(roots[:, 2])))),
            max_joint_step_rad=float(np.max(np.abs(np.diff(joints, axis=0)))),
            max_support_ankle_step_m=float(
                np.max(np.linalg.norm(np.diff(xy[:, support], axis=0), axis=2))
            ),
            initial_velocity_mismatch_rad_s=float(
                np.max(np.abs((joints[1] - joints[0]) / 0.02 - v0))
            ),
            terminal_joint_error_rad=float(np.max(np.abs(joints[-1] - DEFAULT_JOINT_POS_MJ))),
        )
        for key, value in measured.items():
            if not np.isclose(value, case[key], rtol=0, atol=1e-12):
                raise ValueError(f"source continuity metric differs: {entry}/{key}")
        passed = bool(
            all(valid)
            and measured["max_root_step_m"] <= 0.006
            and measured["max_joint_step_rad"] <= 0.12
            and measured["max_support_ankle_step_m"] <= 0.006
            and measured["initial_velocity_mismatch_rad_s"] <= 2.0
            and measured["terminal_joint_error_rad"] <= 0.02
        )
        if passed != case["all_frames_pass"]:
            raise ValueError("source full bridge decision mismatch")
        summaries.append(
            dict(
                entry=entry,
                metrics=measured,
                all_frames_pass=passed,
                invalid_with_geometry_pass_and_unrecorded_solver_status=uncertain,
            )
        )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="MEASURED_SUPPORT_BRIDGE_READBACK_NOT_PHYSICS",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            records=summaries,
            independent_geometry=True,
            shared_bridge_math=True,
            optimizer_not_rerun=True,
            solver_success_not_recorded_in_source=True,
            neural_calls=0,
            native_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
