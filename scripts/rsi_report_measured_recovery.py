"""Read-only native motion diagnostics for BOTH players, not a physics rerun.

Authenticate the closed source/own pair and every consumed primitive. Compile
the pinned scene only for layout lookup; never allocate MjData or step physics.
No reward, task label, source artifact or candidate weight is changed.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "own-root", "scene", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.measured_motion_quality import measured_motion_quality
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.parallel_refinement_batch import FULL_OWN, FULL_SOURCE
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused diagnostic output required")
    reports, pins = [], {}
    for root in (a.source_root, a.own_root):
        if root.is_symlink() or (root / "failed.json").exists():
            raise ValueError("ordinary closed native source/own pair required")
        path = root / "complete.json"
        pins[str(path)] = file_hash(path)
        reports.append(json.loads(path.read_text()))
    source, own = reports
    allowed = {
        "PPO_PAIRED_MEAN12_COLLECTED_PENDING_OWN_NATIVE": "PPO_PAIRED_MEAN12_ALL_OWN_NATIVE_MATCH",
        FULL_SOURCE: FULL_OWN,
    }
    if (
        allowed.get(source["status"]) != own["status"]
        or own["source_hash"] != pins[str(a.source_root / "complete.json")]
        or own["training_hash"] != source["training_hash"]
        or own["config"] != source["config"]
        or own["counts"] != source["counts"]
        or own["independent_whole_world_replay"] is not True
        or own["records"]
        != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
    ):
        raise ValueError("all exact source/own native results required for diagnostics")
    scene_hash = file_hash(a.scene)
    if source["input_file_hashes"].get(str(a.scene)) != scene_hash:
        raise ValueError("actual source-pinned scene layout required")
    pins[str(a.scene)] = scene_hash
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    layouts = {
        role: NativeVelocityLayout(
            model, robot_prefix=role + "/", ball_body_name="soccer_ball", player_count=2
        )
        for role in ("a", "b")
    }
    rows = []
    for record in source["records"]:
        loaded = {}
        for key in ("initial", "physical", "A"):
            art = record["artifacts"][key]
            path = a.source_root / art["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (a.source_root / f"trial{record['index']}").resolve()
                or file_hash(path) != art["hash"]
            ):
                raise ValueError("exact confined measured primitives required")
            pins[str(path)] = art["hash"]
            with np.load(path, allow_pickle=False) as z:
                loaded[key] = {k: z[k].copy() for k in z.files}
        initial, raw, neural = (loaded[k] for k in ("initial", "physical", "A"))
        controls = record["controls"]
        if (
            raw["qpos"].shape != (controls, 10, model.nq)
            or raw["qvel"].shape != (controls, 10, model.nv)
            or neural["standing"].shape != (controls,)
            or neural["standing"].dtype != np.bool_
            or np.count_nonzero(neural["standing"]) != record["stand_controls"]
            or not np.array_equal(
                neural["standing"], np.arange(controls) >= controls - record["stand_controls"]
            )
        ):
            raise ValueError("all actual native samples and contiguous recovery phase required")
        q, v = raw["qpos"].reshape(-1, model.nq), raw["qvel"].reshape(-1, model.nv)
        strike = record["outcome"]["incoming"]["A_first_foot_strike_substep"]
        recovery_start = (controls - record["stand_controls"]) * 10
        for role, layout in layouts.items():
            w, x, y, z = initial["qpos"][layout.rq + 3 : layout.rq + 7]
            if abs(np.linalg.norm([w, x, y, z]) - 1) > 1e-5:
                raise ValueError("measured initial unit body orientation required")
            heading = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])
            if np.linalg.norm(heading) < 1e-6:
                raise ValueError("initial body heading has no horizontal projection")
            heading /= np.linalg.norm(heading)
            for window, start in (
                ("after_A_first_foot_strike", strike),
                ("A_neural_recovery_phase", recovery_start),
            ):
                metrics = None
                if start is not None and 0 <= start < len(q):
                    metrics = measured_motion_quality(
                        q[start:, layout.rq : layout.rq + 3],
                        v[start:, layout.rv + 3 : layout.rv + 6],
                        v[start:, layout.vi],
                        raw["target_" + role][start // 10 :],
                        heading,
                    )
                rows.append(
                    dict(
                        index=record["index"],
                        arm=record.get("arm"),
                        role=role,
                        window=window,
                        first_native_sample=start,
                        metrics=metrics,
                    )
                )
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="ACTUAL_NATIVE_MOTION_DIAGNOSTICS_NOT_EXAM",
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_hash=pins[str(a.source_root / "complete.json")],
            rows=rows,
            physical_steps_executed=0,
            optimizer_updates=0,
            reward_changed=False,
            labels_changed=False,
            physical_naturalness_proven=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
