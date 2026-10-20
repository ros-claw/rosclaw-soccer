"""Recorded-target native replay with whole-robot ball contact classification.

No NN calls or learning. Repeat0 only: four trajectories per source pilot,
maximum12000 native steps. A matching replay is not neural recomputation.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-unit", "source-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("frozen replay source and unused output required")
    wait_for_terminal_collection(a.source_unit)
    source_path = a.source_root / "complete.json"
    if digest(source_path) != a.source_hash:
        raise ValueError("actual source pilot hash required")
    report = json.loads(source_path.read_text())
    if (
        report.get("status") != "BALL_AWARE_NATIVE_CONTACT_NOT_GOAL_OR_TEAM_QUALIFIED"
        or report.get("all_full_independent_repeats_exact") is not True
    ):
        raise ValueError("completed independently repeated native pilot required")
    rows = [r for r in report["results"] if r["repeat"] == 0]
    if len(rows) != 4 or {(r["source"], r["arm"]) for r in rows} != {
        (s, arm) for s in (7, 10) for arm in ("initial", "final")
    }:
        raise ValueError("all four repeat0 source/arm trajectories required")
    pins = dict(report["input_file_hashes"])
    pins[str(source_path)] = a.source_hash
    models = [Path(k) for k in pins if k.endswith("/compiled_model.mjb")]
    if len(models) != 1:
        raise ValueError("one immutable original native body required")
    paths = []
    for row in rows:
        path = a.source_root / f"source{row['source']}-{row['arm']}-repeat0" / "native.npz"
        pins[str(path)] = row["raw_hash"]
        paths.append(path)
    for root in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            maximum_native_steps=12000,
            new_neural_calls=0,
            new_optimizer_steps=0,
            qv_atol=1e-9,
            force_atol_n=1e-7,
            recorded_targets_only=True,
            neural_policy_recomputed=False,
            hardware_authorized=False,
        ),
    )
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

    model = mujoco.MjModel.from_binary_path(str(models[0]))
    if mujoco.get_mjcb_control() is not None or model.opt.timestep != 0.002:
        raise ValueError("original2ms native dynamics without callbacks required")
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    feet = [model.body("robot/" + s + "_ankle_roll_link").id for s in ("left", "right")]
    robot = {b for b in range(model.nbody) if model.body(b).name.startswith("robot/")}
    attempts = steps = 0
    results = []
    try:
        for row, path in zip(rows, paths, strict=True):
            with np.load(path, allow_pickle=False) as z:
                raw = {
                    k: z[k].copy()
                    for k in (
                        "native_qpos_before",
                        "canonical_qvel_before",
                        "target_rad",
                        "canonical_qpos_after",
                        "canonical_qvel_after",
                        "actual_force",
                        "foot_ball_contact_force_n",
                    )
                }
            horizon = len(raw["target_rad"])
            if not 1 <= horizon <= 300 or any(not np.isfinite(v).all() for v in raw.values()):
                raise ValueError("bounded finite complete recorded controls required")
            data = mujoco.MjData(model)
            data.qpos[:] = raw["native_qpos_before"][0]
            v = raw["canonical_qvel_before"][0]
            data.qvel[layout.rv : layout.rv + 6] = v[:6]
            data.qvel[layout.vi] = v[6:35]
            data.qvel[layout.bv : layout.bv + 6] = v[35:]
            warning = tuple(int(w.number) for w in data.warning)
            categories, nonfoot_bodies = [], set()
            q_error = v_error = force_error = foot_error = 0.0
            for frame in range(horizon):
                mujoco.mj_forward(model, data)
                data.ctrl[layout.ai] = raw["target_rad"][frame]
                for substep in range(10):
                    if attempts >= 12000:
                        raise ValueError("declared replay budget exhausted")
                    attempts += 1
                    before = data.time
                    mujoco.mj_step(model, data)
                    steps += 1
                    if (
                        not np.isclose(data.time - before, 0.002, rtol=0, atol=1e-10)
                        or tuple(int(w.number) for w in data.warning) != warning
                    ):
                        raise ValueError("native warning or clock discontinuity")
                    q, v = layout.canonical(data)
                    q_error = max(
                        q_error, float(abs(q - raw["canonical_qpos_after"][frame, substep]).max())
                    )
                    v_error = max(
                        v_error, float(abs(v - raw["canonical_qvel_after"][frame, substep]).max())
                    )
                    force_error = max(
                        force_error,
                        float(
                            abs(
                                data.actuator_force[layout.ai] - raw["actual_force"][frame, substep]
                            ).max()
                        ),
                    )
                    category = np.zeros(3)
                    for cid in range(data.ncon):
                        contact = data.contact[cid]
                        b1, b2 = model.geom_bodyid[[contact.geom1, contact.geom2]]
                        other = b2 if b1 == layout.ball else b1 if b2 == layout.ball else -1
                        if other not in robot:
                            continue
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, cid, force)
                        magnitude = float(np.linalg.norm(force[:3]))
                        side = feet.index(other) if other in feet else 2
                        category[side] = max(category[side], magnitude)
                        if side == 2 and magnitude > 1:
                            nonfoot_bodies.add(model.body(other).name)
                    foot_error = max(
                        foot_error,
                        float(
                            abs(
                                category[:2] - raw["foot_ball_contact_force_n"][frame, substep]
                            ).max()
                        ),
                    )
                    categories.append(category)
            if max(q_error, v_error) > 1e-9 or max(force_error, foot_error) > 1e-7:
                raise ValueError("recorded-target replay does not reproduce source physical truth")
            folder = a.output / f"source{row['source']}-{row['arm']}"
            folder.mkdir()
            values = np.asarray(categories)
            np.savez_compressed(folder / "contacts.npz", force=values)
            results.append(
                dict(
                    source=row["source"],
                    arm=row["arm"],
                    world=row["world_index"],
                    q_error=q_error,
                    v_error=v_error,
                    actuator_force_error_n=force_error,
                    foot_force_error_n=foot_error,
                    nonfoot_bodies_over_1n=sorted(nonfoot_bodies),
                    maximum_contact_force_n=values.max(axis=0).tolist(),
                    nonfoot_over_1n=bool((values[:, 2] > 1).any()),
                    nonfoot_over_10n=bool((values[:, 2] > 10).any()),
                    raw_hash=digest(folder / "contacts.npz"),
                )
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="RECORDED_TARGET_CONTACT_REPLAY_NOT_NEURAL_OR_TASK_QUALIFICATION",
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=results,
                native_steps=steps,
                attempted_native_steps=attempts,
                new_neural_calls=0,
                new_optimizer_steps=0,
                neural_policy_recomputed=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                results=results,
                native_steps=steps,
                attempted_native_steps=attempts,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
