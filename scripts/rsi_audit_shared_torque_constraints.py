"""Audit actual native force traces without invoking a neural policy or stepping physics."""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("zero", "full", "training", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("zero-hash", "full-hash", "training-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused audit output required")
    pins, reports = {}, {}
    for name in ("zero", "full", "training"):
        root, expected = getattr(a, name), getattr(a, name + "_hash")
        path = root / "complete.json"
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact completed source hash required")
        report = json.loads(path.read_text())
        for key, value in report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting source provenance")
            pins[key] = value
        pins[str(path)] = expected
        reports[name] = report
    for name in ("zero", "full"):
        report = reports[name]
        if (
            report["status"] != "LIVE_NEURAL_TRANSITION_DIAGNOSED_NOT_PROMOTED"
            or len(report["records"]) != 4
        ):
            raise ValueError("four complete torque trial rows required")
        for row in report["records"]:
            for artifact in (row["artifact"], row["torque_execution_artifacts"]["a/"]):
                path = getattr(a, name) / artifact["path"]
                if (
                    path.is_symlink()
                    or path.parent.resolve() != getattr(a, name).resolve()
                    or digest(path) != artifact["hash"]
                ):
                    raise ValueError("confined immutable actual traces required")
                pins[str(path)] = artifact["hash"]
    source_pins(pins)
    check_pins(pins)
    spec = importlib.util.find_spec("rosclaw")
    if spec is None or spec.origin is None:
        raise ValueError("pinned Core constants required")
    constants = Path(spec.origin).parent / "simforge/tasks/g1_goalforge/concepts.py"
    if str(constants) not in pins or digest(constants) != pins[str(constants)]:
        raise ValueError("active Core constants must belong to recorded source provenance")
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.torque_constraint_audit import audit_applied_torques
    from rosclaw_soccer.sim.contracts import hash_json

    scenes = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one exact canonical scene required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    safety = reports["training"]["safety"]
    # Unit-gear raw motors have no ctrlrange limiter; use the pinned HG contract,
    # not MuJoCo's unused default ctrlrange (which is zero).
    hard = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    body_hash = hash_json(layout.profile())
    records = []
    for name in ("zero", "full"):
        for index, row in enumerate(reports[name]["records"]):
            fraction = 0.0 if name == "zero" else 1.0
            execution = row["torque_execution_artifacts"]["a/"]
            if (
                row["index"] != index
                or row["arm"] != "single"
                or row["torque_intervention_fraction"] != fraction
                or execution["body_hash"] != body_hash
            ):
                raise ValueError("ordered exact body and declared physical intervention required")
            with (
                np.load(getattr(a, name) / row["artifact"]["path"]) as raw,
                np.load(getattr(a, name) / execution["path"]) as episode,
            ):
                applied, parent = raw["torque"][:, layout.ai], raw["parent_torque"][:, layout.ai]
                if (
                    not np.array_equal(episode["actions"], applied.astype(np.float32))
                    or not np.array_equal(episode["parent_actions"], parent.astype(np.float32))
                    or not np.array_equal(
                        raw["applied_torque_change"][:, layout.ai], applied - parent
                    )
                ):
                    raise ValueError("recorded actions must equal the actual actuator force trace")
                before_velocity = np.vstack(
                    (episode["observations"][0, 29:58], raw["qvel"][:-1, layout.vi])
                )
                result = audit_applied_torques(
                    applied,
                    parent,
                    before_velocity,
                    hard,
                    guard_scale=safety["torque_guard_scale"],
                    parent_deviation_ratio=safety["maximum_parent_deviation_ratio"],
                    delta_ratio=safety["max_delta_ratio_per_step"],
                    maximum_power_w=safety["maximum_mechanical_power_w"],
                )
                if fraction == 0 and result["nonparent_steps"]:
                    raise ValueError("zero intervention must preserve actual parent torque")
                positions = raw["qpos"][:, layout.bq : layout.bq + 3]
                cross = np.flatnonzero((positions[:-1, 0] < 5) & (positions[1:, 0] >= 5))
                error = None
                if len(cross):
                    k = cross[0]
                    t = (5 - positions[k, 0]) / (positions[k + 1, 0] - positions[k, 0])
                    point = positions[k] + t * (positions[k + 1] - positions[k])
                    error = float(np.linalg.norm(point[1:] - [0, 0.11]))
                reported = row["actual_goal_plane"]
                if (error is None) != (reported is None) or (
                    error is not None and abs(error - reported["error_to_fixed_target_m"]) > 1e-10
                ):
                    raise ValueError("raw independently calculated goal error disagrees")
                records.append(
                    dict(
                        arm=name,
                        index=index,
                        constraints=result,
                        actual_goal_error_m=error,
                        healthy=row["healthy_full_horizon"],
                    )
                )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="ACTUAL_TORQUE_CONSTRAINTS_NUMERICALLY_AUDITED_NOT_PROMOTED",
            input_file_hashes=pins,
            source_head=a.expected_head,
            records=records,
            all_constraints_pass=all(
                all(v == 0 for k, v in r["constraints"].items() if k.endswith("violations"))
                for r in records
            ),
            independent_neural_replay=False,
            new_physics_steps=0,
            optimizer_updates=0,
            goal_complete=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
