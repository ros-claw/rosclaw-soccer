"""Independent execution of recorded PD targets and entry-MPC decisions.

Reintegrates all actual H300 bodies and every candidate from a warm clone of
its independently replayed actual prefix. Does NOT rerun neural policies or
navigation-reference generation; those checks belong to the producer's
prospective-vs-actual comparison, not this recorded-target physics reader.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("producer", "world", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-producer-head",
        "expected-producer-hash",
        "producer-unit",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != args.expected_head
    ):
        raise ValueError("sealed reader source and unused output required")
    terminal = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                args.producer_unit,
                "-p",
                "MainPID",
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
                "-p",
                "SubState",
            ],
            text=True,
        ).splitlines()
    )
    if terminal != {
        "MainPID": "0",
        "Result": "success",
        "ExecMainStatus": "0",
        "SubState": "exited",
    }:
        raise ValueError("successfully closed durable producer required")
    if digest(args.producer / "complete.json") != args.expected_producer_hash:
        raise ValueError("producer completion identity differs")
    declaration = json.loads((args.producer / "declaration.json").read_text())
    complete = json.loads((args.producer / "complete.json").read_text())
    if (
        declaration["schema"] != "soccer.rsi.predictive_ball_entry4.v1"
        or declaration["source_head"] != args.expected_producer_head
        or complete["actual_native_steps"] != 24000
        or complete["optimizer_steps"] != 0
        or [r["job"] for r in complete["results"]]
        != [f"course{i}-{arm}" for i in range(4) for arm in ("nominal", "predictive")]
        or (args.producer / "failure.json").exists()
    ):
        raise ValueError("complete exact paired eight-body study required")
    for p, h in declaration["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("producer source/assets/evidence changed")
    if declaration["input_file_hashes"].get(str(args.world)) != digest(args.world):
        raise ValueError("reader world binding differs")
    paths = [
        p
        for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [p for p in args.producer.rglob("*") if p.is_file()]
    pins = {str(p): digest(p) for p in paths}
    for row in complete["results"]:
        if digest(args.producer / (row["job"] + ".npz")) != row["artifact_hash"]:
            raise ValueError("actual trajectory hash differs")
        for item in row["candidate_artifacts"]:
            for key in ("artifact", "receipt"):
                path = (args.producer / item[key]).resolve()
                if (
                    not path.is_relative_to(args.producer.resolve())
                    or digest(path) != item[key + "_hash"]
                ):
                    raise ValueError("candidate artifact binding differs")

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.physics.sonic_navigation_forecast import _clone_body
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision
    from rosclaw_soccer.rsi.full_task_branch import integration_hash
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.predictive_entry_recorded_review.v1",
            input_file_hashes=pins,
            producer_terminal=terminal,
            expected_actual_native_steps=24000,
            expected_forecast_native_steps=complete["forecast_native_steps"],
            neural_policy_replayed=False,
            navigation_planner_reexecuted=False,
            warm_clone_helper_shared_with_producer=True,
            all_candidate_pd_and_native_physics_reexecuted=True,
            optimizer_steps=0,
            hardware_authorized=False,
        ),
    )
    try:
        stored = json.loads((args.producer / "entry-affordance.json").read_text())
        fields = {
            k: stored[k]
            for k in (
                "ball_offset_xy_m",
                "historical_frame",
                "historical_forward_speed_mps",
                "historical_artifact_hash",
            )
        }
        fields["ball_offset_xy_m"] = tuple(fields["ball_offset_xy_m"])
        affordance = SkillEntryAffordance(**fields)
        if affordance.contract_hash != stored["contract_hash"]:
            raise ValueError("affordance contract differs")
        model = mujoco.MjModel.from_binary_path(str(args.world))
        kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
        robot_root = model.body_rootid[model.body("pelvis").id]
        feet = [
            model.body(side + suffix).id
            for side in ("left", "right")
            for suffix in ("_ankle_roll_link", "_ankle_pitch_link")
        ]
        nonfeet = np.flatnonzero(
            (model.body_rootid == robot_root) & ~np.isin(np.arange(model.nbody), feet)
        )
        results, forecast_native = [], 0
        for row in complete["results"]:
            name = row["job"]
            with np.load(args.producer / (name + ".npz"), allow_pickle=False) as archive:
                raw = {k: archive[k] for k in archive.files}
            body = G1TargetEpisode(
                model,
                world_hash=digest(args.world),
                kp=kp,
                kd=kd,
                course=tuple(row["course"]),
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )
            plans = {p["frame"]: p for p in row["entry_planner_receipts"]}
            held, stage, handoff = None, 0, None
            goal, error, candidate_count = np.zeros(2), 0.0, 0
            for frame in range(300):
                q, v = body.canonical()
                np.testing.assert_array_equal(q, raw["canonical_qpos"][frame])
                np.testing.assert_array_equal(v, raw["canonical_qvel"][frame])
                command = (0.0, 0.0, 0.0)
                if stage in (0, 2):
                    d = ball_entry_decision(q, v, frame=frame, affordance=affordance)
                    command, goal, error = (
                        d.command,
                        np.asarray(d.target_root_xy_m),
                        d.horizontal_error_m,
                    )
                    if d.timed_out:
                        stage = 2
                    if (
                        name.endswith("-predictive")
                        and frame in (20, 40, 60, 80)
                        and not d.ready
                        and frame not in plans
                    ):
                        raise ValueError("required planning decision missing")
                    if frame in plans:
                        plan = plans[frame]
                        if d.ready or d.timed_out or frame not in (20, 40, 60, 80):
                            raise ValueError("planning at undeclared or terminal entry state")
                        # Independently reconstruct the fixed command bank.
                        xy = np.asarray(d.command[:2])
                        alternatives = [
                            xy,
                            0.5 * xy,
                            np.zeros(2),
                            xy - 0.75 * v[:2],
                            0.75 * xy - 1.5 * v[:2],
                            -v[:2],
                        ]
                        bank = []
                        for index, a in enumerate(alternatives):
                            if index == 0:
                                bank.append(d.command)
                                continue
                            a = a * min(1.0, 0.6 / max(float(np.linalg.norm(a)), 1e-12))
                            bank.append(
                                (float(a[0]), float(a[1]), 0.0 if index == 2 else d.command[2])
                            )
                        if [c["index"] for c in plan["candidates"]] != list(range(6)):
                            raise ValueError("candidate bank coverage differs")
                        costs = []
                        for candidate in plan["candidates"]:
                            index = candidate["index"]
                            np.testing.assert_array_equal(candidate["command"], bank[index])
                            item = next(
                                m
                                for m in row["candidate_artifacts"]
                                if m["frame"] == frame and m["index"] == index
                            )
                            receipt = json.loads((args.producer / item["receipt"]).read_text())
                            if any(receipt[k] != value for k, value in candidate.items()):
                                raise ValueError("candidate scoring receipt differs")
                            with np.load(
                                args.producer / item["artifact"], allow_pickle=False
                            ) as archive:
                                future = {k: archive[k] for k in archive.files}
                            branch = _clone_body(body)
                            donor = integration_hash(model, body.data)
                            if donor != candidate["donor_body_hash"]:
                                raise ValueError("forecast cloned a different actual prefix")
                            for offset in range(40):
                                bq, bv = branch.canonical()
                                expected = (
                                    (0.0, 0.0, 0.0)
                                    if frame + offset > 90
                                    else bank[index]
                                    if offset < 20
                                    else ball_entry_decision(
                                        bq, bv, frame=frame + offset, affordance=affordance
                                    ).command
                                )
                                np.testing.assert_array_equal(
                                    expected, future["forecast_command"][offset]
                                )
                                physical = branch.step(
                                    frame + offset, future["joint_target_rad"][offset].copy()
                                )
                                for key, value in physical.items():
                                    np.testing.assert_array_equal(value[0], future[key][offset])
                            forecast_native += 400
                            candidate_count += 1
                            if integration_hash(model, body.data) != donor:
                                raise ValueError("candidate replay changed actual body")
                            bq = np.vstack((future["canonical_qpos"], branch.canonical()[0]))
                            bv = np.vstack((future["canonical_qvel"], branch.canonical()[1]))
                            ds = [
                                ball_entry_decision(x, y, frame=frame + j, affordance=affordance)
                                for j, (x, y) in enumerate(zip(bq, bv, strict=True))
                            ]
                            ready = next(
                                (j for j, decision in enumerate(ds) if decision.ready), None
                            )
                            stop = ready if ready is not None else 40
                            forces = future["ball_contact_force_by_body_n"][:stop, :, nonfeet]
                            heights = future["pelvis_z_per_substep_m"][:stop]
                            quat = future["target_body_canonical_qpos_after_substep"][:stop, :, 3:7]
                            peak = float(forces.max()) if stop else 0.0
                            low = float(heights.min()) if stop else float(bq[0, 2])
                            up = (
                                float((1 - 2 * (quat[:, :, 1] ** 2 + quat[:, :, 2] ** 2)).min())
                                if stop
                                else 1.0
                            )
                            safe = peak <= 1.0 and low >= 0.65 and up >= 0.9
                            distances = [decision.horizontal_error_m for decision in ds]
                            speed2 = float(np.dot(bv[stop, :2], bv[stop, :2]))
                            value = (
                                0.01 * stop / 40 + 0.02 * speed2
                                if ready is not None
                                else 1
                                + 10 * min(distances) ** 2
                                + 5 * distances[-1] ** 2
                                + 0.02 * speed2
                            )
                            cost = float(value) if safe else None
                            expected_score = dict(
                                predicted_safe=bool(safe),
                                cost=cost,
                                first_ready_offset=ready,
                                evaluated_navigation_prefix_frames=stop,
                                maximum_prefix_nonfoot_force_n=peak,
                                minimum_prefix_pelvis_z_m=low,
                                minimum_prefix_upright=up,
                                minimum_entry_error_m=min(distances),
                                end_entry_error_m=distances[-1],
                                root_speed_at_evaluation_mps=float(np.sqrt(speed2)),
                            )
                            if any(candidate[k] != value for k, value in expected_score.items()):
                                raise ValueError("literal independent score differs")
                            costs.append(cost)
                        eligible = [i for i, c in enumerate(costs) if c is not None]
                        selected = min(eligible, key=lambda i: costs[i]) if eligible else 2
                        if (
                            plan["selected_index"] != selected
                            or plan["all_candidates_predicted_unsafe"] != (not bool(eligible))
                            or plan["selection_predicted_safe"] != bool(eligible)
                        ):
                            raise ValueError("deterministic selection/fallback differs")
                        held = bank[selected]
                    if stage == 0 and held is not None:
                        command = held
                    if d.ready:
                        stage, handoff = 1, frame
                        if (
                            integration_hash(model, body.data)
                            != row["handoff_native_state_unchanged_hash"]
                        ):
                            raise ValueError("handoff native state differs")
                np.testing.assert_array_equal(command, raw["entry_command"][frame])
                np.testing.assert_array_equal(goal, raw["entry_target_root_xy_m"][frame])
                if raw["entry_stage"][frame] != stage or raw["entry_error_m"][frame] != error:
                    raise ValueError("entry state/error differs")
                physical = body.step(frame, raw["joint_target_rad"][frame].copy())
                for key, value in physical.items():
                    np.testing.assert_array_equal(value[0], raw[key][frame])
            if handoff != row["handoff_frame"] or (stage == 2) != row["entry_timed_out"]:
                raise ValueError("terminal handoff summary differs")
            if summarize_reference_ball_strike(model, raw) != row["strike"]:
                raise ValueError("literal actual strike summary differs")
            results.append(
                dict(
                    job=name,
                    actual_native_steps=3000,
                    candidate_native_steps=400 * candidate_count,
                    actual_and_all_candidate_physics_bit_exact=True,
                    command_bank_score_selection_and_entry_logic_exact=True,
                    native=body.review(),
                    strike=row["strike"],
                )
            )
            write_once(args.output / (name + "-complete.json"), results[-1])
            print("PREDICTIVE_ENTRY_RECORDED_REVIEW_EXACT", name, candidate_count, flush=True)
        if forecast_native != complete["forecast_native_steps"]:
            raise ValueError("total candidate native accounting differs")
        for p, h in {**declaration["input_file_hashes"], **pins}.items():
            if digest(Path(p)) != h:
                raise ValueError("source/assets/evidence changed during reader execution")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                additional_actual_native_steps=24000,
                additional_candidate_native_steps=forecast_native,
                additional_total_native_steps=24000 + forecast_native,
                neural_policy_replayed=False,
                navigation_planner_reexecuted=False,
                optimizer_steps=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json", dict(error=repr(error), promotion_authorized=False)
        )
        raise


if __name__ == "__main__":
    main()
