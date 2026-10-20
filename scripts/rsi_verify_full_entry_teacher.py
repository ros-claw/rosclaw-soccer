"""Reintegrate all full-entry actual and prospective recorded PD controls.

Not neural-policy replay: shares the score function and qualified warm-copy
helper. Each forecast starts from independently replayed actual frame20, and
all physical arrays must be exact. Scope is explicit, no promotion authority.
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
        raise ValueError("sealed source and unused verification root required")
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
                "SubState",
                "-p",
                "ExecMainStatus",
            ],
            text=True,
        ).splitlines()
    )
    if terminal != {
        "MainPID": "0",
        "Result": "success",
        "SubState": "exited",
        "ExecMainStatus": "0",
    }:
        raise ValueError("closed successful durable producer required")
    if digest(args.producer / "complete.json") != args.expected_producer_hash:
        raise ValueError("producer closure identity differs")
    declaration = json.loads((args.producer / "declaration.json").read_text())
    complete = json.loads((args.producer / "complete.json").read_text())
    family = declaration.get("law_family", "held20")
    if family not in ("held20", "feedback"):
        raise ValueError("unknown full-task command-law family")
    expected_schema = (
        "soccer.rsi.full_remaining_entry_teacher4.v1"
        if family == "held20"
        else "soccer.rsi.full_remaining_entry_teacher4.v2"
    )
    candidate_count = 7 if family == "held20" else 6
    expected_forecast = 4 * candidate_count * 2800
    if (
        declaration["schema"] != expected_schema
        or declaration["source_head"] != args.expected_producer_head
        or complete["actual_native_steps"] != 24000
        or complete["forecast_native_steps"] != expected_forecast
        or complete["optimizer_steps"] != 0
        or (args.producer / "failure.json").exists()
        or [r["job"] for r in complete["results"]]
        != [f"course{i}-{arm}" for i in range(4) for arm in ("nominal", "full_teacher")]
    ):
        raise ValueError("complete exact four-course paired task required")
    for p, h in declaration["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("historical producer identity differs")
    if declaration["input_file_hashes"].get(str(args.world)) != digest(args.world):
        raise ValueError("same world binding required")
    paths = [
        p
        for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [p for p in args.producer.iterdir() if p.is_file()]
    pins = {str(p): digest(p) for p in paths}
    for row in complete["results"]:
        if digest(args.producer / (row["job"] + ".npz")) != row["artifact_hash"]:
            raise ValueError("actual raw evidence identity differs")
        for candidate in row["candidates"]:
            p = (args.producer / candidate["artifact"]).resolve()
            if (
                not p.is_relative_to(args.producer.resolve())
                or digest(p) != candidate["artifact_hash"]
            ):
                raise ValueError("candidate artifact identity differs")
            if json.loads((args.producer / (p.stem + ".json")).read_text()) != candidate:
                raise ValueError("individual candidate receipt differs")

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.physics.sonic_navigation_forecast import _clone_body
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision
    from rosclaw_soccer.rsi.full_entry_forecast import (
        feedback_entry_laws,
        full_entry_laws,
        score_full_entry,
    )
    from rosclaw_soccer.rsi.full_task_branch import integration_hash
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.full_entry_recorded_review.v1",
            input_file_hashes=pins,
            producer_terminal=terminal,
            expected_additional_native_steps=24000 + expected_forecast,
            neural_policy_replayed=False,
            navigation_planner_reexecuted=False,
            warm_clone_and_score_helpers_shared=True,
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
            raise ValueError("past affordance binding differs")
        model = mujoco.MjModel.from_binary_path(str(args.world))
        kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
        total = 0
        results = []

        def replay(body, raw, law, start):
            stage, handoff = 0, None
            goal, error = np.zeros(2), 0.0
            for frame in range(start, 300):
                offset = frame - start
                q, v = body.canonical()
                d = None
                cmd = (0.0, 0.0, 0.0)
                if stage in (0, 2):
                    d = ball_entry_decision(q, v, frame=frame, affordance=affordance)
                    goal, error = np.asarray(d.target_root_xy_m), d.horizontal_error_m
                    cmd = law.command(q, v, frame=frame, affordance=affordance)
                    if d.timed_out:
                        stage = 2
                if d is not None and d.ready:
                    handoff, stage = frame, 1
                np.testing.assert_array_equal(cmd, raw["entry_command"][offset])
                np.testing.assert_array_equal(goal, raw["entry_target_root_xy_m"][offset])
                if raw["entry_stage"][offset] != stage or raw["entry_error_m"][offset] != error:
                    raise ValueError("causal task entry trace differs")
                physical = body.step(frame, raw["joint_target_rad"][offset].copy())
                for key, value in physical.items():
                    np.testing.assert_array_equal(value[0], raw[key][offset])
            return handoff

        for row in complete["results"]:
            name = row["job"]
            expected_candidates = candidate_count if name.endswith("-full_teacher") else 0
            if len(row["candidates"]) != expected_candidates:
                raise ValueError("all declared candidate branches required")
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
            for frame in range(20):
                q, v = body.canonical()
                d = ball_entry_decision(q, v, frame=frame, affordance=affordance)
                np.testing.assert_array_equal(d.command, raw["entry_command"][frame])
                physical = body.step(frame, raw["joint_target_rad"][frame].copy())
                for key, value in physical.items():
                    np.testing.assert_array_equal(value[0], raw[key][frame])
            laws = (
                full_entry_laws(*body.canonical(), affordance=affordance)
                if family == "held20"
                else feedback_entry_laws()
            )
            costs = []
            for index, candidate in enumerate(row["candidates"]):
                if candidate["index"] != index or candidate["name"] != laws[index].name:
                    raise ValueError("candidate bank order differs")
                with np.load(args.producer / candidate["artifact"], allow_pickle=False) as archive:
                    future = {k: archive[k] for k in archive.files}
                branch = _clone_body(body)
                donor = integration_hash(model, body.data)
                if donor != candidate["donor_body_hash"]:
                    raise ValueError("forecast did not start at actual replayed prefix")
                handoff = replay(branch, future, laws[index], 20)
                if handoff != candidate["report"]["handoff_frame"]:
                    raise ValueError("candidate handoff timing differs")
                if integration_hash(model, body.data) != donor:
                    raise ValueError("candidate replay changed donor body")
                score = score_full_entry(model, future)
                if score != candidate["score"]:
                    raise ValueError("candidate shared-metric score differs")
                costs.append(score["cost"])
                total += 2800
            eligible = [i for i, cost in enumerate(costs) if cost is not None]
            selected = min(eligible, key=lambda i: costs[i]) if eligible else 0
            if row["selected_index"] != selected:
                raise ValueError("deterministic candidate selection differs")
            suffix = {k: a[20:] for k, a in raw.items()}
            handoff = replay(body, suffix, laws[selected], 20)
            if (
                handoff != row["handoff_frame"]
                or score_full_entry(model, suffix) != row["actual_score"]
            ):
                raise ValueError("actual task result differs")
            if summarize_reference_ball_strike(model, raw) != row["strike"]:
                raise ValueError("whole actual contact summary differs")
            if candidates := row["candidates"]:
                with np.load(
                    args.producer / candidates[selected]["artifact"], allow_pickle=False
                ) as chosen:
                    for key in suffix:
                        np.testing.assert_array_equal(suffix[key], chosen[key])
            total += 3000
            results.append(
                dict(
                    job=name,
                    actual_physics_bit_exact=True,
                    all_candidate_physics_bit_exact=True,
                    candidate_count=len(costs),
                    own_neural_reexecution_not_verified_by_this_reader=True,
                    handoff_frame=handoff,
                    actual_score=row["actual_score"],
                    native=body.review(),
                )
            )
            write_once(args.output / (name + "-complete.json"), results[-1])
            print("FULL_ENTRY_RECORDED_REVIEW_EXACT", name, len(costs), flush=True)
        if total != 24000 + expected_forecast:
            raise ValueError("native replay accounting differs")
        for p, h in {**declaration["input_file_hashes"], **pins}.items():
            if digest(Path(p)) != h:
                raise ValueError("sealed evidence/source changed during verification")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                additional_actual_native_steps=24000,
                additional_forecast_native_steps=expected_forecast,
                additional_total_native_steps=total,
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
