"""Recorded-PD and source-reference review of ONE complete perturbation shard.

This reader does not rerun neural inference/planner math and grants no promotion.
All four public variants, both actual arms and all six forecasts are required.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("producer", "world", "motion-csv", "core-source", "output"):
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
        raise ValueError("closed successful durable shard required")
    if digest(args.producer / "complete.json") != args.expected_producer_hash:
        raise ValueError("producer completion identity differs")
    declaration = json.loads((args.producer / "declaration.json").read_text())
    complete = json.loads((args.producer / "complete.json").read_text())
    if (
        declaration["schema"] != "soccer.rsi.close_contact_perturb16.v1"
        or declaration["stage"] != "cluster_physics"
        or declaration["source_head"] != args.expected_producer_head
        or complete["source_head"] != args.expected_producer_head
        or type(complete["cluster"]) is not int
        or complete["cluster"] not in range(4)
        or declaration["cluster"] != complete["cluster"]
        or complete["bank_hash"] != declaration["bank_hash"]
        or complete["actual_native_steps"] != 24000
        or complete["forecast_native_steps"] != 67200
        or complete["total_new_native_steps"] != 91200
        or complete["optimizer_steps"] != 0
        or len(complete["results"]) != 4
        or any(args.producer.rglob("failure.json"))
    ):
        raise ValueError("whole four-variant paired shard required")
    for p, h in declaration["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("producer source/assets/evidence changed")
    for p in (args.world, args.motion_csv):
        if declaration["input_file_hashes"].get(str(p)) != digest(p):
            raise ValueError("bound original world and source CSV required")
    paths = [
        p
        for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [p for p in args.producer.rglob("*") if p.is_file()]
    pins = {str(p): digest(p) for p in paths}

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.physics.sonic_navigation_forecast import _clone_body
    from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import quarantine_wrist_reference
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision
    from rosclaw_soccer.rsi.contact_reference_offset import offset_contact_reference
    from rosclaw_soccer.rsi.full_entry_forecast import coupled_entry_options, score_full_entry
    from rosclaw_soccer.rsi.full_task_branch import integration_hash
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.close_contact_courses import close_contact_perturbations
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once

    bank = close_contact_perturbations()
    if (
        preflight_receiving_courses(bank) != declaration["bank_hash"]
        or [asdict(x) | {"course": list(x.course)} for x in bank] != declaration["courses"]
    ):
        raise ValueError("full bank declaration differs before physics")
    shard = [x for x in bank if x.parent_index == complete["cluster"]]
    for item, pair in zip(shard, complete["results"], strict=True):
        folder = args.producer / f"variant{item.variant_index}"
        if (
            pair != json.loads((folder / "complete.json").read_text())
            or pair["course"] != asdict(item) | {"course": list(item.course)}
            or [r["arm"] for r in pair["results"]] != ["nominal", "full_teacher"]
            or pair["actual_native_steps"] != 6000
            or pair["forecast_native_steps"] != 16800
        ):
            raise ValueError("whole ordered pair identity differs")
        for row in pair["results"]:
            if row != json.loads((folder / (row["arm"] + "-complete.json")).read_text()):
                raise ValueError("actual individual receipt differs")
            for receipt in [row] + row["candidates"]:
                artifact = (folder / receipt["artifact"]).resolve()
                if (
                    not artifact.is_relative_to(folder.resolve())
                    or digest(artifact) != receipt["artifact_hash"]
                ):
                    raise ValueError("raw artifact identity differs")
            for candidate in row["candidates"]:
                if candidate != json.loads(
                    (folder / f"candidate{candidate['index']}.json").read_text()
                ):
                    raise ValueError("individual candidate receipt differs")
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.close_contact_recorded_review.v1",
            input_file_hashes=pins,
            producer_terminal=terminal,
            cluster=complete["cluster"],
            bank_hash=declaration["bank_hash"],
            expected_additional_native_steps=91200,
            neural_policy_replayed=False,
            navigation_planner_reexecuted=False,
            shared_native_clone_reference_and_score_helpers=True,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    total = 0
    current_body = None
    current_start_count = 0
    try:
        # The producer pins the authenticated historical affordance before NN allocation.
        choices = [
            Path(p)
            for p in declaration["input_file_hashes"]
            if Path(p).name == "entry-affordance.json"
        ]
        if len(choices) != 1:
            raise ValueError("one bound historical entry affordance required")
        stored = json.loads(choices[0].read_text())
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
            raise ValueError("historical affordance identity differs")
        model = mujoco.MjModel.from_binary_path(str(args.world))
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
        options = coupled_entry_options()
        if [x.name for x in options] != declaration["option_names"]:
            raise ValueError("fixed six-option identity differs")

        def replay(body, raw, option, report):
            stage, handoff, reference = 0, None, None
            goal, error = np.zeros(2), 0.0
            for frame in range(20, 300):
                offset = frame - 20
                q, v = body.canonical()
                d, cmd = None, (0.0, 0.0, 0.0)
                if stage in (0, 2):
                    d = ball_entry_decision(q, v, frame=frame, affordance=affordance)
                    goal, error = np.asarray(d.target_root_xy_m), d.horizontal_error_m
                    cmd = option.law.command(q, v, frame=frame, affordance=affordance)
                    if d.timed_out:
                        stage = 2
                if d is not None and d.ready:
                    handoff, stage = frame, 1
                    if (
                        integration_hash(model, body.data)
                        != report["handoff_native_state_unchanged_hash"]
                    ):
                        raise ValueError("full native handoff state differs")
                    held, quarantine = quarantine_wrist_reference(poses, q[:36])
                    prepared = prepare_sonic_reference(
                        held[286:767],
                        q[:36],
                        source_frequency_hz=120,
                        entry_intervals=50,
                        source_evidence_hash=hash_json(declaration),
                    )
                    modified, geometry = offset_contact_reference(
                        model, prepared.reference, offset_root_m=option.offset_root_m
                    )
                    if (
                        geometry != report["contact_reference_adaptation"]
                        or quarantine.contract_hash != report["wrist_reference_quarantine_hash"]
                        or prepared.contract_hash != report["reference_preparation_hash"]
                    ):
                        raise ValueError("source reference and contact geometry receipt differs")
                    reference = modified.initialize(q[:36], frames=410)
                if reference is not None:
                    local = frame - handoff
                    np.testing.assert_array_equal(
                        reference[local : local + 10], raw["encoder_reference_poses"][offset]
                    )
                np.testing.assert_array_equal(cmd, raw["entry_command"][offset])
                np.testing.assert_array_equal(goal, raw["entry_target_root_xy_m"][offset])
                if raw["entry_stage"][offset] != stage or raw["entry_error_m"][offset] != error:
                    raise ValueError("measured task entry trace differs")
                physical = body.step(frame, raw["joint_target_rad"][offset].copy())
                for key, value in physical.items():
                    np.testing.assert_array_equal(value[0], raw[key][offset])
            if handoff != report["handoff_frame"]:
                raise ValueError("measured handoff frame differs")

        results = []
        for item, pair in zip(shard, complete["results"], strict=True):
            folder = args.producer / f"variant{item.variant_index}"
            baseline = None
            reviewed = []
            for row in pair["results"]:
                expected_count = 6 if row["arm"] == "full_teacher" else 0
                if len(row["candidates"]) != expected_count or row["course"] != pair["course"]:
                    raise ValueError("all six candidates and actual own course required")
                if (
                    row["actual_native_steps"] != 3000
                    or row["forecast_native_steps"] != expected_count * 2800
                ):
                    raise ValueError("declared actual/forecast work differs")
                with np.load(folder / row["artifact"], allow_pickle=False) as archive:
                    raw = {k: archive[k] for k in archive.files}
                if not raw or any(len(a) != 300 for a in raw.values()):
                    raise ValueError("whole actual H300 arrays required")
                if baseline is not None:
                    for k in raw:
                        np.testing.assert_array_equal(raw[k][:20], baseline[k][:20])
                body = G1TargetEpisode(
                    model,
                    world_hash=digest(args.world),
                    kp=kp,
                    kd=kd,
                    course=item.course,
                    defaults=G1SonicRunupController.default_angles.copy(),
                    record_all_ball_contacts=True,
                )
                current_body, current_start_count = body, body.integrated_steps
                for frame in range(20):
                    q, v = body.canonical()
                    d = ball_entry_decision(q, v, frame=frame, affordance=affordance)
                    np.testing.assert_array_equal(d.command, raw["entry_command"][frame])
                    np.testing.assert_array_equal(
                        d.target_root_xy_m, raw["entry_target_root_xy_m"][frame]
                    )
                    if (
                        raw["entry_stage"][frame] != 0
                        or raw["entry_error_m"][frame] != d.horizontal_error_m
                    ):
                        raise ValueError("measured prefix entry trace differs")
                    physical = body.step(frame, raw["joint_target_rad"][frame].copy())
                    for key, value in physical.items():
                        np.testing.assert_array_equal(value[0], raw[key][frame])
                total += 200
                current_start_count = body.integrated_steps
                scores = []
                for index, candidate in enumerate(row["candidates"]):
                    option = options[index]
                    if (
                        candidate["index"] != index
                        or candidate["name"] != option.name
                        or candidate["law"] != json.loads(json.dumps(asdict(option.law)))
                        or candidate["contact_reference_offset_m"] != list(option.offset_root_m)
                        or candidate["new_integrated_native_steps"] != 2800
                        or candidate["new_attempted_native_steps"] != 2800
                    ):
                        raise ValueError("fixed candidate configuration differs")
                    with np.load(folder / candidate["artifact"], allow_pickle=False) as archive:
                        future = {k: archive[k] for k in archive.files}
                    if set(future) != set(raw) or any(len(a) != 280 for a in future.values()):
                        raise ValueError("whole matching H280 future arrays required")
                    if index == 0:
                        for k in future:
                            np.testing.assert_array_equal(future[k], baseline[k][20:])
                    donor = integration_hash(model, body.data)
                    if candidate["donor_body_hash"] != donor:
                        raise ValueError("forecast does not start at its own actual prefix")
                    branch = _clone_body(body)
                    current_body, current_start_count = branch, branch.integrated_steps
                    replay(branch, future, option, candidate["report"])
                    if integration_hash(model, body.data) != donor:
                        raise ValueError("review candidate mutated donor")
                    score = score_full_entry(model, future)
                    if score != candidate["score"]:
                        raise ValueError("shared candidate metric differs")
                    scores.append(score)
                    total += 2800
                    current_body = None
                eligible = [i for i, s in enumerate(scores) if s["qualified"]]
                selected = min(eligible, key=lambda i: scores[i]["cost"]) if eligible else 0
                if (
                    row["selected_index"] != selected
                    or row["selected_name"] != options[selected].name
                    or row["selected_contact_reference_offset_m"]
                    != list(options[selected].offset_root_m)
                    or row["uncertified_nominal_fallback"] != (expected_count == 6 and not eligible)
                ):
                    raise ValueError("deterministic option selection/fallback differs")
                suffix = {k: a[20:] for k, a in raw.items()}
                current_body, current_start_count = body, body.integrated_steps
                replay(body, suffix, options[selected], row)
                score = score_full_entry(model, suffix)
                if (
                    score != row["score"]
                    or summarize_reference_ball_strike(model, raw) != row["strike"]
                ):
                    raise ValueError("actual task metric/contact summary differs")
                if expected_count:
                    with np.load(
                        folder / row["candidates"][selected]["artifact"], allow_pickle=False
                    ) as chosen:
                        for k in suffix:
                            np.testing.assert_array_equal(suffix[k], chosen[k])
                else:
                    baseline = raw
                total += 2800
                current_body = None
                reviewed.append(
                    dict(
                        arm=row["arm"],
                        score=score,
                        selected_index=selected,
                        actual_physics_bit_exact=True,
                        all_candidate_physics_bit_exact=True,
                        candidate_count=expected_count,
                        posthandoff_contact_reference_reconstructed=True,
                        neural_policy_replayed=False,
                        navigation_planner_reexecuted=False,
                    )
                )
                print(
                    "CLOSE_PERTURB_REVIEW_EXACT",
                    item.parent_index,
                    item.variant_index,
                    row["arm"],
                    expected_count,
                    flush=True,
                )
            a, b = (r["score"] for r in reviewed)
            flags = dict(
                rescue=not a["qualified"] and b["qualified"],
                regression=a["qualified"] and not b["qualified"],
                added_unsafe=a["physical_prefix_and_suffix_safe"]
                and not b["physical_prefix_and_suffix_safe"],
            )
            if any(pair[k] != value for k, value in flags.items()):
                raise ValueError("paired rescue/regression/safety accounting differs")
            result = dict(course=pair["course"], results=reviewed, **flags)
            results.append(result)
            write_once(args.output / f"variant{item.variant_index}-complete.json", result)
        if total != 91200:
            raise ValueError("native replay accounting differs")
        for p, h in {**declaration["input_file_hashes"], **pins}.items():
            if digest(Path(p)) != h:
                raise ValueError("sealed inputs changed during review")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                cluster=complete["cluster"],
                bank_hash=declaration["bank_hash"],
                additional_actual_native_steps=24000,
                additional_forecast_native_steps=67200,
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
            args.output / "failure.json",
            dict(
                error=repr(error),
                completed_native_steps=total,
                incomplete_attempted_native_steps=current_body.attempted_steps - current_start_count
                if current_body is not None
                else 0,
                incomplete_integrated_native_steps=current_body.integrated_steps
                - current_start_count
                if current_body is not None
                else 0,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
