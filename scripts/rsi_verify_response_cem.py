"""Recorded native/target/reference and CEM sequence review of one closed shard.

Shares native clone/reference/score and pure CEM helpers; does NOT rerun NN math.
"""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path


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
    parser.add_argument("--previous-producer", type=Path, action="append", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists():
        raise ValueError("sealed reader and unused output required")
    for name in (
        Path(__file__).name,
        "rsi_response_cem_builder.py",
        "rsi_native_context_recovery_common.py",
    ):
        if (
            subprocess.check_output(["git", "show", f"{head}:scripts/{name}"])
            != (Path.cwd() / "scripts" / name).read_bytes()
        ):
            raise ValueError("reader bootstrap differs from sealed source")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if any(
        p.endswith(".py")
        for p in subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "src", "scripts"], text=True
        ).splitlines()
    ):
        raise ValueError("unsealed reader Python source refused")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

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
    if terminal != dict(MainPID="0", Result="success", SubState="exited", ExecMainStatus="0"):
        raise ValueError("producer must be closed and successful before replay")
    if digest(args.producer / "complete.json") != args.expected_producer_hash:
        raise ValueError("producer completion identity differs")
    declaration = json.loads((args.producer / "declaration.json").read_text())
    complete = json.loads((args.producer / "complete.json").read_text())
    if (
        declaration["schema"] != "soccer.rsi.bounded_response_cem16.v1"
        or declaration["source_head"] != args.expected_producer_head
        or complete["source_head"] != args.expected_producer_head
        or complete["cluster"] != declaration["cluster"]
        or complete["bank_hash"] != declaration["bank_hash"]
        or complete["actual_native_steps"] != 12000
        or complete["forecast_native_steps"] != 336000
        or complete["cem_distribution_updates"] != 24
        or complete["neural_optimizer_updates"] != 0
        or len(complete["results"]) != 4
        or any(args.producer.rglob("failure.json"))
    ):
        raise ValueError("complete fixed four-course response CEM cluster required")
    pins = dict(declaration["input_file_hashes"])
    check_pins(pins)
    for p in (args.world, args.motion_csv):
        if pins.get(str(p)) != digest(p):
            raise ValueError("original world and motion CSV required")
    previous = {}
    for root in args.previous_producer:
        p = root / "complete.json"
        if pins.get(str(p)) != digest(p):
            raise ValueError("bound original teacher comparison required")
        record = json.loads(p.read_text())
        if record["cluster"] in previous:
            raise ValueError("unique previous clusters required")
        previous[record["cluster"]] = (root, record)
    if set(previous) != set(range(4)):
        raise ValueError("all four original comparison roots required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    pins.update({str(p): digest(p) for p in args.producer.rglob("*") if p.is_file()})

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.physics.sonic_navigation_forecast import _clone_body
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.contact_response_cem import bounded_offset
    from rosclaw_soccer.rsi.full_entry_forecast import feedback_entry_laws, score_full_entry
    from rosclaw_soccer.rsi.full_task_branch import integration_hash
    from rosclaw_soccer.rsi.recorded_entry_review import (
        replay_prefix,
        replay_suffix,
        validate_recording,
    )
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.close_contact_courses import close_contact_perturbations
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_response_cem_builder import build_candidates

    bank = close_contact_perturbations()
    if (
        preflight_receiving_courses(bank) != declaration["bank_hash"]
        or [asdict(c) | {"course": list(c.course)} for c in bank] != declaration["courses"]
    ):
        raise ValueError("full16 course commitment differs before replay")
    shard = [c for c in bank if c.parent_index == complete["cluster"]]

    def load(p):
        with np.load(p, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    # Validate every raw file before a model/body is allocated; one bad case aborts all.
    recordings = []
    for item, row in zip(shard, complete["results"], strict=True):
        folder = args.producer / f"variant{item.variant_index}"
        if row != json.loads((folder / "complete.json").read_text()) or row["course"] != asdict(
            item
        ) | {"course": list(item.course)}:
            raise ValueError("ordered complete case identity differs")
        receipts = [row] + row["candidates"]
        if (
            len(row["candidates"]) != 30
            or len(row["proposals"]) != 6
            or row["actual_native_steps"] != 3000
            or row["forecast_native_steps"] != 84000
        ):
            raise ValueError("exact per-course physical budget required")
        arrays = []
        for i, receipt in enumerate(receipts):
            p = (folder / receipt["artifact"]).resolve()
            if not p.is_relative_to(folder.resolve()) or digest(p) != receipt["artifact_hash"]:
                raise ValueError("raw artifact identity differs")
            data = load(p)
            validate_recording(data, frames=300 if i == 0 else 280, nbody=46)
            if i and receipt != json.loads((folder / f"candidate{i - 1}.json").read_text()):
                raise ValueError("candidate individual receipt differs")
            arrays.append(data)
        if [r["index"] for r in row["candidates"]] != list(range(30)) or [
            r["law_index"] for r in row["candidates"]
        ] != [0, 0, 3, 3, 4, 4] + [0] * 8 + [3] * 8 + [4] * 8:
            raise ValueError("exact response candidate order required")
        for r in row["candidates"]:
            bounded_offset(r["offset_root_m"])
            if r["new_attempted_native_steps"] != 2800 or r["new_integrated_native_steps"] != 2800:
                raise ValueError("complete individual forecast required")
        recordings.append((item, row, arrays[0], arrays[1:]))
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.response_cem_recorded_review.v1",
            source_head=head,
            input_file_hashes=pins,
            producer_terminal=terminal,
            bank_hash=declaration["bank_hash"],
            expected_additional_native_steps=348000,
            encoder_decoder_planner_math_replayed=False,
            shared_native_clone_reference_score_cem_helpers=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    total = 0
    current, origin = None, 0
    results = []
    try:
        choices = [
            Path(p)
            for p in declaration["input_file_hashes"]
            if Path(p).name == "entry-affordance.json"
        ]
        if len(choices) != 1:
            raise ValueError("one bound affordance required")
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
            raise ValueError("original affordance differs")
        model = mujoco.MjModel.from_binary_path(str(args.world))
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
        laws = feedback_entry_laws()
        evidence_hash = hash_json(declaration)
        for item, row, raw, futures in recordings:
            prefix = {k: a[:20] for k, a in raw.items()}
            # Recreate CEM draws/updates from recorded, subsequently replayed responses.
            # This reuses the pure generator; it is not independent optimizer math.
            reconstructed = []

            def recorded_prediction(
                law_index,
                offset,
                proposal=None,
                *,
                _rows=row,
                _rebuilt=reconstructed,
                _futures=futures,
            ):
                from types import SimpleNamespace

                index = len(_rebuilt)
                recorded = _rows["candidates"][index]
                if (
                    recorded["law_index"] != law_index
                    or recorded["offset_root_m"] != list(offset)
                    or recorded["geometry"] != proposal
                ):
                    raise ValueError("CEM samples are not generated from their own prior responses")
                _rebuilt.append(recorded)
                return SimpleNamespace(arrays=_futures[index])

            folder = args.output / f"variant{item.variant_index}-regenerated-cem"
            folder.mkdir()
            iterations = build_candidates(
                recorded_prediction,
                reconstructed,
                prefix,
                previous[item.parent_index],
                item,
                folder,
            )
            if iterations != row["proposals"] or len(reconstructed) != 30:
                raise ValueError("complete own-response distribution update trace differs")
            body = G1TargetEpisode(
                model,
                world_hash=digest(args.world),
                kp=kp,
                kd=kd,
                course=item.course,
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )
            current, origin = body, body.integrated_steps
            replay_prefix(body, prefix, affordance)
            total += 200
            current, origin = None, 0
            donor = integration_hash(model, body.data)
            scores = []
            for candidate, future in zip(row["candidates"], futures, strict=True):
                if candidate["donor_body_hash"] != donor or candidate["law"] != asdict(
                    laws[candidate["law_index"]]
                ):
                    raise ValueError("own-state donor or original law differs")
                branch = _clone_body(body)
                current, origin = branch, branch.integrated_steps
                replay_suffix(
                    branch,
                    future,
                    law=laws[candidate["law_index"]],
                    offset_root_m=tuple(candidate["offset_root_m"]),
                    affordance=affordance,
                    source_poses=poses,
                    source_evidence_hash=evidence_hash,
                    report=candidate["report"],
                )
                if integration_hash(model, body.data) != donor:
                    raise ValueError("review clone changed own prefix donor")
                score = score_full_entry(model, future)
                if score != candidate["score"]:
                    raise ValueError("recorded candidate task metric differs")
                scores.append(score)
                total += 2800
                current, origin = None, 0
            eligible = [i for i, s in enumerate(scores) if s["qualified"]]
            selected = min(eligible, key=lambda i: (scores[i]["cost"], i)) if eligible else 0
            if selected != row["selected_index"] or row["uncertified_fallback"] != (not eligible):
                raise ValueError("original task-gate/cost selection differs")
            chosen = row["candidates"][selected]
            suffix = {k: a[20:] for k, a in raw.items()}
            current, origin = body, body.integrated_steps
            replay_suffix(
                body,
                suffix,
                law=laws[chosen["law_index"]],
                offset_root_m=tuple(chosen["offset_root_m"]),
                affordance=affordance,
                source_poses=poses,
                source_evidence_hash=evidence_hash,
                report=row["report"],
            )
            for k in suffix:
                np.testing.assert_array_equal(suffix[k], futures[selected][k])
            if (
                score_full_entry(model, suffix) != row["score"]
                or summarize_reference_ball_strike(model, raw) != row["strike"]
                or row["report"] != chosen["report"]
            ):
                raise ValueError("actual native task outcome differs")
            total += 2800
            current, origin = None, 0
            old = previous[item.parent_index][1]["results"][item.variant_index]["results"][1][
                "score"
            ]
            flags = dict(
                old_teacher_success_lost=old["qualified"] and not row["score"]["qualified"],
                old_teacher_failure_rescued=not old["qualified"] and row["score"]["qualified"],
                added_unsafe_vs_old_teacher=old["physical_prefix_and_suffix_safe"]
                and not row["score"]["physical_prefix_and_suffix_safe"],
            )
            if any(row[k] != v for k, v in flags.items()):
                raise ValueError("retention/rescue/safety accounting differs")
            result = dict(
                course=row["course"],
                score=row["score"],
                selected_index=selected,
                actual_and_all30_forecast_native_physics_bit_exact=True,
                recorded_action_to_target_and_posthandoff_reference_verified=True,
                own_response_cem_sequence_reconstructed=True,
                neural_math_replayed=False,
                **flags,
            )
            results.append(result)
            write_once(args.output / f"variant{item.variant_index}-complete.json", result)
            print(
                "RESPONSE_CEM_REVIEW_NATIVE_EXACT",
                item.parent_index,
                item.variant_index,
                selected,
                flush=True,
            )
        if total != 348000:
            raise ValueError("complete recorded replay accounting differs")
        check_pins(pins)
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                cluster=complete["cluster"],
                bank_hash=declaration["bank_hash"],
                additional_native_steps=total,
                new_independent_courses=0,
                encoder_decoder_planner_math_replayed=False,
                optimizer_updates=0,
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
                incomplete_attempted_steps=current.attempted_steps - origin if current else 0,
                incomplete_integrated_steps=current.integrated_steps - origin if current else 0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
