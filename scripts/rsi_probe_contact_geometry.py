"""Finite own-NN contact proposal bank; no teacher-label lookup or real hardware."""

import argparse
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path


def choose_candidate(rows):
    """Same research gate/cost as existing teacher; explicit uncertified fallback."""
    if not 3 <= len(rows) <= 6 or [r["law_index"] for r in rows[:3]] != [0, 3, 4]:
        raise ValueError("three zero-C own forecasts and at most three proposals required")
    extras = [r["law_index"] for r in rows[3:]]
    if len(set(extras)) != len(extras) or any(i not in (0, 3, 4) for i in extras):
        raise ValueError("at most one new proposal per declared law")
    for row in rows:
        score = row["score"]
        if type(score["qualified"]) is not bool or (
            score["qualified"]
            and (type(score["cost"]) not in (float, int) or not math.isfinite(score["cost"]))
        ):
            raise ValueError("boolean gate and finite qualified cost required")
    eligible = [i for i, r in enumerate(rows) if r["score"]["qualified"]]
    return min(eligible, key=lambda i: (rows[i]["score"]["cost"], i)) if eligible else 0


def run_course(args, item, source, poses, affordance, pins):
    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.contact_geometry_proposal import forecast_contact_geometry
    from rosclaw_soccer.rsi.full_entry_forecast import (
        FullEntryForecaster,
        feedback_entry_laws,
        nominal_prefix20,
        run_entry_suffix,
        score_full_entry,
    )
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    folder = args.output / f"variant{item.variant_index}"
    folder.mkdir(exist_ok=False)
    body = forecaster = None
    candidates, proposals, actual_rows = [], [], []
    try:
        check_pins(pins)
        oldroot, oldcomplete = source
        previous = oldcomplete["results"][item.variant_index]["results"]
        nominal, old_teacher = previous
        p = oldroot / folder.name / nominal["artifact"]
        if digest(p) != nominal["artifact_hash"]:
            raise ValueError("archived own-state nominal changed")
        with np.load(p, allow_pickle=False) as a:
            gold = {k: a[k] for k in a.files}
        model = mujoco.MjModel.from_binary_path(str(args.world))
        nav = G1SonicNavigation(
            args.model_root,
            item.agent_id,
            SonicNavigationConfig(
                maximum_frames=300,
                model_variant="low_latency",
                inference_threads=1,
                record_default_planner_events=True,
            ),
        )
        body = G1TargetEpisode(
            model,
            world_hash=digest(args.world),
            kp=nav.backend.kp.copy(),
            kd=nav.backend.kd.copy(),
            course=item.course,
            defaults=G1SonicRunupController.default_angles.copy(),
            record_all_ball_contacts=True,
        )
        prefix = nominal_prefix20(nav, body, affordance, step_observer=actual_rows.append)
        for k in prefix:
            np.testing.assert_array_equal(prefix[k], gold[k][:20])
        np.savez_compressed(folder / "own-prefix20.npz", **prefix)
        forecaster = FullEntryForecaster(world_hash=digest(args.world))
        laws = feedback_entry_laws()
        evidence_hash = hash_json(json.loads((args.output / "declaration.json").read_text()))

        def predict(law_index, offset, proposal=None):
            future = forecaster.forecast(
                nav,
                body,
                model_root=args.model_root,
                source_poses=poses,
                source_evidence_hash=evidence_hash,
                affordance=affordance,
                law=laws[law_index],
                contact_reference_offset_m=offset,
            )
            if (
                future.new_attempted_native_steps != 2800
                or future.new_integrated_native_steps != 2800
            ):
                raise ValueError("complete2800 native suffix required")
            index = len(candidates)
            artifact = folder / f"candidate{index}.npz"
            np.savez_compressed(artifact, **future.arrays)
            row = dict(
                index=index,
                law_index=law_index,
                law=asdict(laws[law_index]),
                offset_root_m=list(offset),
                geometry=proposal,
                report=future.report,
                score=score_full_entry(model, future.arrays),
                artifact=artifact.name,
                artifact_hash=digest(artifact),
                donor_body_hash=future.donor_body_hash,
                donor_navigation_hash=future.donor_navigation_hash,
                new_attempted_native_steps=2800,
                new_integrated_native_steps=2800,
                promotion_authorized=False,
                hardware_authorized=False,
            )
            write_once(folder / f"candidate{index}.json", row)
            candidates.append(row)
            print(
                "CONTACT_GEOMETRY_FUTURE",
                item.parent_index,
                item.variant_index,
                index,
                row["score"]["qualified"],
                flush=True,
            )
            return future

        for law_index in (0, 3, 4):
            future = predict(law_index, (0.0, 0.0, 0.0))
            if law_index == 0:
                for k in future.arrays:
                    np.testing.assert_array_equal(future.arrays[k], gold[k][20:])
            own_complete = {k: np.concatenate((prefix[k], future.arrays[k])) for k in prefix}
            proposal = forecast_contact_geometry(
                model, own_complete, handoff_frame=future.report["handoff_frame"]
            )
            proposals.append(dict(law_index=law_index, proposal=proposal))
        write_once(
            folder / "proposals.json",
            dict(proposals=proposals, source="own_warm_native_forecasts_not_archive_or_labels"),
        )
        for item_proposal in proposals:
            proposal = item_proposal["proposal"]
            if proposal["applicable"]:
                predict(item_proposal["law_index"], tuple(proposal["offset_root_m"]), proposal)
        selected = choose_candidate(candidates)
        chosen = candidates[selected]
        with np.load(folder / chosen["artifact"], allow_pickle=False) as a:
            selected_future = {k: a[k] for k in a.files}
        if (
            forecaster.attempted_native_steps != 2800 * len(candidates)
            or forecaster.integrated_native_steps != forecaster.attempted_native_steps
        ):
            raise ValueError("all forecast accounting must include every candidate")
        suffix, report = run_entry_suffix(
            nav,
            body,
            model_root=args.model_root,
            source_poses=poses,
            source_evidence_hash=evidence_hash,
            affordance=affordance,
            law=laws[chosen["law_index"]],
            contact_reference_offset_m=tuple(chosen["offset_root_m"]),
            step_observer=actual_rows.append,
        )
        for k in suffix:
            np.testing.assert_array_equal(suffix[k], selected_future[k])
        if report != chosen["report"]:
            raise ValueError("own predicted/actual entry/contact reports differ")
        arrays = {k: np.concatenate((prefix[k], suffix[k])) for k in prefix}
        for k in arrays:
            np.testing.assert_array_equal(arrays[k], np.concatenate([r[k] for r in actual_rows]))
        if body.attempted_steps != 3000 or body.integrated_steps != 3000 or len(actual_rows) != 300:
            raise ValueError("actual H300 accounting differs")
        artifact = folder / "actual.npz"
        np.savez_compressed(artifact, **arrays)
        score = score_full_entry(model, suffix)
        result = dict(
            course=asdict(item),
            score=score,
            strike=summarize_reference_ball_strike(model, arrays),
            candidates=candidates,
            proposals=proposals,
            selected_index=selected,
            old_fixed_six_teacher_score_comparison_only=old_teacher["score"],
            own_nominal_score=candidates[0]["score"],
            old_teacher_success_lost=old_teacher["score"]["qualified"] and not score["qualified"],
            old_teacher_failure_rescued=not old_teacher["score"]["qualified"]
            and score["qualified"],
            added_unsafe_vs_old_teacher=old_teacher["score"]["physical_prefix_and_suffix_safe"]
            and not score["physical_prefix_and_suffix_safe"],
            actual_native_steps=3000,
            forecast_native_steps=forecaster.integrated_native_steps,
            own_prefix_and_nominal_future_bit_exact_to_archive=True,
            selected_full_future_all18_and_report_bit_exact_to_actual=True,
            uncertified_fallback=not any(r["score"]["qualified"] for r in candidates),
            artifact=artifact.name,
            artifact_hash=digest(artifact),
            report=report,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        check_pins(pins)
        write_once(folder / "complete.json", result)
        print(
            "CONTACT_GEOMETRY_ACTUAL",
            item.parent_index,
            item.variant_index,
            selected,
            score["qualified"],
            flush=True,
        )
        return result
    except Exception as error:
        if actual_rows:
            np.savez_compressed(
                folder / "observed-control-ticks.npz",
                **{k: np.concatenate([r[k] for r in actual_rows]) for k in actual_rows[0]},
            )
        if body is not None:
            np.savez_compressed(
                folder / "last-native-state.npz",
                qpos=body.data.qpos.copy(),
                qvel=body.data.qvel.copy(),
                time_sec=np.array([body.data.time]),
            )
        write_once(
            folder / "failure.json",
            dict(
                error=repr(error),
                actual_attempted_steps=body.attempted_steps if body else 0,
                actual_integrated_steps=body.integrated_steps if body else 0,
                forecast_attempted_steps=forecaster.attempted_native_steps if forecaster else 0,
                forecast_integrated_steps=forecaster.integrated_native_steps if forecaster else 0,
                completed_candidates=len(candidates),
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "inventory",
        "bank",
        "actor-source",
        "world",
        "motion-csv",
        "model-root",
        "core-source",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-actor-head", "expected-inventory-hash"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--previous-producer", type=Path, action="append", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cluster", type=int, choices=range(4))
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--expected-preflight-hash")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head:
        raise ValueError("sealed contact geometry source required")
    for name in (Path(__file__).name, "rsi_native_context_recovery_common.py"):
        if (
            subprocess.check_output(["git", "show", f"{head}:scripts/{name}"])
            != (Path.cwd() / "scripts" / name).read_bytes()
        ):
            raise ValueError("bootstrap differs from sealed source")
    if args.preflight_only:
        if any(x is not None for x in (args.cluster, args.preflight, args.expected_preflight_hash)):
            raise ValueError("zero-actor preflight cannot allocate worker")
    elif any(x is None for x in (args.cluster, args.preflight, args.expected_preflight_hash)):
        raise ValueError("worker needs full bank zero-actor proof")
    from scripts.rsi_native_context_recovery_common import admit_inventory, check_pins, digest

    _, original_bank, pins, courses = admit_inventory(args)
    for p in [args.world, args.motion_csv] + [
        args.model_root / s
        for s in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]:
        if pins.get(str(p)) != digest(p):
            raise ValueError("same world, motion and foundation assets required")
    previous = {}
    for root in args.previous_producer:
        p = root / "complete.json"
        if pins.get(str(p)) != digest(p):
            raise ValueError("bound historical comparison required")
        record = json.loads(p.read_text())
        if record["cluster"] in previous:
            raise ValueError("unique previous clusters required")
        previous[record["cluster"]] = (root, record)
    if set(previous) != set(range(4)):
        raise ValueError("all four historical comparison clusters required")
    if not args.preflight_only:
        if digest(args.preflight / "complete.json") != args.expected_preflight_hash:
            raise ValueError("zero-actor proof hash differs")
        old = json.loads((args.preflight / "declaration.json").read_text())
        proof = json.loads((args.preflight / "complete.json").read_text())
        if (
            old["input_file_hashes"] != pins
            or old["source_head"] != head
            or proof["new_native_steps"] != 0
            or proof["new_actor_allocations"] != 0
            or proof["new_model_calls"] != 0
            or proof["bank_hash"] != original_bank["bank_hash"]
            or proof["declaration_hash"] != digest(args.preflight / "declaration.json")
            or (args.preflight / "failure.json").exists()
        ):
            raise ValueError("exact same-source full-bank proof required")
        pins.update({str(p): digest(p) for p in args.preflight.iterdir() if p.is_file()})
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.own_forecast_contact_geometry_bank.v1",
        source_head=head,
        input_file_hashes=pins,
        bank_hash=original_bank["bank_hash"],
        courses=[asdict(c) for c in courses],
        cluster=args.cluster,
        laws=[0, 3, 4],
        maximum_candidates_per_course=6,
        full_bank_native_upper_bound=316800,
        optimizer_steps=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)
    results = []
    try:
        if args.preflight_only:
            check_pins(pins)
            write_once(
                args.output / "complete.json",
                dict(
                    source_head=head,
                    bank_hash=original_bank["bank_hash"],
                    declaration_hash=digest(args.output / "declaration.json"),
                    new_native_steps=0,
                    new_actor_allocations=0,
                    new_model_calls=0,
                    promotion_authorized=False,
                    hardware_authorized=False,
                ),
            )
            print("FULL16_CONTACT_GEOMETRY_ADMITTED_ZERO_ACTORS", flush=True)
            return
        import numpy as np

        from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance

        choices = [Path(p) for p in pins if Path(p).name == "entry-affordance.json"]
        if len(choices) != 1:
            raise ValueError("one original bound entry affordance required")
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
            raise ValueError("original entry affordance differs")
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        for item in courses:
            if item.parent_index == args.cluster:
                results.append(
                    run_course(args, item, previous[args.cluster], poses, affordance, pins)
                )
        if len(results) != 4:
            raise ValueError("complete fixed four-course cluster required")
        check_pins(pins)
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                cluster=args.cluster,
                source_head=head,
                actual_native_steps=sum(r["actual_native_steps"] for r in results),
                forecast_native_steps=sum(r["forecast_native_steps"] for r in results),
                qualified=sum(r["score"]["qualified"] for r in results),
                old_teacher_success_lost=sum(r["old_teacher_success_lost"] for r in results),
                old_teacher_failure_rescued=sum(r["old_teacher_failure_rescued"] for r in results),
                added_unsafe_vs_old_teacher=sum(r["added_unsafe_vs_old_teacher"] for r in results),
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
                completed_courses=len(results),
                completed_actual_steps=sum(r["actual_native_steps"] for r in results),
                completed_forecast_steps=sum(r["forecast_native_steps"] for r in results),
                incomplete_course_counts_in_case_failure=True,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
