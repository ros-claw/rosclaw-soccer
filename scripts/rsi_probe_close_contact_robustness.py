"""Whole-bank preflight, then four explicitly sharded close-contact pilots.

Each shard has four NEW public diagnostic states and own-state paired controls.
No historical different-state targets/labels, no optimizer or hardware calls.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

OPTION_NAMES = [
    "nominal_feedback__contact_off",
    "nominal_feedback__contact_lower",
    "damped_feedback__contact_off",
    "damped_feedback__contact_lower",
    "strong_damped_feedback__contact_off",
    "strong_damped_feedback__contact_lower",
]


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def run_pair(args, item, poses, affordance, evidence_hash):
    """Called only after pinned source/assets and WHOLE bank admission."""
    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.full_entry_forecast import (
        FullEntryForecaster,
        coupled_entry_options,
        nominal_prefix20,
        run_entry_suffix,
        score_full_entry,
    )
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from scripts.rsi_atomic_artifacts import write_once

    folder = args.output / f"variant{item.variant_index}"
    folder.mkdir(exist_ok=False)
    options = coupled_entry_options()
    if [o.name for o in options] != OPTION_NAMES:
        raise ValueError("fixed six-option protocol changed")
    baseline = None
    rows = []
    completed_actual = completed_forecast = 0
    body = forecaster = None
    try:
        for arm in ("nominal", "full_teacher"):
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
            model = mujoco.MjModel.from_binary_path(str(args.world))
            body = G1TargetEpisode(
                model,
                world_hash=digest(args.world),
                kp=nav.backend.kp.copy(),
                kd=nav.backend.kd.copy(),
                course=item.course,
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )
            prefix = nominal_prefix20(nav, body, affordance)
            candidates = []
            selected = 0
            selected_future = None
            forecaster = FullEntryForecaster(world_hash=digest(args.world))
            if arm == "full_teacher":
                assert baseline is not None
                for key in prefix:
                    np.testing.assert_array_equal(prefix[key], baseline[key][:20])
                for index, option in enumerate(options):
                    future = forecaster.forecast(
                        nav,
                        body,
                        model_root=args.model_root,
                        source_poses=poses,
                        source_evidence_hash=evidence_hash,
                        affordance=affordance,
                        law=option.law,
                        contact_reference_offset_m=option.offset_root_m,
                    )
                    if index == 0:
                        for key in future.arrays:
                            np.testing.assert_array_equal(future.arrays[key], baseline[key][20:])
                    artifact = folder / f"candidate{index}.npz"
                    np.savez_compressed(artifact, **future.arrays)
                    row = dict(
                        index=index,
                        name=option.name,
                        law=asdict(option.law),
                        contact_reference_offset_m=list(option.offset_root_m),
                        report=future.report,
                        artifact=artifact.name,
                        artifact_hash=digest(artifact),
                        score=score_full_entry(model, future.arrays),
                        donor_body_hash=future.donor_body_hash,
                        donor_navigation_hash=future.donor_navigation_hash,
                        new_attempted_native_steps=future.new_attempted_native_steps,
                        new_integrated_native_steps=future.new_integrated_native_steps,
                    )
                    candidates.append(row)
                    write_once(folder / f"candidate{index}.json", row)
                    print(
                        "CLOSE_PERTURB_CANDIDATE",
                        item.parent_index,
                        item.variant_index,
                        index,
                        row["score"]["qualified"],
                        flush=True,
                    )
                eligible = [r["index"] for r in candidates if r["score"]["qualified"]]
                selected = (
                    min(eligible, key=lambda i: candidates[i]["score"]["cost"]) if eligible else 0
                )
                with np.load(
                    folder / candidates[selected]["artifact"], allow_pickle=False
                ) as archive:
                    selected_future = {k: archive[k] for k in archive.files}
                if (
                    forecaster.attempted_native_steps != 16800
                    or forecaster.integrated_native_steps != 16800
                ):
                    raise ValueError("six full-future native accounting differs")
            option = options[selected]
            suffix, report = run_entry_suffix(
                nav,
                body,
                model_root=args.model_root,
                source_poses=poses,
                source_evidence_hash=evidence_hash,
                affordance=affordance,
                law=option.law,
                contact_reference_offset_m=option.offset_root_m,
            )
            if selected_future is not None:
                for key in suffix:
                    np.testing.assert_array_equal(suffix[key], selected_future[key])
                if report != candidates[selected]["report"]:
                    raise ValueError("predicted/actual whole handoff/contact reports differ")
            arrays = {k: np.concatenate((prefix[k], suffix[k])) for k in prefix}
            if arm == "nominal":
                baseline = arrays
            artifact = folder / (arm + ".npz")
            np.savez_compressed(artifact, **arrays)
            report.update(
                arm=arm,
                course=asdict(item),
                artifact=artifact.name,
                artifact_hash=digest(artifact),
                candidates=candidates,
                selected_index=selected,
                selected_name=option.name,
                selected_contact_reference_offset_m=list(option.offset_root_m),
                uncertified_nominal_fallback=arm == "full_teacher"
                and not any(r["score"]["qualified"] for r in candidates),
                score=score_full_entry(model, suffix),
                strike=summarize_reference_ball_strike(model, arrays),
                actual_native_steps=3000,
                forecast_native_steps=forecaster.integrated_native_steps,
                current_course_prefix_exact_to_paired_nominal=arm == "full_teacher",
                candidate0_full_future_exact_to_current_paired_nominal=arm == "full_teacher",
                selected_full_future_exact_to_actual_own_neural_execution=arm == "full_teacher",
            )
            rows.append(report)
            write_once(folder / (arm + "-complete.json"), report)
            completed_actual += 3000
            completed_forecast += forecaster.integrated_native_steps
            body = forecaster = None
            print(
                "CLOSE_PERTURB_ACTUAL",
                item.parent_index,
                item.variant_index,
                arm,
                option.name,
                report["score"],
                flush=True,
            )
        result = dict(
            course=asdict(item),
            results=rows,
            actual_native_steps=6000,
            forecast_native_steps=16800,
            rescue=not rows[0]["score"]["qualified"] and rows[1]["score"]["qualified"],
            regression=rows[0]["score"]["qualified"] and not rows[1]["score"]["qualified"],
            added_unsafe=rows[0]["score"]["physical_prefix_and_suffix_safe"]
            and not rows[1]["score"]["physical_prefix_and_suffix_safe"],
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        write_once(folder / "complete.json", result)
        return result
    except Exception as error:
        write_once(
            folder / "failure.json",
            dict(
                error=repr(error),
                completed_actual_native_steps=completed_actual,
                completed_forecast_native_steps=completed_forecast,
                incomplete_actual_attempted_steps=body.attempted_steps if body is not None else 0,
                incomplete_actual_integrated_steps=body.integrated_steps if body is not None else 0,
                incomplete_forecast_attempted_steps=forecaster.attempted_native_steps
                if forecaster is not None
                else 0,
                incomplete_forecast_integrated_steps=forecaster.integrated_native_steps
                if forecaster is not None
                else 0,
                promotion_authorized=False,
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prior",
        "prior-review",
        "world",
        "motion-csv",
        "model-root",
        "core-source",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-prior-review-hash"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cluster", type=int, choices=range(4))
    parser.add_argument("--bank-preflight", type=Path)
    parser.add_argument("--expected-bank-preflight-hash")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists():
        raise ValueError("sealed source and unused output required")
    if args.preflight_only:
        if any(
            v is not None
            for v in (args.cluster, args.bank_preflight, args.expected_bank_preflight_hash)
        ):
            raise ValueError("whole bank preflight cannot allocate a shard")
    elif (
        args.cluster is None
        or args.bank_preflight is None
        or args.expected_bank_preflight_hash is None
    ):
        raise ValueError("worker requires an explicit shard and sealed whole-bank preflight")
    if digest(args.prior_review / "complete.json") != args.expected_prior_review_hash:
        raise ValueError("preceding coupled independent proof differs")
    previous = json.loads((args.prior / "declaration.json").read_text())
    pc = json.loads((args.prior / "complete.json").read_text())
    rd = json.loads((args.prior_review / "declaration.json").read_text())
    rc = json.loads((args.prior_review / "complete.json").read_text())
    if (
        previous["schema"] != "soccer.rsi.full_remaining_entry_teacher4.v3"
        or pc["total_new_native_steps"] != 91200
        or rc["additional_total_native_steps"] != 91200
        or len(rc["results"]) != 8
        or not all(
            r["actual_physics_bit_exact"]
            and r["all_candidate_physics_bit_exact"]
            and r["posthandoff_contact_reference_reconstructed"]
            for r in rc["results"]
        )
        or rd["input_file_hashes"].get(str(args.prior / "complete.json"))
        != digest(args.prior / "complete.json")
        or any((root / "failure.json").exists() for root in (args.prior, args.prior_review))
    ):
        raise ValueError("whole authentic coupled study and reference reconstruction required")
    old_nom = [r for r in pc["results"] if r["arm"] == "nominal"]
    old_new = [r for r in pc["results"] if r["arm"] == "full_teacher"]
    if (
        len(old_nom) != 4
        or len(old_new) != 4
        or sum(r["actual_score"]["qualified"] for r in old_new)
        <= sum(r["actual_score"]["qualified"] for r in old_nom)
        or any(
            a["actual_score"]["qualified"] and not b["actual_score"]["qualified"]
            for a, b in zip(old_nom, old_new, strict=True)
        )
    ):
        raise ValueError(
            "robustness successor requires authentic development net gain without regression"
        )
    for d in (previous, rd):
        for p, h in d["input_file_hashes"].items():
            if digest(Path(p)) != h:
                raise ValueError("preceding evidence/source/assets changed")
    for p in (args.world, args.motion_csv):
        if previous["input_file_hashes"].get(str(p)) != digest(p):
            raise ValueError("same frozen physical world and motion source required")
    paths = [
        p
        for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [p for root in (args.prior, args.prior_review) for p in root.iterdir() if p.is_file()]
    paths += [
        args.world,
        args.motion_csv,
        args.model_root / "planner_sonic.onnx",
        args.model_root / "low_latency/model_encoder.onnx",
        args.model_root / "low_latency/model_decoder.onnx",
        args.model_root / "low_latency/observation_config.yaml",
    ]
    pins = {str(p): digest(p) for p in paths}
    bank_record = None
    if not args.preflight_only:
        if digest(args.bank_preflight / "complete.json") != args.expected_bank_preflight_hash:
            raise ValueError("whole-bank preflight identity differs")
        bank_record = json.loads((args.bank_preflight / "complete.json").read_text())
        bank_decl = json.loads((args.bank_preflight / "declaration.json").read_text())
        if (
            bank_record["source_head"] != head
            or bank_record["actual_actor_allocations"] != 0
            or bank_decl["input_file_hashes"] != pins
            or (args.bank_preflight / "failure.json").exists()
            or digest(args.bank_preflight / "declaration.json") != bank_record["declaration_hash"]
        ):
            raise ValueError("authentic full-bank source/assets commitment required")
        pins.update({str(p): digest(p) for p in args.bank_preflight.iterdir() if p.is_file()})

    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.close_contact_courses import close_contact_perturbations
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once

    bank = close_contact_perturbations()
    bank_hash = preflight_receiving_courses(bank)
    if bank_record is not None and bank_hash != bank_record["bank_hash"]:
        raise ValueError("worker bank differs from the pre-worker whole-bank admission")
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.close_contact_perturb16.v1",
        source_head=head,
        input_file_hashes=pins,
        bank_hash=bank_hash,
        courses=[asdict(x) for x in bank],
        independent_parent_clusters=4,
        new_public_diagnostic_conditions=16,
        fresh_or_iid_claim=False,
        option_names=OPTION_NAMES,
        cluster=args.cluster,
        stage="bank_preflight" if args.preflight_only else "cluster_physics",
        total_planned_actual_native_steps=96000,
        total_planned_forecast_native_steps=268800,
        shard_planned_actual_native_steps=24000,
        shard_planned_forecast_native_steps=67200,
        same_current_course_comparators_only=True,
        horizon=300,
        latest_handoff=90,
        optimizer_steps=0,
        publicity_license_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)
    try:
        if args.preflight_only:
            for p, h in pins.items():
                if digest(Path(p)) != h:
                    raise ValueError("source/assets/evidence changed during bank admission")
            write_once(
                args.output / "complete.json",
                dict(
                    source_head=head,
                    bank_hash=bank_hash,
                    declaration_hash=digest(args.output / "declaration.json"),
                    actual_actor_allocations=0,
                    new_native_steps=0,
                    new_neural_inferences=0,
                    optimizer_steps=0,
                    physics_qualified=False,
                    promotion_authorized=False,
                    hardware_authorized=False,
                ),
            )
            print("ENTIRE16_CLOSE_CONTACT_BANK_ADMITTED_ZERO_ACTORS", bank_hash, flush=True)
        else:
            import numpy as np

            from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance

            stored = json.loads((args.prior / "entry-affordance.json").read_text())
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
                raise ValueError("past motion affordance changed")
            poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
            results = []
            for item in bank:
                if item.parent_index == args.cluster:
                    results.append(run_pair(args, item, poses, affordance, hash_json(declaration)))
            if len(results) != 4:
                raise ValueError("whole declared shard must complete")
            for p, h in pins.items():
                if digest(Path(p)) != h:
                    raise ValueError("source/assets/evidence changed during shard")
            write_once(
                args.output / "complete.json",
                dict(
                    results=results,
                    source_head=head,
                    bank_hash=bank_hash,
                    cluster=args.cluster,
                    actual_native_steps=24000,
                    forecast_native_steps=67200,
                    total_new_native_steps=91200,
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
