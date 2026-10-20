"""Fixed three-arm actualH300 ablation; mandatory full-bank zero-actor preflight."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

ARMS = ("baseline", "blend_only", "measured_blend")


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def run_arm(args, item, arm, gold, poses, affordance, evidence_hash):
    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.full_entry_forecast import (
        feedback_entry_laws,
        nominal_prefix20,
        run_entry_suffix,
        score_full_entry,
    )
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_probe_measured_planner_context import RecordingPlanner

    folder = args.output / f"variant{item.variant_index}-{arm}"
    folder.mkdir(exist_ok=False)
    body = nav = proxy = None
    rows = []

    def persist_trace():
        if rows:
            np.savez_compressed(
                folder / "observed-control-ticks.npz",
                **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]},
            )
        if nav is not None:
            write_once(folder / "planning-events.json", dict(events=nav.backend.events))
        if proxy is not None:
            calls = {}
            for index, feed in enumerate(proxy.feeds):
                calls.update({f"call{index}__feed__{k}": a for k, a in feed.items()})
                if index < len(proxy.outputs):
                    calls[f"call{index}__raw_output"] = proxy.outputs[index][0]
                    calls[f"call{index}__raw_count"] = proxy.outputs[index][1]
            np.savez_compressed(folder / "planner-calls.npz", **calls)
        if body is not None:
            np.savez_compressed(
                folder / "last-native-state.npz",
                qpos=body.data.qpos.copy(),
                qvel=body.data.qvel.copy(),
                time_sec=np.array([body.data.time]),
            )

    try:
        model = mujoco.MjModel.from_binary_path(str(args.world))
        limits = tuple(
            tuple(float(x) for x in model.jnt_range[model.joint(n).id]) for n in G1_DDS_JOINT_NAMES
        )
        measured = arm == "measured_blend"
        config = SonicNavigationConfig(
            maximum_frames=300,
            model_variant="low_latency",
            inference_threads=1,
            record_default_planner_events=True,
            experimental_reference_blend=arm != "baseline",
            experimental_measured_joint_context=measured,
            experimental_context_joint_limits=limits if measured else None,
        )
        nav = G1SonicNavigation(args.model_root, item.agent_id, config)
        original_session, original_method = nav.backend._planner, nav.backend._planner.run
        proxy = RecordingPlanner(original_session)
        nav.backend._planner = proxy  # Owned branch attribute, not shared-session mutation.
        body = G1TargetEpisode(
            model,
            world_hash=digest(args.world),
            kp=nav.backend.kp.copy(),
            kd=nav.backend.kd.copy(),
            course=item.course,
            defaults=G1SonicRunupController.default_angles.copy(),
            record_all_ball_contacts=True,
        )
        prefix = nominal_prefix20(nav, body, affordance, step_observer=rows.append)
        for k in prefix:
            np.testing.assert_array_equal(prefix[k], gold[k][:20])
        suffix, report = run_entry_suffix(
            nav,
            body,
            model_root=args.model_root,
            source_poses=poses,
            source_evidence_hash=evidence_hash,
            affordance=affordance,
            law=feedback_entry_laws()[0],
            contact_reference_offset_m=(0.0, 0.0, 0.0),
            step_observer=rows.append,
        )
        arrays = {k: np.concatenate([prefix[k], suffix[k]]) for k in prefix}
        observed = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        if len(rows) != 300 or body.attempted_steps != 3000 or body.integrated_steps != 3000:
            raise ValueError("complete actual H300 and300recorded control ticks required")
        for k, a in arrays.items():
            np.testing.assert_array_equal(a, observed[k])
            if arm == "baseline":
                np.testing.assert_array_equal(a, gold[k])
        if original_session.run != original_method or proxy.session is not original_session:
            raise ValueError("shared planner session or method was changed")
        if len(proxy.feeds) != len(proxy.outputs) or len(proxy.feeds) != nav.backend.planner_calls:
            raise ValueError("each actual source planning call must be captured")
        if len(nav.backend.events) != len(proxy.feeds):
            raise ValueError("every planner event must correspond to an actual model call")
        for index, event in enumerate(nav.backend.events):
            if event["frame"] != 20 * index:
                raise ValueError("fixed original planning cadence differs")
            if arm != "baseline":
                for k, a in proxy.feeds[index].items():
                    np.testing.assert_array_equal(
                        a,
                        np.asarray(
                            event["planner_inputs"][k], dtype=event["planner_input_dtypes"][k]
                        ),
                    )
                np.testing.assert_array_equal(
                    proxy.outputs[index][0],
                    np.asarray(event["planner_raw_output"], dtype=np.float32),
                )
                if index and "reference_stitch" not in event:
                    raise ValueError(
                        "each experimental replan requires owned complete stitch evidence"
                    )
        persist_trace()
        artifact = folder / "actual.npz"
        np.savez_compressed(artifact, **arrays)
        report.update(
            arm=arm,
            course=asdict(item),
            navigation_config=asdict(config),
            score=score_full_entry(model, suffix),
            strike=summarize_reference_ball_strike(model, arrays),
            artifact_hash=digest(artifact),
            planning_events_hash=digest(folder / "planning-events.json"),
            planner_calls_hash=digest(folder / "planner-calls.npz"),
            observed_control_ticks_hash=digest(folder / "observed-control-ticks.npz"),
            actual_attempted_native_steps=body.attempted_steps,
            actual_integrated_native_steps=body.integrated_steps,
            source_planner_calls=len(proxy.feeds),
            prefix20_all18_exact_to_own_archived_nominal=True,
            baseline_all18_exact_to_own_archived_nominal=arm == "baseline",
            direct_future_or_cross_arm_targets_adopted=False,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        write_once(folder / "complete.json", report)
        print(
            "MEASURED_CONTEXT_NATIVE_ACTUAL",
            item.parent_index,
            item.variant_index,
            arm,
            report["score"],
            flush=True,
        )
        return report
    except Exception as error:
        # Preserve completed control rows, pending model calls, last actual native state.
        persistence_error = None
        try:
            persist_trace()
        except Exception as trace_error:
            persistence_error = repr(trace_error)
        write_once(
            folder / "failure.json",
            dict(
                error=repr(error),
                trace_persistence_error=persistence_error,
                completed_control_ticks=len(rows),
                attempted_native_steps=body.attempted_steps if body is not None else 0,
                integrated_native_steps=body.integrated_steps if body is not None else 0,
                attempted_source_planner_calls=len(proxy.feeds) if proxy is not None else 0,
                returned_source_planner_calls=len(proxy.outputs) if proxy is not None else 0,
                physical_failure_not_retried=True,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "audit",
        "audit-review",
        "world",
        "motion-csv",
        "model-root",
        "core-source",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-producer", type=Path, action="append", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-audit-review-hash", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cluster", type=int, choices=range(4))
    parser.add_argument("--bank-preflight", type=Path)
    parser.add_argument("--expected-bank-preflight-hash")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists() or len(args.previous_producer) != 4:
        raise ValueError("sealed source, unused output and all previous clusters required")
    if args.preflight_only:
        if any(
            x is not None
            for x in (args.cluster, args.bank_preflight, args.expected_bank_preflight_hash)
        ):
            raise ValueError("whole-bank zero-actor stage cannot allocate a worker")
    elif any(
        x is None for x in (args.cluster, args.bank_preflight, args.expected_bank_preflight_hash)
    ):
        raise ValueError("worker requires an explicit cluster and sealed whole-bank admission")
    if digest(args.audit_review / "complete.json") != args.expected_audit_review_hash:
        raise ValueError("independent ONNX/native audit proof identity differs")
    ad = json.loads((args.audit / "declaration.json").read_text())
    ac = json.loads((args.audit / "complete.json").read_text())
    rd = json.loads((args.audit_review / "declaration.json").read_text())
    rc = json.loads((args.audit_review / "complete.json").read_text())
    if (
        ad["schema"] != "soccer.rsi.measured_joint_planner_input16.v1"
        or ac["actual_native_steps"] != 3200
        or ac["additional_real_planner_calls"] != 64
        or ac["model_sensitive_conditions"] != 16
        or ac["rejected_context_conditions"] != 0
        or rc["additional_recorded_pd_native_steps"] != 3200
        or rc["additional_real_planner_calls"] != 80
        or len(rc["results"]) != 16
        or not all(
            r["recorded_prefix_native_bit_exact"]
            and r["initial_and_all_recorded_planner_outputs_bit_exact"]
            and r["original_context_reconstructed_from_initial_model_output"]
            and r["measured_context_and_all_other_feeds_reconstructed"]
            for r in rc["results"]
        )
        or rd["input_file_hashes"].get(str(args.audit / "complete.json"))
        != digest(args.audit / "complete.json")
        or any(any(root.rglob("failure.json")) for root in (args.audit, args.audit_review))
    ):
        raise ValueError("whole successful16 input-sensitivity and independent proof required")
    historical = {}
    for d in (ad, rd):
        for p, h in d["input_file_hashes"].items():
            if digest(Path(p)) != h:
                raise ValueError("preceding source/assets/evidence changed")
            historical[p] = h
    for p in [args.world, args.motion_csv] + [
        args.model_root / name
        for name in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]:
        if historical.get(str(p)) != digest(p):
            raise ValueError("same physical world, motion source and model assets required")
    previous = {}
    for root in args.previous_producer:
        p = root / "complete.json"
        if historical.get(str(p)) != digest(p):
            raise ValueError("exact previous paired controls required")
        record = json.loads(p.read_text())
        if record["cluster"] in previous or record["total_new_native_steps"] != 91200:
            raise ValueError("unique complete prior cluster required")
        previous[record["cluster"]] = (root, record)
    if set(previous) != set(range(4)):
        raise ValueError("one prior cluster per parent required")
    paths = [
        p
        for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [p for root in (args.audit, args.audit_review) for p in root.rglob("*") if p.is_file()]
    pins = {**historical, **{str(p): digest(p) for p in paths}}
    bank_record = None
    if not args.preflight_only:
        if digest(args.bank_preflight / "complete.json") != args.expected_bank_preflight_hash:
            raise ValueError("whole-bank master identity differs")
        bank_record = json.loads((args.bank_preflight / "complete.json").read_text())
        bank_declaration = json.loads((args.bank_preflight / "declaration.json").read_text())
        if (
            bank_record["source_head"] != head
            or bank_record["actual_actor_allocations"] != 0
            or bank_record["new_native_steps"] != 0
            or bank_record["new_model_calls"] != 0
            or bank_declaration["input_file_hashes"] != pins
            or bank_record["declaration_hash"] != digest(args.bank_preflight / "declaration.json")
            or (args.bank_preflight / "failure.json").exists()
        ):
            raise ValueError("authentic same-source zero-actor bank receipt required")
        pins.update({str(p): digest(p) for p in args.bank_preflight.iterdir() if p.is_file()})

    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.close_contact_courses import close_contact_perturbations
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once

    bank = close_contact_perturbations()
    bank_hash = preflight_receiving_courses(bank)
    if bank_hash != ad["bank_hash"] or (
        bank_record is not None and bank_hash != bank_record["bank_hash"]
    ):
        raise ValueError("whole admitted public16 diagnostic bank identity differs")
    for item in bank:
        _, record = previous[item.parent_index]
        if record["results"][item.variant_index]["course"] != asdict(item) | {
            "course": list(item.course)
        }:
            raise ValueError("every paired own-state comparator must match exact declared course")
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.measured_context_native_three_arm16.v1",
        source_head=head,
        input_file_hashes=pins,
        bank_hash=bank_hash,
        courses=[asdict(x) for x in bank],
        arms=list(ARMS),
        cluster=args.cluster,
        stage="zero_actor_bank_preflight" if args.preflight_only else "cluster_physics",
        total_planned_native_steps=144000,
        shard_planned_native_steps=36000,
        fixed_feedback_index=0,
        contact_reference_offset_m=[0.0, 0.0, 0.0],
        reference_blend_frames=10,
        measured_joint_context_maximum_rad=0.12,
        horizon=300,
        latest_handoff=90,
        original_m0_or_private_fresh=False,
        independent_parent_clusters=4,
        optimizer_steps=0,
        publicity_license_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)
    results = []
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
                    new_model_calls=0,
                    physics_qualified=False,
                    promotion_authorized=False,
                    hardware_authorized=False,
                ),
            )
            print("NATIVE_THREE_ARM_BANK16_ADMITTED_ZERO_ACTORS", bank_hash, flush=True)
            return
        import numpy as np

        from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance

        choices = [Path(p) for p in historical if Path(p).name == "entry-affordance.json"]
        if len(choices) != 1:
            raise ValueError("one authenticated historical affordance required")
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
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        for item in bank:
            if item.parent_index != args.cluster:
                continue
            root, record = previous[item.parent_index]
            nominal = record["results"][item.variant_index]["results"][0]
            with np.load(
                root / f"variant{item.variant_index}" / nominal["artifact"], allow_pickle=False
            ) as archive:
                gold = {k: archive[k] for k in archive.files}
            for arm in ARMS:
                results.append(
                    run_arm(args, item, arm, gold, poses, affordance, hash_json(declaration))
                )
        if len(results) != 12 or sum(r["actual_integrated_native_steps"] for r in results) != 36000:
            raise ValueError("complete declared12 actual H300 per cluster required")
        for p, h in pins.items():
            if digest(Path(p)) != h:
                raise ValueError("pinned source/assets/evidence changed during native ablation")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                source_head=head,
                bank_hash=bank_hash,
                cluster=args.cluster,
                actual_native_steps=36000,
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
                completed_h300_cases=len(results),
                completed_h300_native_steps=sum(
                    r["actual_integrated_native_steps"] for r in results
                ),
                incomplete_case_counts_in_individual_failure=True,
                missing_cases_are_not_verified=True,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
