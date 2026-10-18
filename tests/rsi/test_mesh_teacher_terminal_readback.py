"""Full synthetic 18-receipt readback; NO physics or payload model qualification."""

import copy
import json
import sqlite3
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.rsi import first_touch_campaign_worker, reviewed_contact_world
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import planning_contexts
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts import rsi_read_mesh_contact_native_pilot as audit
from tests.rsi.test_mesh_contact_terminal_readback import fixture as original_fixture
from tests.rsi.test_mesh_contact_terminal_readback import write


def expanded_fixture(tmp_path, monkeypatch, tamper=None, *, receding=False, positive_lateral=False):
    original_fixture(tmp_path, monkeypatch, public_four=True)
    original = json.loads((tmp_path / "run/complete.json").read_text())
    root = tmp_path / "teacher"
    root.mkdir()
    output = tmp_path / "teacher-audit.json"
    world = root / "model.mjb"
    world.write_bytes((tmp_path / "run/mesh.mjb").read_bytes())
    payload = root / "shared.json"
    payload.write_bytes(b"synthetic payload, numerical reconstruction mocked")
    native_sources = {}
    for filename in (
        "native_first_touch_episode.py",
        "cpu_motor_transfer_evidence.py",
        "foundation_torch_capture_audit.py",
    ):
        file = tmp_path / "src/rosclaw_soccer/rsi" / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("synthetic source, no actual physics")
        native_sources[filename] = audit.digest(file)
    nominal_path = tmp_path / "nominal-complete.json"
    write(nominal_path, original)
    nominal_readback = write(
        tmp_path / "nominal-readback.json", dict(complete_hash=original["report_hash"])
    )
    candidate = dict(
        parameters=asdict(BoundedFirstTouchControl()), swing="bounded_contact_diagnostic"
    )
    fit = write(
        tmp_path / "fit.json",
        dict(options=[dict(candidate=None)] + [dict(candidate=candidate)] * 7),
    )
    selection_root = tmp_path / "selection"
    selection_root.mkdir()
    selection = write(selection_root / "declaration.json", dict(options=fit["options"]))
    selected = []
    for group in (8, 56):
        receipt = copy.deepcopy(next(r for r in original["results"] if r["group"] == group))
        receipt.update(job=f"option-4-raw-{group}", option_index=4, candidate=candidate)
        receipt["outcome"].update(task_pass=True, safe=True)
        selected.append(write(selection_root / f"selected-{group}.json", receipt, "receipt_hash"))
    upstream_complete = write(
        selection_root / "complete.json",
        dict(declaration_hash=selection["report_hash"], results=[{}] * 8 + selected + [{}] * 6),
    )
    reader_source = tmp_path / "upstream-reader.py"
    reader_source.write_text("synthetic reader, no actual prior verification")
    upstream = write(
        tmp_path / "upstream-readback.json",
        dict(
            schema="soccer.rsi.mesh_contact_catalog_terminal_readback.v1",
            complete_hash=upstream_complete["report_hash"],
            source_native_bodies=16,
            reader_source_hash=audit.digest(reader_source),
            whole_input_and_witness_files_verified=tamper != "upstream_unverified",
            shared_payload_logical_seals_rebuilt=True,
            prior_independent_replays_authenticated=True,
            rows=[
                dict(job=r["job"], receipt_hash=r["receipt_hash"], outcome=r["outcome"])
                for r in selected
            ],
        ),
    )
    if tamper == "upstream_complete":
        upstream["complete_hash"] = hash_json("wrong complete")
        write(tmp_path / "upstream-readback.json", upstream)
    sampling_view = tmp_path / "source-view.json"
    sampling_view.write_text("synthetic source view, no numeric model")
    manifest = dict(
        records=[dict(witnesses=dict(sampling_view=dict(path=str(sampling_view))))] * 160
    )
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), "sha256:" + "a" * 64, 18)
    contexts = planning_contexts(plan)
    if receding:
        from rosclaw_soccer.rsi.receding_teacher_scope import receding_contexts, scope_contract

        contexts = receding_contexts(plan)
    positive_scope = None
    if positive_lateral:
        from rosclaw_soccer.rsi.positive_lateral_episode_inputs import prospective_split
        from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
        from rosclaw_soccer.rsi.public_physics_option_planning import PublicPlanningContext

        positive_scope = build_scope(plan.behavior_model_hash)
        plan = PublicFirstTouchTrainingPlan(
            tuple(positive_scope["source_plan"]["seeds"]), plan.behavior_model_hash, 19
        )
        contexts = tuple(
            PublicPlanningContext(
                c["context_index"],
                c["source_group"],
                c["source_context_id"],
                c["source_seed"],
                c["source_lane"],
                c["sampling_seed"],
                tuple(c["course"]),
                c["perturbation_index"],
            )
            for c in positive_scope["contexts"]
        )
    profile = (
        "positive-lateral-sixteen"
        if positive_lateral
        else "receding-catalog2-ten"
        if receding
        else "teacher-eighteen"
    )
    spec = audit.bounded_profile(profile, plan.behavior_model_hash)
    config = dict(
        source_root=str(tmp_path),
        core_root=str(tmp_path),
        output_root=str(root),
        groups=[4, 8, 32, 56],
        max_attempts=18,
        new_physical_budget=18,
        mesh_task_contact_world_hash=audit.digest(world),
        nominal_complete=str(nominal_path),
        nominal_readback=str(tmp_path / "nominal-readback.json"),
        nominal_readback_hash=nominal_readback["report_hash"],
        nominal_readback_file_hash=audit.digest(tmp_path / "nominal-readback.json"),
        portfolio_fit=str(tmp_path / "fit.json"),
        portfolio_fit_hash=fit["report_hash"],
        upstream_terminal_audit_required=True,
        selected_option_index=4,
        upstream_catalog_readback=str(tmp_path / "upstream-readback.json"),
        selection_declaration=str(selection_root / "declaration.json"),
        selection_declaration_hash=selection["report_hash"],
        selection_receipts=[
            dict(
                path=str(selection_root / f"selected-{g}.json"),
                file_hash=audit.digest(selection_root / f"selected-{g}.json"),
                receipt_hash=r["receipt_hash"],
            )
            for g, r in zip((8, 56), selected, strict=True)
        ],
    )
    config_path = tmp_path / "teacher-config.json"
    if receding:
        config.update(
            groups=spec["groups"],
            max_attempts=10,
            new_physical_budget=10,
            selected_option_index=2,
            neural_training_authorized=False,
            teacher_auto_admission=False,
        )
    if positive_lateral:
        from rosclaw_soccer.rsi import positive_lateral_episode_inputs
        from scripts import rsi_positive_lateral_physical_pilot as positive_pilot

        config.update(
            study_profile=profile,
            behavior_model_hash=plan.behavior_model_hash,
            groups=spec["groups"],
            max_attempts=16,
            new_physical_budget=16,
            new_independent_cpu_budget=16,
            combined_body_substep_budget=96000,
            selected_option_index=2,
            mode="SIM_ONLY",
            generated_seed_episode_binding=True,
            hardware_authorized=False,
            promotion_authorized=False,
            private_fresh_accessed=False,
            neural_training_authorized=False,
            teacher_auto_admission=False,
            use_mesh_surface_swing=False,
            use_contact_exit_torque_mpc=False,
            use_directional_contact_compliance=False,
            suppress_postcontact_swing_release=False,
            required_receipt_witnesses=["intent_witness", "encoder_witness"],
        )
        registry_path = tmp_path / "positive-registry.json"
        registry = write(
            registry_path, dict(scope=positive_scope, scope_hash=positive_scope["scope_hash"])
        )
        config.update(
            sampling_registry=str(registry_path),
            sampling_registry_file_hash=audit.digest(registry_path),
            sampling_registry_hash=registry["report_hash"],
        )

        # Separate registry/provenance tests cover these real contracts. This
        # full-reader fixture mocks all numerical and old-source evidence.
        def positive_inputs(*args, **kwargs):
            if tamper == "positive_registry":
                raise ValueError("synthetic missing registry authentication")
            return plan, manifest, contexts

        monkeypatch.setattr(positive_lateral_episode_inputs, "episode_inputs", positive_inputs)
        monkeypatch.setattr(
            positive_pilot,
            "whole_predecessor",
            lambda *a: (
                dict(report_hash="previous"),
                dict(report_hash="diagnostic", research_decision="retained"),
            ),
        )
    write(config_path, config, key=None)
    reused = [
        dict(role="control", pair_id=f"raw-{r['group']}", receipt=r) for r in original["results"]
    ]
    reused += [
        dict(role="teacher", pair_id=f"raw-{g}", receipt=r)
        for g, r in zip((8, 56), selected, strict=True)
    ]
    archived = None
    if receding:
        from rosclaw_soccer.rsi import receding_scope_evidence

        reused = [
            dict(role="control", pair_id=f"receding-g{g}", receipt=original["results"][0])
            for g in (76, 77)
        ]
        archived = dict(
            receipt_hash=hash_json("synthetic archived expert"),
            candidate=candidate,
            outcome=original["results"][0]["outcome"],
            witnesses={},
        )
        # Authentication and numerical evidence are mocked in this fixture;
        # real archived-source authentication is tested separately and executed.
        monkeypatch.setattr(
            receding_scope_evidence, "load_scope_evidence", lambda *a: (reused, archived)
        )
    if tamper == "reuse_duplicate":
        reused[-1] = reused[-2]
    pins = {str(config_path): audit.digest(config_path)}
    binding = dict(synthetic=True)
    declaration = write(
        root / "declaration.json",
        dict(
            config,
            schema=spec["declaration_schema"],
            jobs=spec["jobs"],
            contexts=[dict(context_hash=c.context_hash, course=list(c.course)) for c in contexts],
            options=[r["candidate"] for r in fit["options"]],
            reused_endpoints=reused,
            input_file_hashes=pins,
            prepared_pins=pins,
            source_heads={},
            numerical_world_review=binding,
            **(
                dict(
                    scope_contract=scope_contract(plan),
                    archived_reference_receipt_hash=archived["receipt_hash"],
                )
                if receding
                else {}
            ),
            **(
                dict(
                    source_plan=plan.contract(),
                    distinct_derived_course_count=8,
                    source_scene_clusters=4,
                    scope_hash=positive_scope["scope_hash"],
                    course_cluster_contract=positive_scope["cluster_accounting"],
                    prospective_split=prospective_split(positive_scope),
                    preceding_complete_hash="previous",
                    preceding_diagnostic_hash="diagnostic",
                    preceding_research_decision="retained",
                )
                if positive_lateral
                else {}
            ),
        ),
    )
    if positive_lateral and tamper in (
        "positive_split",
        "positive_source_plan",
        "positive_previous",
    ):
        if tamper == "positive_split":
            declaration["prospective_split"]["train_source_seeds"].reverse()
        elif tamper == "positive_source_plan":
            declaration["source_plan"]["stream"] = 18
        else:
            declaration["preceding_research_decision"] = "invented"
        declaration = write(root / "declaration.json", declaration)
    monkeypatch.setattr(
        first_touch_campaign_worker,
        "prepare",
        lambda p: (
            SimpleNamespace(PINS=pins, HEADS={}, stable=lambda: None),
            SimpleNamespace(MEAN=plan.behavior_model_hash) if positive_lateral else None,
            manifest,
            config,
        ),
    )
    monkeypatch.setattr(
        reviewed_contact_world,
        "ReviewedContactWorld",
        lambda *a: SimpleNamespace(verify=lambda h: binding),
    )
    receipts, events = [], []
    state = dict(active=None, attempts=0, completed={}, declaration_hash=hash_json(declaration))
    for index, job in enumerate(spec["jobs"]):
        folder = root / str(index)
        folder.mkdir()
        draw = plan.draws()[job["group"]]
        context = (
            next(c for c in contexts if c.source_group == job["group"])
            if receding
            else None
            if job["context_index"] is None
            else contexts[job["context_index"]]
        )
        course = draw.course if context is None else context.course
        trace = dict(
            force_n=np.zeros((300, 1, 6)),
            pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
            ball_position_after_step_m=np.zeros((300, 1, 3)),
            canonical_qpos=np.zeros((300, 1, 43)),
            canonical_qvel=np.zeros((300, 1, 41)),
            foundation_neural_encoder_features=np.zeros((300, 1, 640), dtype=np.float32),
        )
        q = trace["canonical_qpos"][0, 0]
        q[:7] = (0, 0, 0.793, 1, 0, 0, 0)
        q[7:36] = G1SonicRunupController.default_angles
        q[36:43] = (*course[:2], 0.13, 1, 0, 0, 0)
        trace["canonical_qvel"][0, 0, 35:41] = (course[2], 0, 0, 0, course[2] / 0.11, 0)
        reproduction = None
        if receding and index == 0:
            from rosclaw_soccer.rsi.receding_teacher_scope import reproduce_zero_control

            old_trace = dict(trace, cerebellar_sampling_mean=np.zeros((300, 1, 10, 29)))
            old_path = tmp_path / "archived-trace.npz"
            np.savez(old_path, **old_trace)
            archived["witnesses"]["trace"] = dict(path=str(old_path), sha256=audit.digest(old_path))
            reproduction = reproduce_zero_control(trace, old_trace)
            reproduction["archived_receipt_hash"] = archived["receipt_hash"]
            if tamper == "reproduction":
                trace["canonical_qpos"] = trace["canonical_qpos"].copy()
                trace["canonical_qpos"][299, 0, 10] += 1e-12
        np.savez(folder / "trace.npz", **trace)
        outcome = label_first_touch(trace)
        option = None if job["option_index"] == 0 else candidate
        commitment = dict(
            compiled_model_hash=config["mesh_task_contact_world_hash"],
            mesh_task_contact_protocol=dict(
                expected_world_hash=config["mesh_task_contact_world_hash"]
            ),
            approach_mode="legacy" if option is None else "bounded_approach_diagnostic",
            swing_mode="legacy" if option is None else candidate["swing"],
            executed_motor_policy=dict(
                step_motor_proof=dict(
                    model=dict(
                        mean_model=dict(model_hash=plan.behavior_model_hash),
                        seed=draw.sampling_seed,
                    )
                )
            ),
            default_locomotion_planner_events=[
                dict(frame=i, planner_mode=2, reference_hash=hash_json(i))
                for i in range(0, 300, 20)
            ],
        )
        if option is not None:
            commitment["bounded_control_profile"] = BoundedFirstTouchControl(
                **candidate["parameters"]
            ).contract()
        if positive_lateral:
            commitment["generated_seed_episode_binding"] = dict(
                schema="soccer.rsi.native_private_generated_episode_binding.v1",
                seed=draw.sampling_seed,
                full_external_sampling_view_restored_and_validated=True,
                complete_generated_model_equals_external_view=True,
                full_original_factory_constructor_retained=True,
                sampling_law_changed=False,
            )
            if index == 0 and tamper == "positive_binding":
                commitment["generated_seed_episode_binding"]["seed"] += 1
            if index == 0 and tamper == "positive_mixed_intervention":
                commitment["mesh_surface_swing"] = {}
        write(folder / "commitment.json", commitment, key=None)
        report = write(
            folder / "report.json",
            dict(
                commitment,
                commitment_hash=hash_json(commitment),
                course=list(course),
                seed=draw.seed,
                lane=draw.lane,
                physical_trace_hash=audit.digest(folder / "trace.npz"),
                source_hash=native_sources["native_first_touch_episode.py"],
            ),
        )
        review = write(
            folder / "independent.json",
            dict(
                physical_substeps=3000,
                actual_recurrent_sampling_draws_reconstructed=270,
                high_quality=False,
                safety_passed=True,
                clean_foot_only=False,
                minimum_pelvis_z_m=0.7,
                reviewed_report_hash=report["report_hash"],
                source_hash=native_sources["cpu_motor_transfer_evidence.py"],
            ),
        )
        foundation = write(
            folder / "foundation.json",
            dict(
                recomputed_executed_calls=300,
                physical_report_hash=report["report_hash"],
                physical_trace_hash=report["physical_trace_hash"],
                audit_source_hash=native_sources["foundation_torch_capture_audit.py"],
            ),
        )
        intent = write(
            folder / "intent.json",
            dict(
                job,
                candidate=option,
                context_hash=None if context is None else context.context_hash,
                declaration_hash=declaration["report_hash"],
                **(
                    dict(
                        source_plan_hash=plan.contract()["plan_hash"],
                        prospective_split_hash=declaration["prospective_split"]["split_hash"],
                    )
                    if positive_lateral
                    else {}
                ),
            ),
        )
        write(
            folder / "encoder.json",
            dict(
                schema="soccer.rsi.sonic_encoder_reference_consumption.v1",
                actual_cpu_onnx_planner_calls=15,
                compared_encoder_frames=300,
                compared_features_per_frame=640,
                recorded_features_hash=hash_bytes(
                    trace["foundation_neural_encoder_features"].tobytes()
                ),
                recorded_features_dtype="float32",
                maximum_absolute_error=0.0,
                planner_events=commitment["default_locomotion_planner_events"],
            ),
        )

        def witness(file):
            return dict(path=str(file), sha256=audit.digest(file))

        receipt = dict(
            job=job["job"],
            role=job["role"],
            pair_id=job["pair_id"],
            context_index=job["context_index"],
            option_index=job["option_index"],
            group=job["group"],
            candidate=option,
            physical_substeps=3000,
            foundation_calls=300,
            outcome=outcome.to_dict(),
            qualification_alignment=require_first_touch_qualification(review, foundation, outcome),
            physical_report_hash=report["report_hash"],
            witnesses={
                k: witness(f)
                for k, f in dict(
                    trace=folder / "trace.npz",
                    report=folder / "report.json",
                    commitment=folder / "commitment.json",
                    independent_review=folder / "independent.json",
                    foundation_review=folder / "foundation.json",
                    proof_payload=payload,
                    world=world,
                ).items()
            },
            intent_witness=witness(folder / "intent.json"),
            encoder_witness=witness(folder / "encoder.json"),
        )
        if context is not None:
            document = dict(
                schema="soccer.rsi.public_planning_execution_context.v1",
                context=asdict(context),
                context_hash=context.context_hash,
                source_plan_hash=plan.contract()["plan_hash"],
                stage="SELECTED_OPTION_EVALUATION",
                source_sampling_view_hash=audit.digest(sampling_view),
                candidate_hash=hash_json(option),
                outcome_contract_changed=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
            if index == 0 and tamper == "planning_stage":
                document["stage"] = "PLANNING_SIMULATION"
            if index == 0 and tamper == "view_identity":
                document["source_sampling_view_hash"] = hash_json("wrong")
            write(folder / "context.json", document)
            receipt.update(
                planning_context_witness=witness(folder / "context.json"),
                planning_context_hash=context.context_hash,
            )
        if index == 0 and tamper == "receipt_role":
            receipt["role"] = "control" if receding else "teacher"
        if positive_lateral:
            receipt["context_hash"] = job["context_hash"]
        if reproduction is not None:
            p = root / "archived-c7-reproduction.json"
            write(p, reproduction)
            receipt["reproduction_witness"] = witness(p)
        receipt = write(root / f"receipt-{index:04d}.json", receipt, "receipt_hash")
        receipts.append(receipt)
        state.update(next_intent_hash=intent["report_hash"], active=None)
        events.append(dict(event="INTENT_COMMITTED_BEFORE_BODY", state=copy.deepcopy(state)))
        state.update(active=dict(job=job["job"], attempt=index), attempts=index + 1)
        events.append(dict(event="ATTEMPT_STARTED", state=copy.deepcopy(state)))
        state["completed"][job["job"]] = receipt
        state["active"] = None
    state["phase"] = spec["phase"]
    if tamper == "ledger_declaration":
        state["declaration_hash"] = hash_json("wrong")
    events.append(dict(event=spec["phase"], state=copy.deepcopy(state)))
    with sqlite3.connect(root / "campaign.sqlite") as db:
        db.execute(
            "CREATE TABLE events(seq INTEGER PRIMARY KEY, previous TEXT, body TEXT, digest TEXT)"
        )
        previous = "GENESIS"
        for seq, body in enumerate(events, 1):
            value = hash_json(dict(seq=seq, previous=previous, body=body))
            db.execute(
                "INSERT INTO events VALUES(?,?,?,?)", (seq, previous, json.dumps(body), value)
            )
            previous = value
    controls = (
        {r["pair_id"]: r["receipt"]["outcome"] for r in reused}
        if receding
        else {}
        if positive_lateral
        else {f"raw-{r['group']}": r["outcome"] for r in original["results"]}
    )
    evaluations = (
        {}
        if receding or positive_lateral
        else {f"raw-{g}": r["outcome"] for g, r in zip((8, 56), selected, strict=True)}
    )
    for job, receipt in zip(spec["jobs"], receipts, strict=True):
        (controls if job["role"] == "control" else evaluations)[job["pair_id"]] = receipt["outcome"]
    paired = assess_pairs(controls, evaluations)
    if tamper == "pairing":
        paired["evaluated_passes"] = 999
    write(
        root / "complete.json",
        dict(
            schema=spec["complete_schema"],
            declaration_hash=declaration["report_hash"],
            new_native_bodies=spec["count"],
            native_substeps=3000 * spec["count"],
            independent_cpu_substeps=3000 * spec["count"],
            new_optimizer_updates=0,
            independent_course_count=6 if receding else 12,
            paired_endpoint_count=12 if receding else 24,
            results=receipts,
            reused_endpoints=reused,
            paired_development=paired,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            **(dict(archived_c7_full_numeric_reproduction=True) if receding else {}),
            **(
                dict(
                    new_independent_cpu_bodies=16,
                    distinct_derived_course_count=8,
                    source_scene_clusters=4,
                    old_global_failures_retained_not_relabelled=True,
                    source_plan_hash=plan.contract()["plan_hash"],
                    prospective_split=declaration["prospective_split"],
                    dataset_admission_authorized=False,
                    neural_training_authorized=False,
                    fresh_exam=False,
                )
                if positive_lateral
                else {}
            ),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--config",
            str(config_path),
            "--unit",
            "fake.service",
            "--output",
            str(output),
            "--profile",
            profile,
            "--upstream-audit-source",
            str(reader_source),
        ],
    )
    return output


def test_full_synthetic_teacher_profile_keeps_failures_and_no_admission(tmp_path, monkeypatch):
    output = expanded_fixture(tmp_path, monkeypatch)
    audit.main()
    report = json.loads(output.read_text())
    assert len(report["rows"]) == 18 and report["reused_endpoint_count"] == 6
    assert report["paired_development"]["pairs"] == 12
    assert report["teacher_qualified"] is False
    assert report["original_prepared_pins_persisted_by_producer"] is True
    assert report["new_physical_replays"] == 0


@pytest.mark.parametrize(
    "tamper",
    [
        "planning_stage",
        "view_identity",
        "receipt_role",
        "upstream_unverified",
        "upstream_complete",
        "reuse_duplicate",
        "ledger_declaration",
        "pairing",
    ],
)
def test_teacher_profile_rejects_resealed_binding_failures(tmp_path, monkeypatch, tamper):
    output = expanded_fixture(tmp_path, monkeypatch, tamper)
    with pytest.raises(ValueError):
        audit.main()
    assert not output.exists()
