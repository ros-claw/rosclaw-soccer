"""Whole own-body orchestration with ALL physics/NN/readback engines mocked.

These are not thirteen physical trials or a fitted G1 performance result.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import PublicPlanningContext
from rosclaw_soccer.rsi.reference_student_exam_scope import exam_scope
from rosclaw_soccer.rsi.reference_student_execution import FIELDS, PREFIX
from rosclaw_soccer.rsi.reference_student_probe import probe_models
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_read_reference_student_exam as audit
from scripts import rsi_reference_student_exam as pilot


def config():
    return dict(
        exam_profile="reference-own-body-thirteen-v1",
        mode="SIM_ONLY",
        max_attempts=13,
        max_generations=1,
        plateau_generations=1,
        new_physical_budget=13,
        new_independent_cpu_budget=13,
        combined_body_substep_budget=78000,
        new_optimizer_updates=0,
        deadline_unix=10**20,
        required_receipt_witnesses=["intent_witness", "encoder_witness", "prefix_witness"],
        hardware_authorized=False,
        promotion_authorized=False,
        private_fresh_accessed=False,
        neural_training_authorized=False,
        teacher_auto_admission=False,
        use_mesh_surface_swing=False,
        suppress_postcontact_swing_release=False,
        use_contact_exit_torque_mpc=False,
        use_directional_contact_compliance=False,
    )


@pytest.mark.parametrize(
    "changed", [None, "budget", "cpu", "training", "hardware", "witness", "bool"]
)
def test_own_body_protocol_cannot_widen_or_disable_independent_replay(changed):
    cfg = config()
    if changed == "budget":
        cfg["max_attempts"] = 14
    elif changed == "cpu":
        cfg["new_independent_cpu_budget"] = 0
    elif changed == "training":
        cfg["neural_training_authorized"] = True
    elif changed == "hardware":
        cfg["hardware_authorized"] = True
    elif changed == "witness":
        cfg["required_receipt_witnesses"].pop()
    elif changed == "bool":
        cfg["new_optimizer_updates"] = False
    if changed:
        with pytest.raises(ValueError):
            pilot.require_scope(cfg)
    else:
        pilot.require_scope(cfg)


def reference_review(model_hash):
    return dict(
        schema="soccer.rsi.pre_parent_reference_student_reconstruction.v1",
        model_hash=model_hash,
        frames_reconstructed=300,
        own_hidden_and_applied_reference_reconstructed=True,
        raw_foundation_navigation_targets_bit_exact=True,
        reconstruction_uses_same_pinned_decoder=True,
        parent_target_composition_reconstructed_separately=True,
        independent_neural_implementation=False,
        training_provenance_authenticated=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def fixture(tmp_path, monkeypatch, mode):
    source = Path(pilot.__file__).resolve().parents[1]
    parent = "sha256:" + "a" * 64
    world = tmp_path / "world.mjb"
    world.write_bytes(b"synthetic world, not an actual MjModel")
    world_hash = pilot.digest(world)
    payload = tmp_path / "original-payload.json"
    payload.write_bytes(b"synthetic payload, not actual mean weights")
    zero, student = probe_models(world_hash, parent)
    scope = exam_scope(parent)
    positive = build_scope(parent)
    ctx = {
        r["source_group"]: PublicPlanningContext(
            r["context_index"],
            r["source_group"],
            r["source_context_id"],
            r["source_seed"],
            r["source_lane"],
            r["sampling_seed"],
            tuple(r["course"]),
            r["perturbation_index"],
        )
        for r in positive["contexts"]
    }
    pos_plan = PublicFirstTouchTrainingPlan(tuple(positive["source_plan"]["seeds"]), parent, 19)
    old_plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), parent, 18)

    def trace(course):
        t = {f"field{i}": np.zeros((300, 1, 1)) for i in range(22)}
        t.update(
            {
                k: np.zeros((300, 1, 29))
                for k in (
                    "pre_motor_joint_target_rad",
                    "joint_target_rad",
                    "joint_position_rad",
                    "joint_velocity_rad_s",
                )
            }
        )
        t.update(
            canonical_qpos=np.zeros((300, 1, 43)),
            canonical_qvel=np.zeros((300, 1, 41)),
            root_pose_xyzw_m=np.zeros((300, 1, 7)),
            root_velocity_world=np.zeros((300, 1, 6)),
            force_n=np.zeros((300, 1, 6)),
            ball_position_after_step_m=np.zeros((300, 1, 3)),
            pelvis_z_per_substep_m=np.full((300, 1, 10), 0.8),
        )
        assert len(t) == 33
        t["canonical_qpos"][0, 0, :7] = (0, 0, 0.793, 1, 0, 0, 0)
        t["canonical_qpos"][0, 0, 7:36] = G1SonicRunupController.default_angles
        t["canonical_qpos"][0, 0, 36:43] = (*course[:2], 0.13, 1, 0, 0, 0)
        t["canonical_qvel"][0, 0, 35:41] = (course[2], 0, 0, 0, course[2] / 0.11, 0)
        t["root_pose_xyzw_m"][:, :, 6] = 1
        t["force_n"][200, 0, 0] = 2
        t["ball_position_after_step_m"][200:261, 0, 0] = np.linspace(0, 1.2, 61)
        t["ball_position_after_step_m"][261:, 0, 0] = 1.2
        return t

    teachers, retained = {}, {}
    for dest, courses in (
        (teachers, {g: c.course for g, c in ctx.items()}),
        (retained, {g: old_plan.draws()[g].course for g in (4, 32)}),
    ):
        for group, course in courses.items():
            t = trace(course)
            path = tmp_path / f"{'positive' if dest is teachers else 'old'}-{group}.npz"
            np.savez(path, **t)
            dest[group] = dict(
                receipt_hash=hash_json(str(path)),
                outcome=label_first_touch(t).to_dict(),
                witnesses=dict(
                    trace=dict(path=str(path), sha256=pilot.digest(path)),
                    proof_payload=dict(path=str(payload), sha256=pilot.digest(payload)),
                ),
            )
    cfg = dict(
        config(),
        mesh_task_contact_world_hash=world_hash,
        source_root=str(source),
        source_head="test-head",
        core_root=str(tmp_path),
        output_root=str(tmp_path / "run"),
    )
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))

    def sealed(path, key="report_hash"):
        value = json.loads(Path(path).read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("test seal changed")
        return value

    p = SimpleNamespace(PINS={}, HEADS={}, stable=lambda: None)
    c = SimpleNamespace(
        cfg=cfg,
        source=source,
        core=tmp_path,
        root=Path(cfg["output_root"]),
        p=p,
        assets=SimpleNamespace(MEAN=parent),
        pins={},
        sealed=sealed,
        old_manifest={},
        positive_manifest={},
        old_plan=old_plan,
        positive_plan=pos_plan,
        contexts=ctx,
        teachers=teachers,
        retained=retained,
        candidate=dict(
            parameters=asdict(BoundedFirstTouchControl()), swing="bounded_contact_diagnostic"
        ),
        review=None,
        binding={},
        nominal=dict(results=[next(iter(teachers.values()))]),
        models=dict(student=student, zero=zero, off=None),
        scope=scope,
        fit_stage=dict(report_hash="mock-fit-stage"),
        fit_report=dict(report_hash="mock-fit"),
    )
    monkeypatch.setattr(pilot, "load_inputs", lambda _: c)
    monkeypatch.setattr(audit, "load_inputs", lambda _: c)
    monkeypatch.setattr(pilot.subprocess, "check_output", lambda *a, **kw: "")
    read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: (
            "MemAvailable: 67108864 kB\n"
            if str(path) == "/proc/meminfo"
            else read_text(path, *a, **kw)
        ),
    )
    allocations, calls = [], []

    def save(path, value, key="report_hash"):
        if key:
            value[key] = hash_json({k: v for k, v in value.items() if k != key})
        path.write_text(json.dumps(value))
        return value

    def bundle(*a, **kw):
        allocations.append(True)
        (c.root / "shared-payload.json").write_bytes(payload.read_bytes())
        return SimpleNamespace(
            factory=None, compiler=None, auditor=None, reader=None, encoder_session=None
        )

    monkeypatch.setattr(pilot, "allocate_native_first_touch_bundle", bundle)

    def episode(folder, group, candidate, **kw):
        calls.append((group, kw))
        if mode == "body_failure" and len(calls) == 4:
            raise ValueError("mock body failed")
        folder.mkdir()
        course = (
            kw["planning_context"].course
            if kw["planning_context"] is not None
            else kw["plan"].draws()[group].course
        )
        t = trace(course)
        model = kw["reference_student_model"]
        if model is not None:
            t.update({PREFIX + k: np.zeros((300, 1, width)) for k, width in FIELDS.items()})
        if mode == "off_corruption" and model is None:
            t["field0"][100, 0, 0] = 1e-12
        np.savez(folder / "physical_trace.npz", **t)
        (folder / "world.mjb").write_bytes(world.read_bytes())
        view = dict(
            mean_model=dict(model_hash=parent),
            seed=kw["plan"].draws()[group].sampling_seed,
            model_hash=hash_json(group),
        )
        commitment = dict(
            compiled_model_hash=world_hash,
            executed_motor_policy=dict(step_motor_proof=dict(model=view)),
            step_model_hash=view["model_hash"],
            hardware_authorized=False,
            promotion_authorized=False,
        )
        if model is not None:
            commitment.update(
                reference_student_model=model, reference_student_contract=model["decoder_contract"]
            )
        if kw["generated_seed_episode_binding"]:
            commitment["generated_seed_episode_binding"] = dict(
                seed=view["seed"],
                full_external_sampling_view_restored_and_validated=True,
                complete_generated_model_equals_external_view=True,
                full_original_factory_constructor_retained=True,
                sampling_law_changed=False,
            )
        save(folder / "commitment.json", commitment, key=None)
        report = save(
            folder / "report.json",
            dict(
                commitment,
                commitment_hash=hash_json(commitment),
                physical_trace_hash=pilot.digest(folder / "physical_trace.npz"),
                source_hash=pilot.digest(
                    source / "src/rosclaw_soccer/rsi/native_first_touch_episode.py"
                ),
                course=list(course),
                default_locomotion_planner_events=[],
            ),
        )
        review = dict(
            physical_substeps=3000,
            actual_recurrent_sampling_draws_reconstructed=270,
            high_quality=True,
            safety_passed=True,
            clean_foot_only=True,
            minimum_pelvis_z_m=0.8,
            reviewed_report_hash=report["report_hash"],
            source_hash=pilot.digest(
                source / "src/rosclaw_soccer/rsi/cpu_motor_transfer_evidence.py"
            ),
        )
        if model is not None:
            review.update(
                actual_mujoco_dynamics_replayed=True,
                actual_pd_torque_reconstructed=True,
                neural_target_reconstructed=True,
                reference_student_review=reference_review(model["model_hash"]),
            )
        foundation = dict(
            recomputed_executed_calls=300,
            physical_report_hash=report["report_hash"],
            physical_trace_hash=report["physical_trace_hash"],
            audit_source_hash=pilot.digest(
                source / "src/rosclaw_soccer/rsi/foundation_torch_capture_audit.py"
            ),
        )
        save(folder / "independent.json", review)
        save(folder / "foundation.json", foundation)
        outcome = label_first_touch(t)
        return dict(
            physical_substeps=3000,
            foundation_calls=300,
            sampling_draws=270,
            hardware_authorized=False,
            promotion_authorized=False,
            group=group,
            candidate=candidate,
            outcome=outcome.to_dict(),
            physical_report_hash=report["report_hash"],
            qualification_alignment=require_first_touch_qualification(
                review,
                foundation,
                outcome,
                reference_student_model_hash=None if model is None else model["model_hash"],
            ),
            **(dict(reference_student_model_hash=model["model_hash"]) if model is not None else {}),
            witnesses={
                k: dict(path=str(path), sha256=pilot.digest(path))
                for k, path in dict(
                    trace=folder / "physical_trace.npz",
                    report=folder / "report.json",
                    independent_review=folder / "independent.json",
                    foundation_review=folder / "foundation.json",
                    commitment=folder / "commitment.json",
                    world=folder / "world.mjb",
                    proof_payload=c.root / "shared-payload.json",
                ).items()
            },
        )

    monkeypatch.setattr(pilot, "qualified_episode", episode)

    def encoder(*a, **kw):
        value = dict(
            schema="soccer.rsi.sonic_encoder_reference_consumption.v1",
            actual_cpu_onnx_planner_calls=15,
            compared_encoder_frames=300,
            physical_improvement_proven=False,
        )
        return dict(value, report_hash=hash_json(value))

    monkeypatch.setattr(pilot, "audit_encoder_reference_consumption", encoder)
    for module in (pilot, audit):
        monkeypatch.setattr(module, "_sealed", lambda path, **kw: sealed(path))
    monkeypatch.setattr(audit, "require_terminal", lambda _: None)
    monkeypatch.setattr(audit, "FixedPhysicalPayloadReader", lambda *a, **kw: object())
    monkeypatch.setattr(
        audit,
        "reconstruct",
        lambda commitment, trace, **kw: reference_review(
            commitment["reference_student_model"]["model_hash"]
        ),
    )
    import mujoco

    model = SimpleNamespace(
        nq=43,
        nv=41,
        nu=29,
        jnt_range=np.tile([-3.0, 3.0], (29, 1)),
        joint=lambda n: SimpleNamespace(id=G1_DDS_JOINT_NAMES.index(n)),
    )
    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(from_binary_path=lambda _: model))
    return c, cfg_path, allocations, calls


@pytest.mark.parametrize("mode", ["normal", "preflight", "off_corruption", "body_failure"])
def test_all_thirteen_or_once_only_preserved_failure(tmp_path, monkeypatch, mode):
    c, path, allocations, calls = fixture(tmp_path, monkeypatch, mode)
    monkeypatch.setattr(
        sys,
        "argv",
        ["exam", "--config", str(path)] + (["--preflight"] if mode == "preflight" else []),
    )
    if mode in ("off_corruption", "body_failure"):
        with pytest.raises(ValueError, match="byte-exact|mock body"):
            pilot.main()
        d = json.loads((c.root / "declaration.json").read_text())
        state = FirstTouchCampaign(c.root, d).state
        assert state["phase"] == "FATAL_CONTRACT_VIOLATION"
        assert state["attempts"] == (1 if mode == "off_corruption" else 4)
        assert not (c.root / "complete.json").exists()
        with pytest.raises(ValueError, match="unused"):
            pilot.main()
        return
    pilot.main()
    if mode == "preflight":
        assert not allocations and not calls and not c.root.exists()
        return
    assert len(allocations) == 1 and len(calls) == 13
    assert sum(k[1]["generated_seed_episode_binding"] for k in calls) == 9
    output = tmp_path / "readback.json"
    monkeypatch.setattr(
        sys, "argv", ["reader", "--config", str(path), "--unit", "fake", "--output", str(output)]
    )
    audit.main()
    r = json.loads(output.read_text())
    assert r["source_native_bodies"] == 13 and r["causal_reference_frames_reconstructed"] == 3600
    assert not r["official_52_case_exam_completed"] and not r["promotion_authorized"]
    assert len(r["rows"]) == 13 and r["paired_results"]["positive_teacher_vs_student"]["pairs"] == 8


@pytest.mark.parametrize("changed", ["trace", "intent", "encoder", "prefix", "journal"])
def test_reader_cannot_accept_changed_original_evidence(tmp_path, monkeypatch, changed):
    c, path, _, _ = fixture(tmp_path, monkeypatch, "normal")
    monkeypatch.setattr(sys, "argv", ["exam", "--config", str(path)])
    pilot.main()
    if changed == "journal":
        import sqlite3

        with sqlite3.connect(c.root / "campaign.sqlite") as db:
            db.execute("DELETE FROM events")
    else:
        target = {
            "trace": c.root / "attempt-0012/physical_trace.npz",
            "intent": c.root / "intent-reference-exam-student-retention-g32.json",
            "encoder": c.root / "encoder-12.json",
            "prefix": c.root / "prefix-12.json",
        }[changed]
        target.write_bytes(target.read_bytes() + b" ")
    output = tmp_path / "rejected.json"
    monkeypatch.setattr(
        sys, "argv", ["reader", "--config", str(path), "--unit", "fake", "--output", str(output)]
    )
    with pytest.raises(ValueError, match="witness|journal"):
        audit.main()
    assert not output.exists()
