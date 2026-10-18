"""Whole orchestration with explicitly mocked physics, payload and NN engines."""

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.receding_teacher_scope import receding_contexts
from rosclaw_soccer.rsi.reference_student_execution import PREFIX
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_read_reference_student_three as audit
from scripts import rsi_reference_student_three_pilot as pilot
from tests.rsi.test_reference_student_probe import config


def fixture(tmp_path, monkeypatch, mode):
    source = Path(pilot.__file__).resolve().parents[1]
    mean = "sha256:" + "a" * 64
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), mean, 18)
    context = next(c for c in receding_contexts(plan) if c.source_group == 76)
    candidate = dict(
        parameters=asdict(BoundedFirstTouchControl()), swing="bounded_contact_diagnostic"
    )

    def save(path, value, key="report_hash"):
        path = Path(path)
        if key:
            value[key] = hash_json({k: v for k, v in value.items() if k != key})
        path.write_text(json.dumps(value))
        return value

    baseline = {f"field{i}": np.zeros((300, 1, 1)) for i in range(25)}
    baseline.update(
        {
            k: np.zeros((300, 1, 29))
            for k in ("pre_motor_joint_target_rad", "joint_target_rad", "joint_position_rad")
        }
    )
    baseline.update(
        canonical_qpos=np.zeros((300, 1, 43)),
        canonical_qvel=np.zeros((300, 1, 41)),
        force_n=np.zeros((300, 1, 6)),
        ball_position_after_step_m=np.zeros((300, 1, 3)),
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.8),
    )
    baseline["canonical_qpos"][0, 0, :7] = (0, 0, 0.793, 1, 0, 0, 0)
    baseline["canonical_qpos"][0, 0, 7:36] = G1SonicRunupController.default_angles
    baseline["canonical_qpos"][0, 0, 36:43] = (*context.course[:2], 0.13, 1, 0, 0, 0)
    baseline["canonical_qvel"][0, 0, 35:41] = (
        context.course[2],
        0,
        0,
        0,
        context.course[2] / 0.11,
        0,
    )
    baseline["force_n"][200, 0, 0] = 2
    baseline["ball_position_after_step_m"][200:261, 0, 0] = np.linspace(0, 1.2, 61)
    baseline["ball_position_after_step_m"][261:, 0, 0] = 1.2
    old_trace = tmp_path / "old-trace.npz"
    np.savez(old_trace, **baseline)
    old_payload = tmp_path / "old-payload.json"
    old_payload.write_text("synthetic payload")
    old = dict(
        receipt_hash="baseline",
        candidate=candidate,
        outcome=label_first_touch(baseline).to_dict(),
        witnesses=dict(
            trace=dict(path=str(old_trace), sha256=pilot.digest(old_trace)),
            proof_payload=dict(path=str(old_payload), sha256=pilot.digest(old_payload)),
        ),
    )
    prior, diagnostic = (
        dict(report_hash="prior", reused_endpoints=[dict(pair_id="receding-g76", receipt=old)]),
        dict(report_hash="diagnostic"),
    )
    owner_path, fit_path, nominal_path = (
        tmp_path / n for n in ("owner.json", "fit.json", "nominal.json")
    )
    save(owner_path, dict(source_root=str(source)), key=None)
    fit = save(fit_path, dict(options=[dict(candidate=None)] + [dict(candidate=candidate)] * 7))
    save(nominal_path, dict(results=[{}]))
    world = tmp_path / "world.mjb"
    world.write_bytes(b"synthetic world, no dynamics")
    cfg = dict(
        config(),
        source_root=str(source),
        core_root=str(tmp_path),
        source_head="test-head",
        output_root=str(tmp_path / "run"),
        max_generations=1,
        plateau_generations=1,
        deadline_unix=10**20,
        evidence_owner_configuration=str(owner_path),
        evidence_owner_configuration_file_hash=pilot.digest(owner_path),
        baseline_receipt_hash="baseline",
        portfolio_fit=str(fit_path),
        portfolio_fit_hash=fit["report_hash"],
        nominal_readback=str(tmp_path / "review"),
        nominal_readback_file_hash="review",
        nominal_complete=str(nominal_path),
        mesh_task_contact_world_hash=pilot.digest(world),
    )
    cfg_path = tmp_path / "config.json"
    save(cfg_path, cfg, key=None)
    p = SimpleNamespace(
        PINS={},
        HEADS={},
        stable=lambda: None,
        sealed=lambda path: json.loads(Path(path).read_text()),
    )
    for module in (pilot, audit):
        monkeypatch.setattr(
            module, "prepare", lambda path: (p, SimpleNamespace(MEAN=mean), {}, cfg)
        )
        monkeypatch.setattr(module, "whole_predecessor", lambda *a: (prior, diagnostic))
        monkeypatch.setattr(
            module, "ReviewedContactWorld", lambda *a: SimpleNamespace(verify=lambda h: {})
        )
    monkeypatch.setattr(
        pilot.subprocess,
        "check_output",
        lambda args, **kw: "test-head" if "rev-parse" in args else "",
    )
    old_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: (
            "MemAvailable: 67108864 kB\n"
            if str(path) == "/proc/meminfo"
            else old_read(path, *a, **kw)
        ),
    )
    allocations, calls = [], []

    def bundle(*args, **kwargs):
        allocations.append(True)
        (Path(cfg["output_root"]) / "shared-payload.json").write_bytes(old_payload.read_bytes())
        return SimpleNamespace(factory=None, compiler=None, auditor=None, reader=None)

    def episode(folder, group, selected, **kwargs):
        calls.append(kwargs["reference_student_model"])
        folder.mkdir()
        trace = copy.deepcopy(baseline)
        model = kwargs["reference_student_model"]
        if model is not None:
            trace[PREFIX + "applied_reference_increment_rad"] = np.zeros((300, 1, 12))
            if model["parameters"]["head_bias"][4] != 0:
                trace[PREFIX + "applied_reference_increment_rad"][30:, 0, 4] = 0.05
                trace["pre_motor_joint_target_rad"][30:, 0, 4] = 0.05
                trace["joint_target_rad"][30:, 0, 4] = 0.04
                trace["joint_position_rad"][31:, 0, 4] = 0.03
        elif mode == "off_corruption":
            trace["field0"][100, 0, 0] = 1e-12
        np.savez(folder / "physical_trace.npz", **trace)
        (folder / "world.mjb").write_bytes(world.read_bytes())
        view = dict(
            mean_model=dict(model_hash=mean), seed=plan.draws()[76].sampling_seed, model_hash="view"
        )
        commitment = dict(
            compiled_model_hash=cfg["mesh_task_contact_world_hash"],
            executed_motor_policy=dict(step_motor_proof=dict(model=view)),
            step_model_hash="view",
        )
        if model is not None:
            commitment.update(
                reference_student_model=model, reference_student_contract=model["decoder_contract"]
            )
        save(folder / "commitment.json", commitment, key=None)
        report = save(
            folder / "report.json.gz",
            dict(
                commitment,
                commitment_hash=hash_json(commitment),
                source_hash=pilot.digest(
                    source / "src/rosclaw_soccer/rsi/native_first_touch_episode.py"
                ),
                course=list(context.course),
                physical_trace_hash=pilot.digest(folder / "physical_trace.npz"),
            ),
        )
        outcome = label_first_touch(trace)
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
                reference_student_review=dict(
                    schema="soccer.rsi.pre_parent_reference_student_reconstruction.v1",
                    model_hash=model["model_hash"],
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
                ),
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
        save(folder / "context.json", dict(context_hash=context.context_hash))
        names = dict(
            trace=folder / "physical_trace.npz",
            report=folder / "report.json.gz",
            independent_review=folder / "independent.json",
            foundation_review=folder / "foundation.json",
            commitment=folder / "commitment.json",
            world=folder / "world.mjb",
            proof_payload=Path(cfg["output_root"]) / "shared-payload.json",
        )
        receipt = dict(
            physical_substeps=3000,
            foundation_calls=300,
            sampling_draws=270,
            group=group,
            candidate=selected,
            outcome=outcome.to_dict(),
            physical_report_hash=report["report_hash"],
            qualification_alignment=require_first_touch_qualification(
                review,
                foundation,
                outcome,
                reference_student_model_hash=None if model is None else model["model_hash"],
            ),
            witnesses={
                k: dict(path=str(path), sha256=pilot.digest(path)) for k, path in names.items()
            },
            planning_context_witness=dict(
                path=str(folder / "context.json"), sha256=pilot.digest(folder / "context.json")
            ),
            hardware_authorized=False,
            promotion_authorized=False,
        )
        if model is not None:
            receipt["reference_student_model_hash"] = model["model_hash"]
        return receipt

    monkeypatch.setattr(pilot, "allocate_native_first_touch_bundle", bundle)
    monkeypatch.setattr(pilot, "qualified_episode", episode)
    monkeypatch.setattr(audit, "require_terminal", lambda unit: None)
    monkeypatch.setattr(audit, "FixedPhysicalPayloadReader", lambda *a, **kw: object())
    monkeypatch.setattr(audit, "_sealed", lambda path, **kw: json.loads(Path(path).read_text()))
    return cfg, cfg_path, allocations, calls


@pytest.mark.parametrize("mode", ["normal", "off_corruption", "preflight"])
def test_three_body_sequence_full_off_gate_and_independent_readback(tmp_path, monkeypatch, mode):
    cfg, path, allocations, calls = fixture(tmp_path, monkeypatch, mode)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--config", str(path)] + (["--preflight"] if mode == "preflight" else []),
    )
    if mode == "off_corruption":
        with pytest.raises(ValueError, match="byte-exact"):
            pilot.main()
        declaration = json.loads((Path(cfg["output_root"]) / "declaration.json").read_text())
        state = FirstTouchCampaign(Path(cfg["output_root"]), declaration).state
        assert state["phase"] == "FATAL_CONTRACT_VIOLATION" and state["attempts"] == 1
        assert len(calls) == 1 and not (Path(cfg["output_root"]) / "complete.json").exists()
    else:
        pilot.main()
        if mode == "preflight":
            assert not allocations and not calls and not Path(cfg["output_root"]).exists()
            return
        assert len(calls) == 3 and len(allocations) == 1 and calls[0] is None
        output = tmp_path / "readback.json"
        monkeypatch.setattr(
            sys, "argv", ["audit", "--config", str(path), "--unit", "fake", "--output", str(output)]
        )
        audit.main()
        report = json.loads(output.read_text())
        assert report["original_off_all33_arrays_bit_exact"]
        assert report["observed_authority"]["reference_control_authority_observed"]
        assert not report["probe_is_trained_policy"] and not report["teacher_qualified"]


@pytest.mark.parametrize("changed", ["trace", "intent", "parity", "payload", "journal"])
def test_whole_reader_rejects_changed_original_evidence(tmp_path, monkeypatch, changed):
    cfg, path, _, _ = fixture(tmp_path, monkeypatch, "normal")
    monkeypatch.setattr(sys, "argv", ["pilot", "--config", str(path)])
    pilot.main()
    root = Path(cfg["output_root"])
    if changed == "journal":
        import sqlite3

        with sqlite3.connect(root / "campaign.sqlite") as db:
            db.execute("DELETE FROM events")
    else:
        target = {
            "trace": root / "attempt-0002/physical_trace.npz",
            "intent": root / "intent-reference-left-ankle-c6.json",
            "parity": root / "parity-2.json",
            "payload": root / "shared-payload.json",
        }[changed]
        target.write_bytes(target.read_bytes() + b" ")
    output = tmp_path / "rejected.json"
    monkeypatch.setattr(
        sys, "argv", ["audit", "--config", str(path), "--unit", "fake", "--output", str(output)]
    )
    with pytest.raises(ValueError, match="witness|journal"):
        audit.main()
    assert not output.exists()
