"""Orchestration only: numerical/physics engines explicitly mocked throughout."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import PublicPlanningContext
from scripts import rsi_positive_lateral_physical_pilot as pilot
from tests.rsi.test_surface_receding_diagnostics import dense_fixture


def config():
    return dict(
        behavior_model_hash="sha256:" + "a" * 64,
        groups=[12, 13, 28, 29, 76, 77, 92, 93],
        study_profile="positive-lateral-sixteen",
        mode="SIM_ONLY",
        max_attempts=16,
        new_physical_budget=16,
        new_independent_cpu_budget=16,
        combined_body_substep_budget=96000,
        selected_option_index=2,
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


@pytest.mark.parametrize(
    "tamper",
    [None, "partial", "cpu", "steps", "option", "bool", "mesh", "fresh", "bind", "witness"],
)
def test_fixed_original_sixteen_body_protocol(tamper):
    cfg = config()
    if tamper == "partial":
        cfg["new_physical_budget"] = 8
    elif tamper == "cpu":
        cfg["new_independent_cpu_budget"] = 0
    elif tamper == "steps":
        cfg["combined_body_substep_budget"] = 48000
    elif tamper == "option":
        cfg["selected_option_index"] = 4
    elif tamper == "bool":
        cfg["selected_option_index"] = True
    elif tamper == "mesh":
        cfg["use_mesh_surface_swing"] = True
    elif tamper == "fresh":
        cfg["private_fresh_accessed"] = True
    elif tamper == "bind":
        cfg["generated_seed_episode_binding"] = False
    elif tamper == "witness":
        cfg["required_receipt_witnesses"].pop()
    if tamper:
        with pytest.raises(ValueError):
            pilot.require_scope(cfg)
    else:
        pilot.require_scope(cfg)


@pytest.mark.parametrize(
    "tamper", [None, "teacher", "budget", "union", "hash", "world", "witness", "receipt"]
)
def test_full_preceding_failure_bank_is_reopened_before_new_source_trials(
    tmp_path, monkeypatch, tamper
):
    # This tests orchestration and file identity. Numerical receipt/audit
    # qualification is mocked and is NOT claimed as physical proof.
    from rosclaw_soccer.rsi.surface_receding_diagnostics import scoped_metrics

    def save(name, document):
        path = tmp_path / name
        path.write_text(json.dumps(document))
        return str(path)

    reader = save("reader.py", {})
    shared = save("witness.json", dict(report_hash="witness"))
    receipt_paths = []
    receipts = []
    for i in range(6):
        r = dict(receipt_hash=f"r{i}", outcome=dict(task_pass=False, safe=i != 2))
        for key in ("intent_witness", "prefix_witness", "encoder_witness"):
            r[key] = dict(path=shared, sha256=pilot.digest(shared))
        receipts.append(r)
        receipt_paths.append(save(f"receipt-{i:04d}.json", r))
    previous_path = save(
        "previous.json",
        dict(
            output_root=str(tmp_path),
            mesh_task_contact_world_hash="world",
            portfolio_fit_hash="fit",
        ),
    )
    save("declaration.json", dict(prepared_pins={}, input_file_hashes={}))
    save("complete.json", dict(report_hash="complete", results=receipts))
    readback = save(
        "readback.json", dict(report_hash="audit", reader_source_hash=pilot.digest(reader))
    )
    paired = dict(regressed_pairs=0, newly_unsafe_pairs=0, net_success_gain=0)
    selected = scoped_metrics(*dense_fixture())
    diagnostic = dict(
        schema="soccer.rsi.surface_receding_contact_diagnostic_complete.v1",
        report_hash="diagnostic",
        upstream_readback_hash="audit",
        new_fixed_control_replay_substeps=18000,
        total_metrics_endpoints=28,
        scoped_contact_metrics=selected,
        research_decision=pilot.research_decision(paired, selected),
        teacher_qualified=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        fresh_exam=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if tamper == "teacher":
        diagnostic["teacher_qualified"] = True
    elif tamper == "budget":
        diagnostic["new_fixed_control_replay_substeps"] = 0
    elif tamper == "union":
        diagnostic["total_metrics_endpoints"] = 18
    elif tamper == "hash":
        diagnostic["upstream_readback_hash"] = "other"
    elif tamper == "witness":
        Path(shared).write_text("changed")
    elif tamper == "receipt":
        Path(receipt_paths[-1]).write_text("{}")
    diagnostic_path = save("diagnostic.json", diagnostic)
    cfg = dict(
        mesh_task_contact_world_hash="other" if tamper == "world" else "world",
        portfolio_fit_hash="fit",
        surface_predecessor_review=dict(
            configuration=previous_path,
            configuration_file_hash=pilot.digest(previous_path),
            readback=readback,
            reader_source=reader,
            diagnostic_complete=diagnostic_path,
            complete_hash="complete",
            readback_hash="audit",
            diagnostic_complete_hash="diagnostic",
        ),
    )
    calls, baseline_calls = [], []
    monkeypatch.setattr(pilot, "require_endpoints", lambda *a: (paired, {}))
    monkeypatch.setattr(pilot, "authenticate_receipt", lambda r, **kw: calls.append(r))
    monkeypatch.setattr(pilot, "load_baselines", lambda *a: baseline_calls.append(True))
    args = (cfg, tmp_path, lambda p: json.loads(Path(p).read_text()), {})
    if tamper:
        with pytest.raises(ValueError):
            pilot.whole_predecessor(*args)
    else:
        complete, _ = pilot.whole_predecessor(*args)
        assert len(calls) == 6 and not calls[2]["outcome"]["safe"]
        assert len(baseline_calls) == 1 and complete["results"] == receipts


@pytest.mark.parametrize("fail_at,preflight", [(None, False), (3, False), (None, True)])
def test_all_jobs_journaled_or_failure_preserved_without_implicit_retry(
    tmp_path, monkeypatch, fail_at, preflight
):
    source = Path(pilot.__file__).resolve().parents[1]
    behavior = "sha256:" + "a" * 64
    scope = build_scope(behavior)
    plan = PublicFirstTouchTrainingPlan(tuple(scope["source_plan"]["seeds"]), behavior, 19)
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
        for c in scope["contexts"]
    )

    def save(name, document):
        path = tmp_path / name
        path.write_text(json.dumps(document))
        return str(path)

    candidate = dict(
        parameters=asdict(BoundedFirstTouchControl()), swing="bounded_contact_diagnostic"
    )
    fit = dict(report_hash="fit", options=[dict(candidate=c) for c in [None] + [candidate] * 7])
    reg_path = save("registry.json", dict(scope=scope, scope_hash=scope["scope_hash"]))
    fit_path = save("fit.json", fit)
    nominal = save("nominal.json", dict(results=[{}]))
    cfg = dict(
        config(),
        source_root=str(source),
        core_root=str(tmp_path),
        source_head="frozen-test-source",
        output_root=str(tmp_path / "run"),
        sampling_registry=reg_path,
        sampling_registry_file_hash=pilot.digest(reg_path),
        sampling_registry_hash="registry",
        portfolio_fit=fit_path,
        portfolio_fit_hash="fit",
        nominal_readback=str(tmp_path / "review"),
        nominal_readback_file_hash="review",
        nominal_complete=nominal,
        mesh_task_contact_world_hash="world",
        deadline_unix=10**20,
        max_generations=1,
        plateau_generations=1,
        slot_predecessor_unit="previous",
    )
    path = save("config.json", cfg)
    monkeypatch.setattr(
        pilot.subprocess,
        "check_output",
        lambda args, **kw: "frozen-test-source" if "rev-parse" in args else "",
    )
    p = SimpleNamespace(
        sealed=lambda path: json.loads(Path(path).read_text()),
        PINS={},
        HEADS={},
        stable=lambda: None,
    )
    monkeypatch.setattr(pilot, "prepare", lambda path: (p, SimpleNamespace(MEAN=behavior), {}, cfg))
    monkeypatch.setattr(pilot, "episode_inputs", lambda *a, **k: (plan, {}, contexts))
    monkeypatch.setattr(
        pilot,
        "whole_predecessor",
        lambda *a: (
            dict(report_hash="complete"),
            dict(report_hash="diagnostic", research_decision="retained"),
        ),
    )
    monkeypatch.setattr(
        pilot, "ReviewedContactWorld", lambda *a: SimpleNamespace(verify=lambda h: {})
    )
    monkeypatch.setattr(pilot, "require_terminal", lambda unit: None)
    original_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **k: (
            "MemAvailable: 67108864 kB\n"
            if str(path) == "/proc/meminfo"
            else original_read(path, *a, **k)
        ),
    )
    allocations, calls = [], []

    def bundle(*args, **kwargs):
        allocations.append(True)
        return SimpleNamespace(
            factory=None, compiler=None, auditor=None, reader=None, encoder_session=None
        )

    def episode(folder, group, candidate, **kwargs):
        index = len(calls)
        calls.append((group, candidate, kwargs))
        if index == fail_at:
            raise RuntimeError("synthetic body failure")
        folder.mkdir()
        np.savez(folder / "physical_trace.npz", dummy=np.zeros((300, 1, 1)))
        witnesses = {}
        for name in (
            "trace",
            "report",
            "independent_review",
            "foundation_review",
            "commitment",
            "world",
            "proof_payload",
        ):
            f = folder / f"{name}.json"
            f.write_text("{}")
            witnesses[name] = dict(path=str(f), sha256=pilot.digest(f))
        return dict(
            candidate=candidate,
            group=group,
            physical_substeps=3000,
            foundation_calls=300,
            sampling_draws=270,
            witnesses=witnesses,
            outcome=dict(task_pass=candidate is not None, safe=True),
            promotion_authorized=False,
            hardware_authorized=False,
        )

    monkeypatch.setattr(pilot, "allocate_native_first_touch_bundle", bundle)
    monkeypatch.setattr(pilot, "qualified_episode", episode)
    monkeypatch.setattr(
        pilot, "_sealed", lambda *a, **k: dict(default_locomotion_planner_events=[])
    )
    monkeypatch.setattr(pilot, "audit_encoder_reference_consumption", lambda *a: {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--config", path] + (["--preflight"] if preflight else []),
    )
    if fail_at is not None:
        with pytest.raises(RuntimeError, match="synthetic body"):
            pilot.main()
    else:
        pilot.main()
    root = Path(cfg["output_root"])
    if preflight:
        assert not root.exists() and not allocations and not calls
        return
    declaration = json.loads((root / "declaration.json").read_text())
    campaign = FirstTouchCampaign(root, declaration)
    assert len(allocations) == 1
    if fail_at is not None:
        assert campaign.state["phase"] == "FATAL_CONTRACT_VIOLATION"
        assert campaign.state["attempts"] == 4 and len(campaign.state["completed"]) == 3
        assert not (root / "complete.json").exists()
    else:
        result = json.loads((root / "complete.json").read_text())
        assert len(calls) == 16 and len(result["results"]) == 16
        assert result["paired_development"]["pairs"] == 8
        assert result["source_scene_clusters"] == 4
        assert not result["teacher_qualified"] and not result["neural_training_authorized"]
        assert [g for g, _, _ in calls] == [j["group"] for j in scope["proposed_jobs"]]
        assert all(kwargs["generated_seed_episode_binding"] for _, _, kwargs in calls)
        assert [r["candidate"] for r in result["results"]] == [None, candidate] * 8
    with pytest.raises(ValueError, match="unused output"):
        pilot.main()
    assert len(allocations) == 1
