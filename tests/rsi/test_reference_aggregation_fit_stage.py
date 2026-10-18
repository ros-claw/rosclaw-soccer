"""Conditional stage with ALL physics/cohort auditing and optimizer mocked.

Real numerical preparation is exercised; these are not physical/128-step fits.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.reference_state_aggregation import (
    LABEL_CEILING,
    prepare_proposed_aggregation,
)
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_fit_reference_state_aggregation as stage
from tests.rsi.test_reference_aggregation_decision import pair, results
from tests.rsi.test_reference_state_aggregation import PARENT, inputs


def fixture(tmp_path, monkeypatch, mode):
    def save(path, value):
        value = dict(value)
        value["report_hash"] = hash_json(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return value

    arrays, kw = inputs()
    batch, proposed = prepare_proposed_aggregation(*arrays, **kw)
    numeric = tmp_path / "proposed.npz"
    np.savez(
        numeric,
        actual_observations=batch["original_observations"],
        reference_proposals=batch["targets"],
        partitions=np.array(batch["partitions"]),
        source_seeds=np.array(batch["source_seeds"]),
        experimental_supervision=np.array(batch["eligible"]),
    )
    prep_path = tmp_path / "prepared.json"
    prep = save(
        prep_path,
        dict(
            schema="soccer.rsi.proposed_reference_aggregation_prepared.v1",
            proposed_batch=proposed,
            dataset_admission_authorized=False,
            original_query_eligibility_overwritten=False,
            input_file_hashes={},
            numeric_arrays=dict(path=str(numeric), sha256=stage.digest(numeric)),
        ),
    )
    pc_path, cc_path = tmp_path / "producer.json", tmp_path / "contact.json"
    proot = tmp_path / "closed"
    pc = dict(
        output_root=str(proot),
        behavior_model_hash=PARENT,
        mesh_task_contact_world_hash="sha256:" + "b" * 64,
    )
    pc_path.write_text(json.dumps(pc))
    save(proot / "declaration.json", dict(input_file_hashes={}))
    save(proot / "complete.json", dict(mock="all physical endpoints are mocked"))
    reader = tmp_path / "reader.py"
    reader.write_text("# mock audit; not physics evidence\n")
    audit_path = tmp_path / "audit.json"
    audit = save(audit_path, dict(reader_source_hash=stage.digest(reader), input_file_hashes={}))
    replay_root, metrics_path = tmp_path / "replay", tmp_path / "metrics.json"
    replay = save(replay_root / "complete.json", dict(mock="replay"))
    metrics = save(metrics_path, dict(mock="metrics"))
    pairs = results()
    if mode in ("fit", "preflight", "failure"):
        pairs["positive_teacher_vs_student"] = pair(8, 4, 0)
    elif mode == "retention":
        pairs["retention_zero_vs_student"] = pair(2, 1, 0)
    dense_path = tmp_path / "dense.json"
    save(
        dense_path,
        dict(
            upstream_readback_hash=audit["report_hash"],
            replay_complete_hash=replay["report_hash"],
            metrics_hash=metrics["report_hash"],
            paired_results=pairs,
            measured_contact_readout=dict(mock="all physics", rows=[]),
        ),
    )
    cc = dict(
        producer_config=str(pc_path),
        producer_config_file_hash=stage.digest(pc_path),
        readback=str(audit_path),
        complete_path=str(dense_path),
        replay_output_root=str(replay_root),
        metrics_output=str(metrics_path),
        audit_source=str(reader),
    )
    cc_path.write_text(json.dumps(cc))
    cfg = dict(
        profile="conditional-one-round-reference-aggregation-fit-v1",
        mode="SIM_ONLY",
        source_root=str(Path(stage.__file__).resolve().parents[1]),
        source_head="mock-head",
        core_root=str(tmp_path / "empty-core"),
        output_root=str(tmp_path / "run"),
        deadline_unix=10**20,
        acknowledged_label_ceiling=LABEL_CEILING,
        experimental_numeric_fit_authorized=True,
        fit_steps=128,
        fit_seed=2026101902,
        maximum_numeric_candidates=1,
        new_physical_steps=0,
        new_foundation_inferences=0,
        upstream_units=["mock-physical.service", "mock-audit.service", "mock-dense.service"],
        prepared_aggregation=str(prep_path),
        prepared_aggregation_hash=prep["report_hash"],
        current_contact_configuration=str(cc_path),
        current_contact_configuration_file_hash=stage.digest(cc_path),
        **{
            k: False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "dataset_admission_authorized",
                "teacher_auto_admission",
                "private_fresh_accessed",
                "online_rl_claimed",
                "validation_checkpoint_selection",
            )
        },
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    calls = []
    monkeypatch.setattr(stage, "require_terminal", lambda unit: None)
    monkeypatch.setattr(stage, "require_cohort", lambda *a: pairs)
    monkeypatch.setattr(stage, "measured_readout", lambda *a: dict(mock="all physics", rows=[]))
    monkeypatch.setattr(
        stage.subprocess,
        "check_output",
        lambda cmd, **kw: "mock-head\n" if cmd == ["git", "rev-parse", "HEAD"] else "",
    )

    def fit(batch, **kwargs):
        calls.append(kwargs)
        assert int(batch["train"].sum()) == 6 and int(batch["validation"].sum()) == 2
        if mode == "failure":
            raise RuntimeError("mock optimizer failure")
        return dict(
            report_hash="mock-fit",
            numeric_model=dict(model_hash="mock-model"),
            learning_receipt=dict(completed_optimizer_steps=128),
        )

    monkeypatch.setattr(stage, "fit_numeric_reference_student", fit)
    monkeypatch.setattr(
        sys,
        "argv",
        ["stage", "--config", str(config_path)] + (["--preflight"] if mode == "preflight" else []),
    )
    return Path(cfg["output_root"]), calls


@pytest.mark.parametrize("mode", ["fit", "retention", "ready", "preflight", "failure"])
def test_whole_stage_fits_once_only_when_declared_science_branch_calls_for_it(
    tmp_path, monkeypatch, mode
):
    root, calls = fixture(tmp_path, monkeypatch, mode)
    if mode == "failure":
        with pytest.raises(RuntimeError, match="mock optimizer"):
            stage.main()
        assert (root / "failure.json").exists() and not (root / "complete.json").exists()
    else:
        stage.main()
        if mode == "preflight":
            assert not root.exists() and not calls
            return
        report = json.loads((root / "complete.json").read_text())
        assert report["optimizer_updates"] == (128 if mode == "fit" else 0)
        assert (
            not report["dataset_admission_authorized"] and not report["own_body_student_evaluated"]
        )
        assert (root / "numeric-fit.json").exists() == (mode == "fit")
    assert len(calls) == (1 if mode in ("fit", "failure") else 0)
    if calls:
        assert calls[0]["steps"] == 128 and calls[0]["ignore_past_reference"] is True
    with pytest.raises(ValueError, match="one matched"):
        stage.main()
    assert len(calls) <= 1
