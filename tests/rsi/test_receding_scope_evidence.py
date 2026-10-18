"""Scope input authentication tests only; CPU/Foundation qualification mocked."""

import json

import pytest

from rosclaw_soccer.rsi import receding_scope_evidence as evidence


@pytest.mark.parametrize(
    "tamper", [None, "proof", "nonfoot", "newton", "nan", "baseline", "member"]
)
def test_whole_zero_expert_and_both_original_none_controls_required(tmp_path, monkeypatch, tamper):
    def digest(path):
        return "digest-" + path.name

    def receipt(group):
        return dict(
            group=group, candidate=None, receipt_hash=str(group), job=f"control-{group}", outcome={}
        )

    archived = dict(receipt_hash="archived", candidate={})
    controls = [receipt(76), receipt(77)]
    proof = dict(
        report_hash="proof",
        schema="soccer.rsi.archived_receding_expert_readback.v1",
        actual_all3000_cerebellar_outputs_exactly_zero=True,
        teacher_qualified=False,
        input_file_hashes={},
        recovered_receipt_hash="archived",
        recovered_outcome={},
    )
    contact = dict(
        report_hash="contact",
        schema="soccer.rsi.contact_exit_metrics_readback.v1",
        input_file_hashes={},
        rows=[
            dict(
                source_receipt_hash="archived",
                original_outcome={},
                metrics=dict(
                    first_nonfoot_robot_contact_substep=None,
                    maximum_linear_newton_residual_n=1e-14,
                ),
            )
        ],
    )
    results = [{}] * 18
    results[12], results[14] = controls
    screen = dict(report_hash="screen", results=results)
    audit = dict(
        report_hash="audit",
        complete_hash="screen",
        source_native_bodies=18,
        schema="soccer.rsi.mesh_option4_teacher_screen_terminal_readback.v1",
        reader_source_hash="digest-reader",
        whole_input_and_witness_files_verified=True,
        shared_payload_logical_seals_rebuilt=True,
        prior_independent_replays_authenticated=True,
        rows=list(controls),
    )
    cfg = {
        "archived_expert_readback": str(tmp_path / "proof"),
        "archived_expert_readback_hash": "proof",
        "archived_expert_contact_metrics": str(tmp_path / "contact"),
        "archived_expert_contact_metrics_hash": "contact",
        "archived_expert_complete": str(tmp_path / "complete"),
        "baseline_complete": str(tmp_path / "screen"),
        "baseline_terminal_readback": str(tmp_path / "audit"),
        "baseline_terminal_readback_hash": "audit",
        "baseline_reader_source": str(tmp_path / "reader"),
        "mesh_task_contact_world_hash": "world",
    }
    documents = dict(
        proof=proof,
        contact=contact,
        complete=dict(results=[{}] * 7 + [archived]),
        screen=screen,
        audit=audit,
    )
    if tamper == "proof":
        proof["actual_all3000_cerebellar_outputs_exactly_zero"] = False
    elif tamper == "nonfoot":
        contact["rows"][0]["metrics"]["first_nonfoot_robot_contact_substep"] = 2400
    elif tamper == "newton":
        contact["rows"][0]["metrics"]["maximum_linear_newton_residual_n"] = 1
    elif tamper == "nan":
        contact["rows"][0]["metrics"]["maximum_linear_newton_residual_n"] = float("nan")
    elif tamper == "baseline":
        audit["prior_independent_replays_authenticated"] = False
    elif tamper == "member":
        audit["rows"] = audit["rows"][:1]
    for key, doc in documents.items():
        (tmp_path / key).write_text(json.dumps(doc))
    monkeypatch.setattr(evidence, "authenticate_receipt", lambda *a, **k: None)
    if tamper:
        with pytest.raises(ValueError):
            evidence.load_scope_evidence(cfg, lambda p: documents[p.name], digest, {})
    else:
        reused, actual = evidence.load_scope_evidence(cfg, lambda p: documents[p.name], digest, {})
        assert len(reused) == 2 and actual == archived
