import json

import numpy as np
import pytest

from rosclaw_soccer.rsi import output_memory_step_motor as motor
from scripts import rsi_audit_memory_learning_rollouts as audit


@pytest.mark.parametrize("generation,wrong_seed", [(0, 202610336), (1, 202610335)])
def test_actual_sampling_view_cannot_reuse_a_wrong_generation_seed(
    monkeypatch, tmp_path, generation, wrong_seed
):
    model = dict(model_hash="current", generation=generation)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model))
    monkeypatch.setattr(motor, "validate_model", lambda model: None)
    parent = dict(
        report_hash="parent",
        training_course_seed=11,
        single_course_lane=0,
        source_hash="runner",
        asset_hash="asset",
        environments=[dict(course="course")],
    )
    view = dict(schema=motor.SAMPLING_SCHEMA, mean_model=model, model_hash="view", seed=wrong_seed)
    raw = dict(
        parent,
        report_hash="sample",
        parent_report_hash="parent",
        contact_motor_policy_hash="policy",
        contact_motor_policy=dict(step_motor_proof=dict(model=view)),
    )
    monkeypatch.setattr(
        audit, "_sealed", lambda p: parent if "reproduction-parent" in str(p) else raw
    )
    monkeypatch.setattr(audit, "audit_lateral_approach", lambda p: None)

    def measured(folder, policy_hash, commitment, *, decoder_sink):
        decoder_sink.append(object())
        return dict(report=raw, outcome={})

    monkeypatch.setattr(audit, "_outcome", measured)
    monkeypatch.setattr(
        audit, "gpu_observations", lambda folder, report: (np.empty((0, 134)), [], raw)
    )
    job = dict(
        kind="output-memory",
        model_path=str(path),
        model_hash="current",
        index=0,
        course=[11, 0],
        exploration_root=str(tmp_path),
        commitment=dict(
            samples_per_course=4,
            runner_hash="runner",
            asset_hash="asset",
            sampling_view_hashes=["view"] * 4,
        ),
        row=dict(
            index=0,
            seed=11,
            lane=0,
            parent_report_hash="parent",
            samples=[dict(sample=s, view_hash="view", report_hash="sample") for s in range(4)],
        ),
    )
    with pytest.raises(ValueError, match="declared current physical policy"):
        audit.audit_course(job)
