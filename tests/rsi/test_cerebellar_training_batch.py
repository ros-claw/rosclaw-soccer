"""Synthetic packet consistency, NOT physical replay or dataset qualification."""

import copy
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi import cpu_motor_transfer_evidence as cpu_module
from rosclaw_soccer.rsi import foundation_torch_capture_audit as foundation_module
from rosclaw_soccer.rsi.cerebellar_torque_policy import initialize
from rosclaw_soccer.rsi.cerebellar_training_batch import build_batch, episode_arrays
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.native_cerebellar_torque import NativeCerebellarTorque
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

WORLD = hash_bytes(b"synthetic-not-a-world")


def make_packet(seed=92):
    model = initialize(seed=42, world_hash=WORLD, time_aware=True)
    layer = NativeCerebellarTorque(model, world_hash=WORLD, sampling_seed=seed)
    q, v, target = np.zeros(43), np.zeros(41), np.zeros(29)
    q[3] = 1
    q[2] = 0.7
    fields = {name: [] for name in layer.capture_widths()}
    for _ in range(3000):
        _, row = layer.advance(q, v, target, np.zeros(29))
        for name, value in row.items():
            fields[name].append(value)
    trace = {
        name: np.asarray(rows).reshape(300, 1, 10, layer.capture_widths()[name])
        for name, rows in fields.items()
    }
    trace.update(
        joint_target_rad=np.zeros((300, 1, 29)),
        force_n=np.zeros((300, 1, 6)),
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
        ball_position_after_step_m=np.zeros((300, 1, 3)),
    )
    trace["force_n"][85, 0, 1] = 2
    trace["ball_position_after_step_m"][86:, 0, 0] = np.arange(1, 215) / 50
    commitment = dict(
        schema="soccer.rsi.cpu_motor_transfer.v2",
        compiled_model_hash=WORLD,
        cerebellar_torque_contract=layer.contract(),
    )
    report = dict(commitment, physical_trace_hash="sha256:" + "b" * 64)
    report["report_hash"] = hash_json(report)
    review = dict(
        reviewed_report_hash=report["report_hash"],
        source_hash=hash_bytes(Path(cpu_module.__file__).read_bytes()),
        actual_mujoco_dynamics_replayed=True,
        actual_pd_torque_reconstructed=True,
        cerebellar_model_hash=model["model_hash"],
        actual_cerebellar_inferences_reconstructed=3000,
        actual_cerebellar_sampling_draws_reconstructed=3000,
        cerebellar_actual_pd_plus_residual_torque_reconstructed=True,
        physical_substeps=3000,
        actual_recurrent_sampling_draws_reconstructed=270,
        high_quality=True,
        safety_passed=True,
        clean_foot_only=True,
        minimum_pelvis_z_m=0.7,
    )
    foundation = dict(
        audit_source_hash=hash_bytes(Path(foundation_module.__file__).read_bytes()),
        physical_report_hash=report["report_hash"],
        physical_trace_hash=report["physical_trace_hash"],
        recomputed_executed_calls=300,
    )
    outcome = label_first_touch(trace)
    receipt = dict(
        cerebellar_model_hash=model["model_hash"],
        cerebellar_sampling_draws=3000,
        cerebellar_inference_calls=3000,
        cerebellar_model_optimizer_steps=0,
        hardware_authorized=False,
        promotion_authorized=False,
        teacher_qualified=False,
        private_fresh_accessed=False,
        physical_report_hash=report["report_hash"],
        outcome=outcome.to_dict(),
        qualification_alignment=require_first_touch_qualification(
            review,
            foundation,
            outcome,
            cerebellar_model_hash=model["model_hash"],
            cerebellar_sampling_draws=3000,
        ),
    )
    return model, receipt, report, commitment, trace, review, foundation


@pytest.fixture(scope="module")
def packet():
    return make_packet()


def test_full_numeric_packet_keeps_actual_raw_latent_and_127_observations(packet):
    arrays, objective = episode_arrays(*packet)
    assert arrays["features"].shape == (3000, 127)
    assert arrays["latent_action"].shape == (3000, 29)
    np.testing.assert_array_equal(
        arrays["latent_action"], packet[4]["cerebellar_sampling_latent_action"].reshape(3000, 29)
    )
    assert np.all(arrays["terminal_return"] == objective["training_return"])
    assert objective["raw_trace_authenticated_here"] is False


@pytest.mark.parametrize("kind", ["unsafe", "censored", "no_contact"])
def test_legal_failures_are_included_not_dropped(packet, kind):
    model, receipt, report, commitment, trace, review, foundation = copy.deepcopy(packet)
    if kind == "unsafe":
        trace["pelvis_z_per_substep_m"][299, 0, 9] = 0.1
        review.update(minimum_pelvis_z_m=0.1, safety_passed=False)
    else:
        trace["force_n"][:] = 0
        if kind == "censored":
            trace["force_n"][260, 0, 1] = 2
        review["high_quality"] = False
    outcome = label_first_touch(trace)
    receipt["outcome"] = outcome.to_dict()
    receipt["qualification_alignment"] = require_first_touch_qualification(
        review,
        foundation,
        outcome,
        cerebellar_model_hash=model["model_hash"],
        cerebellar_sampling_draws=3000,
    )
    arrays, objective = episode_arrays(
        model, receipt, report, commitment, trace, review, foundation
    )
    assert len(arrays["features"]) == 3000
    assert not objective["formal_outcome"]["task_pass"]
    assert arrays["terminal_return"][0] < 0


@pytest.mark.parametrize(
    "kind", ["model", "world", "density", "capture", "safety", "review_source", "foundation_source"]
)
def test_mismatched_behavior_or_unqualified_data_rejected(packet, kind):
    values = list(copy.deepcopy(packet))
    if kind == "model":
        values[1]["cerebellar_model_hash"] = WORLD
    elif kind == "world":
        values[3]["compiled_model_hash"] = "sha256:" + "b" * 64
    elif kind == "density":
        values[4]["cerebellar_sampling_log_probability"][2, 0, 0, 0] += 0.01
    elif kind == "capture":
        values[4]["cerebellar_input_features"] = values[4]["cerebellar_input_features"][:-1]
    elif kind == "safety":
        values[5]["safety_passed"] = False
    elif kind == "review_source":
        values[5]["source_hash"] = WORLD
    else:
        values[6]["audit_source_hash"] = WORLD
    with pytest.raises(ValueError):
        episode_arrays(*values)


def test_missing_declared_episodes_rejected_before_any_file_or_payload_read(packet, tmp_path):
    with pytest.raises(ValueError, match="complete unique"):
        build_batch(
            [],
            actor=packet[0],
            expected_jobs=[dict(job="missing")],
            declaration_hash=WORLD,
            evidence_root=tmp_path,
            physical_payload_reader=None,
        )


def test_complete_file_batch_pipeline_with_mocked_physical_authentication(tmp_path, monkeypatch):
    """Real file/hash/aggregation paths, but physical report reader is mocked.

    The synthetic world and reviews are NOT physics evidence or admitted data.
    """
    import json

    from rosclaw_soccer.rsi import failure_curriculum_evidence as evidence

    def mocked_sealed(path, **kwargs):
        value = json.loads(path.read_text())
        assert value["report_hash"] == hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        )
        return value

    monkeypatch.setattr(evidence, "_sealed", mocked_sealed)
    root = tmp_path.resolve()
    paths, jobs = [], []
    actor = None
    jobs = [
        dict(
            job=f"collect-{i}",
            group=i,
            context_hash=hash_json(dict(context=i)),
            sampling_seed=92 + i,
            kind="collection",
            actor_seed=42,
        )
        for i in range(2)
    ]
    declaration = dict(
        jobs=jobs,
        mode="SIM_ONLY",
        neural_training_authorized=True,
        hardware_authorized=False,
        promotion_authorized=False,
        private_fresh_accessed=False,
    )
    declaration_hash = hash_json(declaration)
    declaration["report_hash"] = declaration_hash
    (root / "declaration.json").write_text(json.dumps(declaration))
    for i in range(2):
        actor, r, report, commitment, trace, review, foundation = make_packet(92 + i)
        folder = root / f"episode-{i}"
        folder.mkdir()
        features = np.zeros((300, 1, 640), dtype=np.float32)
        trace["foundation_neural_encoder_features"] = features
        trace_path = folder / "trace.npz"
        np.savez_compressed(trace_path, **trace)
        report["physical_trace_hash"] = hash_bytes(trace_path.read_bytes())
        report["commitment_hash"] = hash_json(commitment)
        report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
        review["reviewed_report_hash"] = report["report_hash"]
        foundation.update(
            physical_report_hash=report["report_hash"],
            physical_trace_hash=report["physical_trace_hash"],
        )
        r["physical_report_hash"] = report["report_hash"]
        job = jobs[i]
        r.update(job=job["job"], group=i, planning_context_hash=job["context_hash"])
        contents = dict(
            report=report,
            commitment=commitment,
            independent_review=review,
            foundation_review=foundation,
            intent_witness=dict(
                job, declaration_hash=declaration_hash, model_hash=actor["model_hash"]
            ),
            encoder_witness=dict(
                compared_encoder_frames=300,
                compared_features_per_frame=640,
                actual_cpu_onnx_planner_calls=15,
                recorded_features_hash=hash_bytes(features.tobytes()),
                comparison_atol=1e-5,
                maximum_absolute_error=0.0,
            ),
            model_witness=actor,
        )
        r["witnesses"] = {"trace": dict(path=str(trace_path), sha256=report["physical_trace_hash"])}
        for key, value in contents.items():
            if key != "commitment":
                value = {k: v for k, v in value.items() if k != "report_hash"}
                value["report_hash"] = hash_json(value)
            p = folder / f"{key}.json"
            p.write_text(json.dumps(value))
            w = dict(path=str(p), sha256=hash_bytes(p.read_bytes()))
            if key.endswith("_witness"):
                r[key] = w
            else:
                r["witnesses"][key] = w
                if key in ("independent_review", "foundation_review"):
                    r[key + "_hash"] = value["report_hash"]
        for key, content in (("world", b"synthetic-not-a-world"), ("proof_payload", b"mock")):
            p = folder / key
            p.write_bytes(content)
            r["witnesses"][key] = dict(path=str(p), sha256=hash_bytes(content))
        r["receipt_hash"] = hash_json({k: v for k, v in r.items() if k != "receipt_hash"})
        p = root / f"receipt-{i}.json"
        p.write_text(json.dumps(r))
        paths.append(p)
    arrays, manifest = build_batch(
        paths,
        actor=actor,
        expected_jobs=jobs,
        declaration_hash=declaration_hash,
        evidence_root=root,
        physical_payload_reader=None,
    )
    assert arrays["features"].shape == (6000, 127)
    assert manifest["whole_declared_collection_included"]
    assert manifest["new_physical_replays"] == 0
    filtered = copy.deepcopy(declaration)
    filtered["jobs"].append(dict(jobs[0], job="unreported-third", sampling_seed=94))
    filtered["report_hash"] = hash_json({k: v for k, v in filtered.items() if k != "report_hash"})
    (root / "declaration.json").write_text(json.dumps(filtered))
    with pytest.raises(ValueError, match="entire predeclared"):
        build_batch(
            paths,
            actor=actor,
            expected_jobs=jobs,
            declaration_hash=filtered["report_hash"],
            evidence_root=root,
            physical_payload_reader=None,
        )
    (root / "declaration.json").write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="predeclared"):
        build_batch(
            paths,
            actor=actor,
            expected_jobs=jobs[::-1],
            declaration_hash=declaration_hash,
            evidence_root=root,
            physical_payload_reader=None,
        )
    with pytest.raises(ValueError, match="predeclared"):
        build_batch(
            paths,
            actor=actor,
            expected_jobs=jobs,
            declaration_hash="sha256:" + "c" * 64,
            evidence_root=root,
            physical_payload_reader=None,
        )
    trace_path.write_bytes(trace_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="witness file"):
        build_batch(
            paths,
            actor=actor,
            expected_jobs=jobs,
            declaration_hash=declaration_hash,
            evidence_root=root,
            physical_payload_reader=None,
        )
