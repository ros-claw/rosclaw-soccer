"""Synthetic whole18 readback/physics orchestration, not actual G1 steps."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_verify_constant_forward as mod
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_constant_forward_approach import approach_review
from scripts.rsi_postcontact_release_transfer import paired_summary
from scripts.rsi_stage_release_contact_replay import digest
from tests.rsi.test_constant_forward_probe import scope, trace


@pytest.mark.parametrize("failure", [None, "physics", "raw", "missing", "pin"])
def test_whole18_or_fail_closed(monkeypatch, tmp_path, failure):
    from rosclaw_soccer.physics import recorded_early_episode

    root, output = tmp_path / "producer", tmp_path / "review"
    root.mkdir()

    def seal(path, d):
        d["report_hash"] = hash_json(d)
        write_once(path, d)
        return d

    a = trace()
    a["pelvis_z_per_substep_m"] = np.full((300, 1, 10), 0.793)
    a["ball_position_after_step_m"] = np.zeros((300, 1, 3))
    af = tmp_path / "trace.npz"
    with af.open("xb") as stream:
        np.savez_compressed(stream, **a)
    arrays = dict(path=str(af), sha256=digest(af))
    world = tmp_path / "world.mjb"
    world.write_bytes(b"synthetic test fixture")
    commitment = tmp_path / "commitment.json"
    write_once(
        commitment,
        dict(
            motor_policy_hash="policy",
            physics=dict(canonical_pd_kp=[1.0] * 29, canonical_pd_kd=[0.1] * 29),
        ),
    )
    witnesses = dict(
        trace=arrays,
        world=dict(path=str(world), sha256=digest(world)),
        commitment=dict(path=str(commitment), sha256=digest(commitment)),
    )
    native = tmp_path / "native.json"
    seal(native, dict(results=[dict(receipt_hash="receipt", witnesses=witnesses)] * 6))
    stage = tmp_path / "stage.json"
    seal(stage, dict(effective_cohort_spec=dict(producer_complete=str(native))))
    local = tmp_path / "local"
    local.mkdir()
    control = approach_review(a, a)
    outcome = label_first_touch(a).to_dict()
    rows, jobs = [], []
    for i in range(18):
        baseline_hash = "receipt"
        if i < 12:
            b = seal(local / f"pair-{i}-original_control-complete.json", dict(arrays=arrays))
            baseline_hash = b["report_hash"]
        job = dict(
            index=i,
            source_index=i % 6,
            course=[2.5, 0.1, 0.3],
            partition="CONSUMED_LOCAL_DEVELOPMENT" if i < 12 else "CONSUMED_ORIGINAL_RETENTION",
            baseline_hash=baseline_hash,
            baseline_outcome=outcome,
        )
        jobs.append(job)
        row = dict(
            job,
            source_receipt_hash="receipt",
            original_policy_hash="policy",
            arrays=arrays,
            outcome=outcome,
            control_review=control,
            causal_review=dict(
                constant_forward_approach=True, original_precontact_forward_law_unchanged=False
            ),
        )
        rows.append(seal(root / f"case-{i}-complete.json", row))
    cfg = dict(scope(), output_root=str(root), stage_readback=str(stage), local_root=str(local))
    pins = {str(af): digest(af)}
    if failure == "pin":
        pins[str(af)] = "sha256:" + "0" * 64
    d = seal(
        root / "declaration.json",
        dict(
            schema="soccer.rsi.constant_forward18_declaration.v1",
            config=cfg,
            input_file_hashes=pins,
            jobs=jobs,
        ),
    )
    summary = paired_summary(rows)
    summary["approach_mechanism_locally_promising"] = summary.pop("fixed_action_locally_promising")
    seal(
        root / "complete.json",
        dict(
            schema="soccer.rsi.constant_forward18_complete.v1",
            declaration_hash=d["report_hash"],
            rows=rows,
            new_native_steps=54000,
            forecast_steps=0,
            new_optimizer_updates=0,
            executed_control_foundation_pairs=5400,
            baseline_rollouts_reused=18,
            hardware_authorized=False,
            **summary,
        ),
    )
    if failure == "missing":
        (root / "case-17-complete.json").unlink()
    monkeypatch.setattr(mod, "require_terminal", lambda name: None)
    monkeypatch.setattr(
        mod.subprocess, "check_output", lambda args, **kw: "head\n" if "rev-parse" in args else ""
    )
    # ABI tensor shapes are separately checked by actual original_arrays tests.
    monkeypatch.setattr(mod, "original_arrays", lambda value: value)
    if failure == "raw":
        monkeypatch.setattr(mod, "approach_review", lambda *a: dict(wrong=True))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review",
            "--root",
            str(root),
            "--unit",
            "unit",
            "--source-head",
            "head",
            "--output",
            str(output),
        ],
    )
    calls = []

    class Physics:
        attempted_steps = 0

        def __init__(self, *a, **kw):
            calls.append(kw)

        def verify_unforced_control(self, data):
            if failure == "physics":
                self.attempted_steps = 10
                raise ValueError("synthetic physical mismatch")
            self.attempted_steps = 3000
            return dict(actual_physical_steps=3000, maximum_absolute_errors=dict(qpos=0.0))

    monkeypatch.setattr(recorded_early_episode, "RecordedEarlyEpisode", Physics)
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(MjModel=SimpleNamespace(from_binary_path=lambda path: object())),
    )
    if failure:
        with pytest.raises((ValueError, FileNotFoundError)):
            mod.main()
        assert not (output / "complete.json").exists()
        if failure == "physics":
            f = json.loads((output / "failure.json").read_text())
            assert len(calls) == 1 and f["current_attempted_steps"] == 10
            assert not f["automatic_retry"]
        else:
            assert calls == [] and not output.exists()
        return
    mod.main()
    c = json.loads((output / "complete.json").read_text())
    assert len(calls) == len(c["rows"]) == 18 and c["actual_physical_steps"] == 54000
    assert not c["independent_neural_policy_qualified"] and not c["training_authorized"]
