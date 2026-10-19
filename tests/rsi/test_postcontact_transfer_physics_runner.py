"""Synthetic all18 orchestration; physical guards have separate real toy tests."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import rsi_replay_postcontact_transfer_physics as mod


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("failure_case", [None, 7])
def test_all18_modes_or_explicit_partial_failure(monkeypatch, tmp_path, adaptive, failure_case):
    from rosclaw_soccer.physics import recorded_early_episode

    def sealed(file, value):
        value = dict(value)
        value["report_hash"] = mod.hash_json(value)
        file.write_text(json.dumps(value))
        return value

    root = tmp_path / "producer"
    root.mkdir()
    world = tmp_path / "world.mjb"
    world.write_bytes(b"synthetic, not physical world")
    cm = tmp_path / "commitment.json"
    cm.write_text(
        json.dumps(
            dict(
                motor_policy_hash="policy",
                physics=dict(canonical_pd_kp=[1.0] * 29, canonical_pd_kd=[0.1] * 29),
            )
        )
    )
    native = sealed(
        tmp_path / "native.json",
        dict(
            results=[
                dict(
                    receipt_hash=f"receipt-{i}",
                    witnesses=dict(
                        world=dict(path=str(world), sha256=mod.digest(world)),
                        commitment=dict(path=str(cm), sha256=mod.digest(cm)),
                    ),
                )
                for i in range(6)
            ]
        ),
    )
    stage = sealed(
        tmp_path / "stage.json",
        dict(effective_cohort_spec=dict(producer_complete=str(tmp_path / "native.json"))),
    )
    assert native and stage
    rows = []
    for i in range(18):
        path = root / f"case-{i}-trace.npz"
        np.savez_compressed(path, synthetic_index=np.array([i]))
        rows.append(
            sealed(
                root / f"case-{i}-complete.json",
                dict(
                    index=i,
                    source_index=i % 6,
                    source_receipt_hash=f"receipt-{i % 6}",
                    original_policy_hash="policy",
                    course=[2.5, 0.0, 0.2],
                    arrays=dict(path=str(path), sha256=mod.digest(path)),
                    outcome=dict(task_pass=False, safe=True),
                ),
            )
        )
    d = sealed(
        root / "declaration.json", dict(config=dict(stage_readback=str(tmp_path / "stage.json")))
    )
    c = sealed(root / "complete.json", dict(rows=rows, declaration_hash=d["report_hash"]))
    rb = sealed(
        tmp_path / "readback.json",
        dict(
            schema="soccer.rsi.postcontact_transfer_readback.v1",
            complete_hash=c["report_hash"],
            actual_native_steps=54000,
            input_file_hashes={},
            rows=[
                dict(index=i, source_case_hash=r["report_hash"], outcome=r["outcome"])
                for i, r in enumerate(rows)
            ],
        ),
    )
    assert rb
    output = tmp_path / "physical"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "physical",
            "--root",
            str(root),
            "--unit",
            "producer.service",
            "--readback",
            str(tmp_path / "readback.json"),
            "--readback-unit",
            "readback.service",
            "--readback-file-hash",
            mod.digest(tmp_path / "readback.json"),
            "--source-head",
            "synthetic-head",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        mod.subprocess,
        "check_output",
        lambda argv, **kw: "synthetic-head" if "rev-parse" in argv else "",
    )
    units = []
    monkeypatch.setattr(mod, "require_terminal", lambda name: units.append(name))
    monkeypatch.setattr(mod, "check_complete", lambda *a: (adaptive, {}))
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(MjModel=SimpleNamespace(from_binary_path=lambda path: object())),
    )
    seen = []

    class Replay:
        def __init__(self, model, **kw):
            assert kw["course"] == (2.5, 0.0, 0.2) and kw["kp"].shape == (29,)
            self.attempted_steps = self.integrated_steps = 0
            self.data = SimpleNamespace(qpos=np.zeros(43), qvel=np.zeros(41), ctrl=np.zeros(29))

        def verify(self, trace, mode):
            i = int(trace["synthetic_index"][0])
            seen.append((i, mode))
            if i == failure_case:
                self.attempted_steps = self.integrated_steps = 20
                raise RuntimeError("synthetic partial physics failure")
            self.attempted_steps = self.integrated_steps = 3000
            return dict(actual_physical_steps=3000, teacher_qualified=False)

        def verify_postcontact_exit(self, trace):
            return self.verify(trace, "exit")

        def verify_postcontact_release(self, trace):
            return self.verify(trace, "release")

    monkeypatch.setattr(recorded_early_episode, "RecordedEarlyEpisode", Replay)
    if failure_case is not None:
        with pytest.raises(RuntimeError, match="synthetic partial"):
            mod.main()
        fail = json.loads((output / "failure.json").read_text())
        assert fail["completed_cases"] == 7 and fail["current_case"] == 7
        assert fail["current_attempted_steps"] == 20 and fail["current_integrated_steps"] == 20
        assert not fail["automatic_retry"] and not (output / "complete.json").exists()
        assert (output / "failure-state.npz").exists()
    else:
        mod.main()
        result = json.loads((output / "complete.json").read_text())
        assert result["actual_physical_steps"] == 54000 and len(result["rows"]) == 18
        assert result["new_optimizer_updates"] == result["new_neural_inferences"] == 0
        assert not result["teacher_qualified"] and not result["training_authorized"]
        assert seen == [(i, "exit" if adaptive else "release") for i in range(18)]
    assert units == ["producer.service", "readback.service"]
