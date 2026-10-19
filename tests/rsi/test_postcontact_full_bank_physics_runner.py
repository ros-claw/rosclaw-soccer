"""Synthetic protocol tests, not G1 physical evidence or new training."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import rsi_replay_postcontact_full_bank_physics as mod


@pytest.mark.parametrize("error", [float("nan"), float("inf"), -0.1, 1e-7, True])
def test_nonfinite_or_invalid_review_rejected(error):
    with pytest.raises(ValueError, match="finite matching"):
        mod.require_full_physical_review(
            dict(actual_physical_steps=3000, maximum_absolute_errors=dict(q=error))
        )


@pytest.mark.parametrize("fail_at", [None, 20])
def test_full252_honest_reuse_and_partial_accounting(monkeypatch, tmp_path, fail_at):
    from rosclaw_soccer.physics import recorded_early_episode

    def seal(path, value):
        value = dict(value)
        value["report_hash"] = mod.hash_json(value)
        path.write_text(json.dumps(value))
        return value

    root, oldroot = tmp_path / "bank", tmp_path / "old"
    root.mkdir()
    oldroot.mkdir()
    world = tmp_path / "world.mjb"
    world.write_bytes(b"synthetic only")
    cm = tmp_path / "commitment.json"
    cm.write_text(
        json.dumps(dict(physics=dict(canonical_pd_kp=[1.0] * 29, canonical_pd_kd=[0.1] * 29)))
    )
    seal(
        tmp_path / "native.json",
        dict(
            results=[
                dict(
                    receipt_hash=f"r{i}",
                    witnesses=dict(
                        world=dict(path=str(world), sha256=mod.digest(world)),
                        commitment=dict(path=str(cm), sha256=mod.digest(cm)),
                    ),
                )
                for i in range(6)
            ]
        ),
    )
    seal(
        tmp_path / "stage.json",
        dict(effective_cohort_spec=dict(producer_complete=str(tmp_path / "native.json"))),
    )
    oldrows = [
        seal(
            oldroot / f"trial-{a}-complete.json",
            dict(
                source_trial_hash=f"old-{a}",
                physical_review=dict(
                    actual_physical_steps=3000, maximum_absolute_errors=dict(q=0.0)
                ),
            ),
        )
        for a in range(14)
    ]
    od = seal(
        oldroot / "declaration.json", dict(producer_complete_hash="authority", input_file_hashes={})
    )
    seal(
        oldroot / "complete.json",
        dict(
            schema="soccer.rsi.recorded_postcontact14_complete.v1",
            declaration_hash=od["report_hash"],
            rows=oldrows,
            actual_physical_steps=42000,
        ),
    )
    jobs = [dict(index=i, source_index=i % 6) for i in range(18)]
    contexts = []
    for ci in range(18):
        rows = []
        for ai in range(14):
            trace = root / f"trace-{ci}-{ai}.npz"
            np.savez_compressed(trace, synthetic_index=np.array([ci * 14 + ai]))
            rows.append(
                seal(
                    root / f"context-{ci}-action-{ai}-complete.json",
                    dict(
                        execution_reused=ci == 1,
                        source_receipt_hash=f"r{ci % 6}",
                        reused_case_hash=f"old-{ai}" if ci == 1 else None,
                        course=[2.5, 0.0, 0.2],
                        arrays=dict(path=str(trace), sha256=mod.digest(trace)),
                    ),
                )
            )
        contexts.append(seal(root / f"context-{ci}-complete.json", dict(rows=rows)))
    d = seal(
        root / "declaration.json",
        dict(
            jobs=jobs,
            reused_authority_hash="authority",
            config=dict(
                stage_readback=str(tmp_path / "stage.json"),
                release_physics_unit="old-physics.service",
                release_physics_root=str(oldroot),
                release_physics_complete_file_hash=mod.digest(oldroot / "complete.json"),
                reused_authority_context_index=1,
            ),
        ),
    )
    c = seal(root / "complete.json", dict(contexts=contexts, declaration_hash=d["report_hash"]))
    rbpath = tmp_path / "readback.json"
    seal(
        rbpath,
        dict(
            schema="soccer.rsi.postcontact_full_bank_readback.v1",
            complete_hash=c["report_hash"],
            input_file_hashes={},
            comparison_rollouts=252,
            reused_execution_count=14,
            actual_source_native_steps=714000,
            contexts=[
                dict(context_index=i, source_context_hash=x["report_hash"])
                for i, x in enumerate(contexts)
            ],
        ),
    )
    output = tmp_path / "physical"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "physics",
            "--root",
            str(root),
            "--unit",
            "bank.service",
            "--readback",
            str(rbpath),
            "--readback-unit",
            "raw.service",
            "--readback-file-hash",
            mod.digest(rbpath),
            "--source-head",
            "fake",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        mod.subprocess, "check_output", lambda argv, **kw: "fake" if "rev-parse" in argv else ""
    )
    units = []
    monkeypatch.setattr(mod, "require_terminal", units.append)
    monkeypatch.setattr(mod, "check_complete", lambda *a: None)
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(MjModel=SimpleNamespace(from_binary_path=lambda p: object())),
    )
    seen = []

    class Replay:
        def __init__(self, model, **kw):
            assert kw["kp"].shape == (29,) and kw["course"] == (2.5, 0.0, 0.2)
            self.attempted_steps = self.integrated_steps = 0
            self.data = SimpleNamespace(qpos=np.zeros(43), qvel=np.zeros(41), ctrl=np.zeros(29))

        def verify_postcontact_release(self, trace):
            i = int(trace["synthetic_index"][0])
            seen.append(i)
            if len(seen) - 1 == fail_at:
                self.attempted_steps = self.integrated_steps = 17
                raise RuntimeError("synthetic replay failure")
            self.attempted_steps = self.integrated_steps = 3000
            return dict(actual_physical_steps=3000, maximum_absolute_errors=dict(q=0.0))

    monkeypatch.setattr(recorded_early_episode, "RecordedEarlyEpisode", Replay)
    if fail_at is None:
        mod.main()
        result = json.loads((output / "complete.json").read_text())
        assert len(result["rows"]) == result["comparison_bodies_verified"] == 252
        assert result["new_physical_bodies"] == 238 and result["new_physical_steps"] == 714000
        assert result["reused_physical_bodies"] == 14 and result["reused_physical_steps"] == 42000
        assert result["new_optimizer_updates"] == result["new_neural_inferences"] == 0
        assert not result["training_authorized"] and not result["teacher_qualified"]
        assert seen == [i for i in range(252) if i // 14 != 1]
    else:
        with pytest.raises(RuntimeError, match="synthetic replay failure"):
            mod.main()
        failure = json.loads((output / "failure.json").read_text())
        assert failure["completed_new_bodies"] == 20
        assert failure["completed_new_physical_steps"] == 60000
        assert failure["reused_physical_bodies"] == 14
        assert failure["completed_comparison_bodies"] == 34
        assert failure["current_integrated_steps"] == failure["current_attempted_steps"] == 17
        assert not failure["automatic_retry"] and not (output / "complete.json").exists()
        assert (output / "failure-state.npz").exists()
    assert units == ["bank.service", "raw.service", "old-physics.service"]
