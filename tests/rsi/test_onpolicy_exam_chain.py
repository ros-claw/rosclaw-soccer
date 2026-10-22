import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import rsi_verify_dual_onpolicy_stage as stage


@pytest.mark.parametrize("failure", [None, 1, 2, 3])
def test_onpolicy_exam_chain_closes_each_evidence_stage_once(tmp_path, monkeypatch, failure):
    train = tmp_path / "train"
    train.mkdir()
    (train / "complete.json").write_text(
        json.dumps(
            dict(
                source_head="producer",
                status="DUAL_AR1_ACTOR_CRITIC_ACTUALLY_UPDATED_PENDING_PHYSICAL_EXAM",
                online_RL_claim=True,
                accepted_optimizer_steps=1,
                hardware_authorized=False,
                promotion_authorized=False,
            )
        )
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chain",
            "--training-root",
            str(train),
            "--producer-head",
            "producer",
            "--rehearsal-root",
            str(tmp_path / "old"),
            "--rehearsal-hash",
            "sha256:old",
            "--expected-head",
            "head",
            "--output",
            str(out),
        ],
    )
    monkeypatch.setattr(
        stage.subprocess, "check_output", lambda args, **kw: b"" if "status" in args else "head\n"
    )
    statuses = [
        "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS",
        "DUAL_RECURRENT_NATIVE_TRAIN_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT",
        "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS",
    ]
    calls = []

    def run(argv, *, env, check):
        calls.append(argv)
        assert check and env["CUDA_VISIBLE_DEVICES"] == ""
        if failure == len(calls):
            raise subprocess.CalledProcessError(1, argv)
        path = Path(argv[argv.index("--output") + 1])
        path.mkdir()
        (path / "complete.json").write_text(
            json.dumps(
                dict(
                    source_head="head",
                    status=statuses[len(calls) - 1],
                    old_protected_retained=False,
                    old_success_cases=[],
                    consumed_run_successes=0,
                    hardware_authorized=False,
                    promotion_authorized=False,
                )
            )
        )

    monkeypatch.setattr(stage.subprocess, "run", run)
    if failure is None:
        stage.main()
        r = json.loads((out / "complete.json").read_text())
        assert len(calls) == 3 and len(r["completed"]) == 3
        assert r["goal_complete"] is False and r["promotion_authorized"] is False
    else:
        with pytest.raises(subprocess.CalledProcessError):
            stage.main()
        r = json.loads((out / "failed.json").read_text())
        assert len(calls) == failure and len(r["completed"]) == failure - 1
        assert r["automatic_retry"] is False and not (out / "complete.json").exists()
