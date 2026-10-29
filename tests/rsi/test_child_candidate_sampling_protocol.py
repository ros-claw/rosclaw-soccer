"""Exercise the actual collector's declaration before any physics is allocated."""

import json
import sys

import pytest

from scripts import (
    rsi_atomic_artifacts,
    rsi_collect_neural_relay_chain,
    rsi_native_context_recovery_common,
    rsi_native_fresh_common,
)


class DeclarationCaptured(Exception):
    pass


@pytest.mark.parametrize("block", [0, 1, 2])
def test_candidate_sampling_declares_all_real_candidate_arms(monkeypatch, tmp_path, block):
    frontier = tmp_path / "frontier"
    frontier.mkdir()
    (frontier / "complete.json").write_text(
        json.dumps({"reports": [{"seed": 65023, "achieved_contexts": list(range(22))}]})
    )
    candidate = tmp_path / "candidate"
    candidate.touch()
    captured = {}

    def capture(path, value):
        captured.update(value)
        raise DeclarationCaptured

    monkeypatch.setattr(rsi_atomic_artifacts, "write_once", capture)
    monkeypatch.setattr(rsi_native_fresh_common, "clean", lambda head: None)
    monkeypatch.setattr(
        rsi_native_fresh_common,
        "admit_frontier",
        lambda *args: ({}, "base", [{"seed": 65023}], []),
    )
    monkeypatch.setattr(rsi_native_context_recovery_common, "digest", lambda path: "hash")
    monkeypatch.setattr(rsi_native_context_recovery_common, "check_pins", lambda pins: None)
    args = [
        "collector",
        "--frontier-root",
        str(frontier),
        "--frontier-head",
        "head",
        "--frontier-hash",
        "hash",
        "--expected-head",
        "head",
        "--output",
        str(tmp_path / "out"),
        "--new-context-block",
        str(block),
        "--receiver-learned-pair",
        "--receiver-dense-pair",
        "--receiver-continuation-pair",
        "--receiver-precontact-continuation-pair",
        "--receiver-child-return-role-pair",
        "--receiver-velocity-match-prepared-pair",
        "--receiver-child-velocity-sampling",
    ]
    for name in (
        "receiver-checkpoint",
        "receiver-candidate-audit",
        "receiver-dataset",
        "receiver-velocity-candidate",
        "receiver-velocity-candidate-receipt",
    ):
        args.extend(["--" + name, str(candidate), "--" + name + "-hash", "hash"])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(DeclarationCaptured):
        rsi_collect_neural_relay_chain.main()
    assert captured["protocol"].endswith("candidate_sampling.development.v55")
    assert len(captured["courses"]) == 32
    for offset in range(0, 32, 4):
        arms = [row["arm"] for row in captured["courses"][offset : offset + 4]]
        assert all(arm["receiver_child_candidate"] is True for arm in arms)
        assert [arm["exploration_seed"] for arm in arms] == [None, 92117, 92143, 92171]
        assert all(arm["receiver_adaptive_intercept"] for arm in arms)
    assert captured["joint_targets_remain_actual_neural_outputs"] is False
