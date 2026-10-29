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
@pytest.mark.parametrize("mode", ["sampling", "generation_pair", "successor_transition"])
def test_candidate_sampling_declares_all_real_candidate_arms(monkeypatch, tmp_path, block, mode):
    frontier = tmp_path / "frontier"
    frontier.mkdir()
    (frontier / "complete.json").write_text(
        json.dumps({"reports": [{"seed": 65023, "achieved_contexts": list(range(22))}]})
    )
    candidate = tmp_path / "candidate"
    candidate.touch()
    parent = tmp_path / "parent"
    parent.touch()
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
    monkeypatch.setattr(
        rsi_native_context_recovery_common,
        "digest",
        lambda path: "parenthash" if path == parent else "hash",
    )
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
    ]
    if mode == "sampling":
        args.append("--receiver-child-velocity-sampling")
    for name in (
        "receiver-checkpoint",
        "receiver-candidate-audit",
        "receiver-dataset",
        "receiver-velocity-candidate",
        "receiver-velocity-candidate-receipt",
    ):
        args.extend(["--" + name, str(candidate), "--" + name + "-hash", "hash"])
    if mode in ("generation_pair", "successor_transition"):
        for name in ("receiver-velocity-parent", "receiver-velocity-parent-receipt"):
            args.extend(["--" + name, str(parent), "--" + name + "-hash", "parenthash"])
    if mode == "successor_transition":
        args.append("--receiver-generation-transition-pair")
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(DeclarationCaptured):
        rsi_collect_neural_relay_chain.main()
    if mode == "sampling":
        assert captured["protocol"].endswith("candidate_sampling.development.v55")
        assert len(captured["courses"]) == 32
        for offset in range(0, 32, 4):
            arms = [row["arm"] for row in captured["courses"][offset : offset + 4]]
            assert all(arm["receiver_child_candidate"] is True for arm in arms)
            assert [arm["exploration_seed"] for arm in arms] == [None, 92117, 92143, 92171]
            assert all(arm["receiver_adaptive_intercept"] for arm in arms)
        assert captured["joint_targets_remain_actual_neural_outputs"] is False
    elif mode == "generation_pair":
        assert captured["protocol"].endswith("generation_pair.development.v56")
        assert len(captured["courses"]) == 16
        for offset in range(0, 16, 2):
            arms = [row["arm"] for row in captured["courses"][offset : offset + 2]]
            assert [arm["receiver_child_candidate"] for arm in arms] == [False, True]
            assert all(arm["receiver_adaptive_intercept"] for arm in arms)
        assert captured["receiver_velocity_parent_hash"] == "parenthash"
        assert captured["joint_targets_remain_actual_neural_outputs"] is True
    else:
        assert captured["protocol"].endswith("successor_transition.development.v57")
        assert len(captured["courses"]) == 16
        for offset in range(0, 16, 2):
            arms = [row["arm"] for row in captured["courses"][offset : offset + 2]]
            assert all(arm["receiver_child_candidate"] is True for arm in arms)
            assert [arm["receiver_successor_smooth"] for arm in arms] == [False, True]
        assert captured["receiver_generation_transition_controls"] == 10
        assert captured["joint_targets_remain_actual_neural_outputs"] is False
