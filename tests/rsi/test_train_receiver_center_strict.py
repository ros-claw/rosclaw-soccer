import json
import sys

import pytest

from scripts import rsi_train_receiver_center_strict as train


def test_zero_clean_positive_rejected_before_optimizer(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "clean", lambda _: None)
    monkeypatch.setattr(train, "digest", lambda _: "pinned")
    monkeypatch.setattr(
        train, "ReceiverCritic", lambda: pytest.fail("critic allocated prematurely")
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    neutral = dict(
        strict_both_clean_handoff=False,
        strict_receiver_clean_handoff=False,
        passer_foot_precedes_receiver=False,
        both_bodies_healthy=True,
        original_handoff=False,
        passer_nonfoot_substep=None,
        receiver_nonfoot_substep=None,
    )
    records = [
        dict(context=c, side=s, seed=seed, physical_clean_outcome=neutral, terminal_reward=0.0)
        for c in range(12)
        for s in (-0.2, 0.2)
        for seed in (92117, 92143, 92171)
    ]
    (dataset / "complete.json").write_text(json.dumps(dict(records=records)))
    for name in ("audit.json", "anchor.json", "old.json"):
        (tmp_path / name).write_text("{}")
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--dataset",
            str(dataset),
            "--audit",
            str(tmp_path / "audit.json"),
            "--anchor-admission",
            str(tmp_path / "anchor.json"),
            "--anchor-dataset",
            str(tmp_path / "old.json"),
            "--output",
            str(output),
            "--dataset-hash",
            "pinned",
            "--audit-hash",
            "pinned",
            "--audit-head",
            "head",
            "--anchor-admission-hash",
            "pinned",
            "--anchor-dataset-hash",
            "pinned",
            "--expected-head",
            "head",
            "--seed",
            "92401",
        ],
    )
    with pytest.raises(ValueError, match="strict positive/negative"):
        train.main()
    assert not output.exists()
