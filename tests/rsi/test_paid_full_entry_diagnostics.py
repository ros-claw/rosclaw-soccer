import sys

import pytest

from scripts import rsi_analyze_paid_full_entry_support as analysis
from scripts import rsi_review_paid_full_entry as review


@pytest.mark.parametrize("runner", [analysis, review])
@pytest.mark.parametrize("compact", [False, True])
def test_dirty_diagnostic_source_never_reads_models_or_allocates_physics(
    tmp_path, monkeypatch, runner, compact
):
    args = ["run", "--expected-head", "frozen", "--output", str(tmp_path / "out")]
    if runner is analysis:
        args += ["--data-root", str(tmp_path / "data"), "--data-hash", "hash", "--trained-roots"]
        args += [str(tmp_path / f"trained{i}") for i in range(3)]
        monkeypatch.setattr("scripts.rsi_native_context_recovery_common.digest", lambda _: "hash")
    else:
        args += [
            "--family-root",
            str(tmp_path / "family"),
            "--family-prefix",
            "experiment",
            "--stamp",
            "20261022",
            "--reference-root",
            str(tmp_path / "reference"),
            "--reference-hash",
            "hash",
            "--producer-head",
            "producer",
        ]
        if compact:
            args += ["--compact-support"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="clean|frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()


def test_interventional_pair_requires_actual_updates_and_same_ancestry():
    from copy import deepcopy

    learner = dict(
        data_hash="data",
        warm_hash="warm",
        body_scene_hash="body",
        parent_hash="parent",
        config=dict(protocol="interventional_dagger.warm_contact.v1", new_steps=2048),
        counts=dict(optimizer_updates=2048),
    )
    control = dict(
        data_hash="data",
        warm_hash="warm",
        body_scene_hash="body",
        parent_hash="parent",
        config=dict(
            protocol="interventional_dagger.geometry_only_control.v1",
            new_steps=0,
            control=True,
            seed_index=0,
        ),
        counts=dict(optimizer_updates=0),
    )
    review.validate_interventional_pair(learner, control)
    for key in ("data_hash", "warm_hash", "body_scene_hash", "parent_hash"):
        changed = deepcopy(control)
        changed[key] = "other"
        with pytest.raises(ValueError, match="same intervention"):
            review.validate_interventional_pair(learner, changed)
    changed = deepcopy(control)
    changed["counts"]["optimizer_updates"] = 1
    with pytest.raises(ValueError, match="zero-update"):
        review.validate_interventional_pair(learner, changed)
    changed = deepcopy(learner)
    changed["counts"]["optimizer_updates"] = 0
    with pytest.raises(ValueError, match="zero-update"):
        review.validate_interventional_pair(changed, control)


def test_interventional_reviewer_dirty_source_does_not_allocate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--expected-head",
            "frozen",
            "--output",
            str(tmp_path / "out"),
            "--family-root",
            str(tmp_path),
            "--family-prefix",
            "intervention",
            "--stamp",
            "20261022",
            "--reference-root",
            str(tmp_path),
            "--reference-hash",
            "hash",
            "--producer-head",
            "producer",
            "--interventional",
        ],
    )
    monkeypatch.setattr(review.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="clean"):
        review.main()
    assert not (tmp_path / "out").exists()
