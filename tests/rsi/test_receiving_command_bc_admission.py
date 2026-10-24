"""Synthetic bad-report admission tests, never physical/learning evidence."""

import hashlib
import json
import sys

import pytest


@pytest.mark.parametrize(
    "fault", ["safety", "functional", "controlled", "strict", "no_gain", "own_law", "own_rows"]
)
def test_failed_teacher_review_cannot_start_BC_or_allocate_world(tmp_path, monkeypatch, fault):
    from scripts.rsi_train_receiving_command_bc import main

    def write(path, obj):
        path.parent.mkdir(exist_ok=True)
        raw = json.dumps(obj).encode()
        path.write_bytes(raw)
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    rows = [dict(index=i) for i in range(96)]
    sh = write(tmp_path / "source/complete.json", dict(records=rows))
    own = dict(
        source_hash=sh,
        independent_whole_world_replay=True,
        sampling_law_audit_passed=True,
        records=rows,
    )
    if fault == "own_law":
        own["sampling_law_audit_passed"] = False
    if fault == "own_rows":
        own["records"] = rows[:95]
    oh = write(tmp_path / "own/complete.json", own)
    effect = dict(
        safety_losses=[],
        controlled_regressions=[],
        functional_regressions=[],
        strict_regressions=[],
        functional_new=56,
        functional_old=40,
    )
    if fault == "no_gain":
        effect["functional_new"] = 40
    elif fault in ("safety", "functional", "controlled", "strict"):
        effect[
            {
                "safety": "safety_losses",
                "functional": "functional_regressions",
                "controlled": "controlled_regressions",
                "strict": "strict_regressions",
            }[fault]
        ] = [60]
    rh = write(
        tmp_path / "review/complete.json",
        dict(
            status="PROJECTED_RECEIVER_ROLE_PAIRED96_REVIEWED_NOT_LEARNING_OR_PROMOTION",
            overall=effect,
            source_hash=sh,
            own_hash=oh,
        ),
    )
    monkeypatch.setattr("scripts.rsi_native_fresh_common.clean", lambda _: None)
    monkeypatch.setattr(
        "scripts.rsi_native_fresh_common.source_pins",
        lambda _: pytest.fail("no allocation/pinning after failed gate"),
    )
    args = ["test"]
    for name in ("source-root", "own-root", "review-root", "scene", "velocity-config", "output"):
        value = tmp_path / (
            {"source-root": "source", "own-root": "own", "review-root": "review"}.get(name, name)
        )
        args += ["--" + name, str(value)]
    args += ["--review-hash", rh, "--expected-head", "synthetic"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="ZERO old-role"):
        main()
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--command-seed", "732023"],
        ["--command-training-root", "unused"],
        ["--motor-own-hash", "sha256:unused"],
    ],
)
def test_native_command_pilot_partial_binding_rejected_before_model_import(monkeypatch, extra):
    from scripts.rsi_qualify_receiving_motor_native import main

    args = ["test"]
    for key in (
        "experience-root",
        "own-root",
        "training-root",
        "factual-source-root",
        "factual-own-root",
        "initialization-root",
        "output",
        "own-hash",
        "training-hash",
        "factual-own-hash",
        "initialization-hash",
        "expected-head",
    ):
        args += ["--" + key, "unused"]
    monkeypatch.setattr(sys, "argv", args + extra)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
