import sys

import pytest

from scripts import rsi_run_dagger_cycle, rsi_train_dagger_motor


@pytest.mark.parametrize("cycle", [False, True])
def test_bad_source_rejected_before_wait_training_or_outputs(tmp_path, monkeypatch, cycle):
    module = rsi_run_dagger_cycle if cycle else rsi_train_dagger_motor
    args = ["runner"]
    for key in ("bank", "review", "parent-train", "output"):
        args += ["--" + key, str(tmp_path / key)]
    for key in (
        "expected-head",
        "expected-bank-hash",
        "expected-parent-training-hash",
        "expected-review-head",
        "review-unit",
    ):
        args += ["--" + key, "declared"]
    if cycle:
        args += ["--unit-prefix", "fixed-test-generation"]
        for key in ("world", "model-root", "core-source"):
            args += ["--" + key, str(tmp_path / key)]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "wrong-head\n")
    with pytest.raises(ValueError, match="frozen"):
        module.main()
    assert not (tmp_path / "output").exists()
