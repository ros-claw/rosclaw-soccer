import sys

import pytest

from scripts import rsi_eval_motion_intention as evaluate


@pytest.mark.parametrize("mode", ["evaluate", "replay"])
def test_wrong_source_never_allocates_motion_exam_or_replay(tmp_path, monkeypatch, mode):
    args = ["exam", "--mode", mode]
    for key in ("train-root", "bank", "world", "model-root", "core-source", "output"):
        args += ["--" + key, str(tmp_path / key)]
    for key in (
        "expected-head",
        "train-unit",
        "expected-training-hash",
        "expected-training-source",
        "expected-bank-hash",
    ):
        args += ["--" + key, "declared"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(evaluate.subprocess, "check_output", lambda *a, **kw: "wrong\n")
    with pytest.raises(ValueError, match="clean frozen"):
        evaluate.main()
    assert not (tmp_path / "output").exists()
