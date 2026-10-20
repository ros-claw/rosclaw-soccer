import sys

import pytest

from scripts import rsi_train_motion_intention as train


def test_bad_source_cannot_allocate_motion_training(tmp_path, monkeypatch):
    args = ["train", "--seed", "17"]
    for key in ("bank", "smoke", "prior-cycle", "output"):
        args += ["--" + key, str(tmp_path / key)]
    for key in (
        "expected-head",
        "expected-bank-hash",
        "expected-smoke-hash",
        "expected-prior-cycle-hash",
    ):
        args += ["--" + key, "expected"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(train.subprocess, "check_output", lambda *a, **kw: "wrong\n")
    with pytest.raises(ValueError, match="clean frozen"):
        train.main()
    assert not (tmp_path / "output").exists()
