import sys

import pytest

from scripts import rsi_continue_paid_canonical_memory as runner


def arguments(tmp_path):
    args = ["run", "--expected-head", "frozen", "--data-source-head", "data-head"]
    for name in ("data-root", "train20-plan", "train16-plan", "dev16-plan", "output"):
        args += ["--" + name, str(tmp_path / name)]
    for name in ("train20", "train16", "dev16"):
        args += ["--" + name + "-hash", "hash"]
    return args


@pytest.mark.parametrize("fault", ["dirty", "gpu", "threads"])
def test_invalid_queue_does_not_launch_children_or_create_output(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda args, **kw: (
            "frozen\n" if "rev-parse" in args else (b"dirty" if fault == "dirty" else b"")
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test" if fault == "gpu" else "")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(k, "2" if fault == "threads" else "1")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("child allocated"))
    with pytest.raises(ValueError):
        runner.main()
    assert not (tmp_path / "output").exists()
