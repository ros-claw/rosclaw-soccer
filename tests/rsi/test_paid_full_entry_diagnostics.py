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
