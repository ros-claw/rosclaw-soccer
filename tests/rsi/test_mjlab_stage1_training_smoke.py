import subprocess

import pytest

from scripts.rsi_mjlab_stage1_training_smoke import file_digest, tracked_reference


def test_reference_pin_rejects_dirty_and_wrong_commit(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "asset.py"
    source.write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    assert tracked_reference(tmp_path, head) == {str(source): file_digest(source)}
    with pytest.raises(ValueError, match="clean exact"):
        tracked_reference(tmp_path, "0" * 40)
    source.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="clean exact"):
        tracked_reference(tmp_path, head)
