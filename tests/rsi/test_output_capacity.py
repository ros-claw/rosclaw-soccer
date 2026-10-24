import errno
from types import SimpleNamespace

import pytest

from scripts.rsi_output_capacity import require_output_capacity


def test_existing_ancestor_checked_before_output_creation(tmp_path, monkeypatch):
    observed = []

    def usage(path):
        observed.append(path)
        return SimpleNamespace(free=100)

    monkeypatch.setattr("scripts.rsi_output_capacity.shutil.disk_usage", usage)
    output = tmp_path / "not-created" / "trial"
    assert require_output_capacity(output, 100) == 100
    assert observed == [tmp_path] and not output.exists()


def test_enospc_before_any_writer_or_world(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.rsi_output_capacity.shutil.disk_usage", lambda _: SimpleNamespace(free=99)
    )
    with pytest.raises(OSError) as error:
        require_output_capacity(tmp_path / "new", 100)
    assert error.value.errno == errno.ENOSPC
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("reserve", [0, -1, True, 1.5])
def test_invalid_reserve_rejected(tmp_path, reserve):
    with pytest.raises(ValueError):
        require_output_capacity(tmp_path, reserve)
