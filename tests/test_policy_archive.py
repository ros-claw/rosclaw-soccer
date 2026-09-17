from pathlib import Path

import pytest

from rosclaw_soccer.training.policy_archive import file_digest, preserve_policy_assets


def entry(path, name="policy"):
    return dict(name=name, path=path, roles=["playmaker"], scope="fixture only")


def test_independent_deduplicated_copies_do_not_promote(tmp_path):
    source = tmp_path / "source.npz"
    source.write_bytes(b"fixture bytes; not a loaded model")
    output = tmp_path / "archive"
    report = preserve_policy_assets(
        [entry(source), entry(source, "alias")], output=output, maximum_bytes=100
    )
    assert report["unique_objects"] == 1
    assert not report["composition_qualified"] and not report["promoted"]
    source.write_bytes(b"changed")
    preserved = output / report["entries"][0]["archive_path"]
    assert file_digest(preserved) == report["entries"][0]["file_hash"]
    with pytest.raises(FileExistsError):
        preserve_policy_assets([entry(source)], output=output, maximum_bytes=100)


@pytest.mark.parametrize("fault", ["hash", "budget", "role", "scope", "duplicate"])
def test_invalid_archive_never_writes_manifest(tmp_path: Path, fault):
    source = tmp_path / "source"
    source.write_bytes(b"12345")
    values = [entry(source)]
    if fault == "hash":
        values[0]["expected_hash"] = "sha256:" + "0" * 64
    if fault == "role":
        values[0]["roles"] = ["invented"]
    if fault == "scope":
        values[0]["scope"] = ""
    if fault == "duplicate":
        values *= 2
    with pytest.raises(ValueError):
        preserve_policy_assets(
            values, output=tmp_path / "archive", maximum_bytes=1 if fault == "budget" else 100
        )
    assert not (tmp_path / "archive").exists()
