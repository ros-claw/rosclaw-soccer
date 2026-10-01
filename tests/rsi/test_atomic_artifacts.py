import gzip
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts import rsi_atomic_artifacts as artifacts


def test_destination_appears_only_after_entire_json_is_written(monkeypatch, tmp_path):
    path = tmp_path / "model.json"
    started, release = threading.Event(), threading.Event()

    def delayed(value, stream, **kwargs):
        text = json.dumps(value, **kwargs)
        midpoint = len(text) // 2
        stream.write(text[:midpoint])
        stream.flush()
        started.set()
        assert release.wait(timeout=5)
        stream.write(text[midpoint:])

    monkeypatch.setattr(artifacts.json, "dump", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(artifacts.write_once, path, dict(weights=list(range(10000))))
        try:
            assert started.wait(timeout=5)
            assert not path.exists()
        finally:
            release.set()
        pending.result(timeout=5)
    assert json.loads(path.read_text()) == dict(weights=list(range(10000)))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.json"]


def test_concurrent_same_payload_is_idempotent_and_never_overwrites(tmp_path):
    path = tmp_path / "model.json"
    value = dict(weights=list(range(10000)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: artifacts.write_once(path, value), range(8)))
    original = path.read_bytes()
    artifacts.write_once(path, value)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="commitment differs"):
        artifacts.write_once(path, dict(weights=[999]))
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.json"]


def test_failed_write_cannot_publish_a_partial_document(monkeypatch, tmp_path):
    path = tmp_path / "model.json"

    def failed(value, stream, **kwargs):
        stream.write('{"unfinished":')
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(artifacts.json, "dump", failed)
    with pytest.raises(OSError, match="disk write"):
        artifacts.write_once(path, dict(ok=True))
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_nonfinite_payload_is_rejected_before_any_file_creation(tmp_path):
    with pytest.raises(ValueError):
        artifacts.write_once(tmp_path / "model.json", dict(value=float("nan")))
    assert list(tmp_path.iterdir()) == []


def test_gzip_json_is_lossless_atomic_and_idempotent(tmp_path):
    path = tmp_path / "model.json.gz"
    value = dict(weights=list(range(10000)), title="完整的模型")
    artifacts.write_once(path, value)
    before = path.read_bytes()
    assert artifacts.load_json_artifact(path) == value
    assert json.loads(gzip.decompress(before)) == value
    artifacts.write_once(path, value)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="commitment differs"):
        artifacts.write_once(path, dict(weights=[0]))
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.json.gz"]


def test_gzip_bytes_are_deterministic_across_publication_paths(tmp_path):
    first, second = tmp_path / "first.json.gz", tmp_path / "second.json.gz"
    for p in (first, second):
        artifacts.write_once(p, dict(weight=[0.1, 2.3]))
    assert first.read_bytes() == second.read_bytes()
