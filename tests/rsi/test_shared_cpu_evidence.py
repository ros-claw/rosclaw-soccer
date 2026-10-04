from concurrent.futures import ThreadPoolExecutor

import mujoco
import pytest

from rosclaw_soccer.rsi.shared_cpu_evidence import save_shared_compiled_model
from rosclaw_soccer.sim.contracts import hash_bytes


def model():
    return mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/>"
        '<geom type="sphere" size=".1"/></body></worldbody></mujoco>'
    )


def test_native_binary_bytes_exact_shared_readonly_and_loadable(tmp_path):
    native = model()
    ordinary = tmp_path / "ordinary.mjb"
    mujoco.mj_saveModel(native, str(ordinary), None)
    before = ordinary.read_bytes()
    targets = []
    for i in range(4):
        folder = tmp_path / f"case{i}"
        folder.mkdir()
        targets.append(folder / "compiled_model.mjb")
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = list(pool.map(lambda p: save_shared_compiled_model(native, p), targets))
    assert all(h == hash_bytes(before) for h in hashes)
    assert all(p.read_bytes() == before for p in targets)
    assert len({p.stat().st_ino for p in targets}) == 1
    assert all(p.stat().st_mode & 0o222 == 0 for p in targets)
    assert len(list((tmp_path / ".shared-worlds").iterdir())) == 1
    assert ordinary.read_bytes() == before
    loaded = mujoco.MjModel.from_binary_path(str(targets[0]))
    assert loaded.nq == native.nq and loaded.nv == native.nv


def test_existing_snapshot_never_replaced(tmp_path):
    folder = tmp_path / "case"
    folder.mkdir()
    target = folder / "compiled_model.mjb"
    target.write_bytes(b"user-owned")
    with pytest.raises(ValueError, match="new local"):
        save_shared_compiled_model(model(), target)
    assert target.read_bytes() == b"user-owned"
    assert not (tmp_path / ".shared-worlds").exists()


def test_oversize_model_rejected_before_serialization_or_store_creation(tmp_path, monkeypatch):
    folder = tmp_path / "case"
    folder.mkdir()
    native = model()
    monkeypatch.setattr(mujoco, "mj_sizeModel", lambda _: 256 * 1024**2 + 1)
    monkeypatch.setattr(
        mujoco, "mj_saveModel", lambda *_: pytest.fail("must reject before writing")
    )
    with pytest.raises(ValueError, match="serialization budget"):
        save_shared_compiled_model(native, folder / "compiled_model.mjb")
    assert not (tmp_path / ".shared-worlds").exists()
