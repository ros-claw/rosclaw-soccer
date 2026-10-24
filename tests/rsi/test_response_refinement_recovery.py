import json

import pytest

from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_response_refinement_recovery import admit_recovery, prefix


def fixture(tmp_path):
    root = tmp_path / "source"
    folder = root / "trial0"
    folder.mkdir(parents=True)
    artifacts = {}
    for name in ("initial", "physical", "A", "B"):
        path = folder / (name + ".npz")
        path.write_bytes(b"synthetic hash-only fixture, not physical evidence")
        artifacts[name] = dict(path=str(path.relative_to(root)), hash=digest(path))
    (root / "trial0.json").write_text(json.dumps(dict(index=0, artifacts=artifacts)))
    (root / "declaration.json").write_text(
        json.dumps(dict(source_head="old", model_hash="model", state_id=20))
    )
    item = dict(
        state_id=20,
        root=str(root),
        declaration_hash=digest(root / "declaration.json"),
        row_hashes=[digest(root / "trial0.json")],
    )
    seal = tmp_path / "seal.json"
    seal.write_text(
        json.dumps(
            dict(
                status="FOUR_STORAGE_INTERRUPTED_PREFIXES_SEALED_NOT_PHYSICS_PROOF",
                producer_head="old",
                shards=[item],
            )
        )
    )
    return root, seal


def test_preserves_whole_completed_prefix_and_hashes(tmp_path):
    root, seal = fixture(tmp_path)
    pins = {}
    r, rows, item = admit_recovery(seal, digest(seal), 20, "model", pins)
    assert r == root and len(rows) == 1
    assert item["row_hashes"] == [digest(root / "trial0.json")]
    assert len(pins) == 7  # four raw arrays, row, declaration, seal


@pytest.mark.parametrize(
    "defect", ["hole", "tamper", "escape", "symlink", "complete", "wrongmodel"]
)
def test_ambiguous_or_mutated_recovery_rejected(tmp_path, defect):
    root, seal = fixture(tmp_path)
    if defect == "hole":
        (root / "trial0.json").rename(root / "trial1.json")
    elif defect == "tamper":
        (root / "trial0" / "A.npz").write_bytes(b"changed")
    elif defect == "escape":
        row = json.loads((root / "trial0.json").read_text())
        row["artifacts"]["A"]["path"] = "../outside.npz"
        (root / "trial0.json").write_text(json.dumps(row))
    elif defect == "symlink":
        path = root / "trial0" / "A.npz"
        path.rename(tmp_path / "outside")
        path.symlink_to(tmp_path / "outside")
    elif defect == "complete":
        (root / "complete.json").write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError)):
        admit_recovery(seal, digest(seal), 20, "wrong" if defect == "wrongmodel" else "model", {})


def test_prefix_cannot_be_empty(tmp_path):
    with pytest.raises(ValueError):
        prefix(tmp_path)
