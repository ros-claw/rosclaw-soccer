import hashlib

import pytest

from scripts.rsi_legacy_relative_source_pins import resolve_source_pins


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def setup(tmp_path, monkeypatch):
    root = tmp_path / "old"
    (root / "src/pkg").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "src/pkg/a.py").write_bytes(b"old source")
    script = root / "scripts/rsi_audit_sagittal_reference.py"
    script.write_bytes(b"pinned bootstrap")
    monkeypatch.setattr(
        "scripts.rsi_legacy_relative_source_pins.subprocess.check_output",
        lambda *a, **k: "old-head\n",
    )
    return root, {"src/pkg/a.py": digest(b"old source"), str(script): digest(b"pinned bootstrap")}


def test_legacy_source_bound_to_original_bootstrap_not_current_cwd(tmp_path, monkeypatch):
    root, pins = setup(tmp_path, monkeypatch)
    original = dict(pins)
    result, mapping = resolve_source_pins(pins, root, "old-head")
    assert pins == original and mapping == {"src/pkg/a.py": str(root / "src/pkg/a.py")}
    assert result[str(root / "src/pkg/a.py")] == digest(b"old source")
    assert all(__import__("pathlib").Path(p).is_absolute() for p in result)


@pytest.mark.parametrize("bad", ["missing", "head", "bootstrap", "bytes", "escape"])
def test_rebinding_never_weakens_byte_identity(tmp_path, monkeypatch, bad):
    root, pins = setup(tmp_path, monkeypatch)
    head = "old-head"
    if bad == "missing":
        root = None
    elif bad == "head":
        head = "new-head"
    elif bad == "bootstrap":
        pins[str(root / "scripts/rsi_audit_sagittal_reference.py")] = digest(b"other")
    elif bad == "bytes":
        pins["src/pkg/a.py"] = digest(b"new source")
    else:
        pins["src/../elsewhere.py"] = digest(b"elsewhere")
    with pytest.raises(ValueError):
        resolve_source_pins(pins, root, head)


def test_absolute_only_contract_needs_no_legacy_root(tmp_path):
    pins = {str(tmp_path / "source.py"): digest(b"x")}
    assert resolve_source_pins(pins, None, None) == (pins, {})
