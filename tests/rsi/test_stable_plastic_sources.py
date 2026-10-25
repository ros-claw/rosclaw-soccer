import json

import numpy as np
import pytest

from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_stable_plastic_sources import compare_champion, merge_pins


def test_pins_never_silently_replace_an_earlier_qualified_input():
    pins = {"model": "old"}
    with pytest.raises(ValueError, match="conflicting"):
        merge_pins(pins, {"model": "new"})
    assert pins == {"model": "old"}


@pytest.mark.parametrize("mode", ["exact", "rescue", "loss", "drift", "course", "forged"])
def test_retention_compares_old_champion_not_only_weaker_baseline(tmp_path, mode):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    before, after = old / "course-00", new / "course-00"
    before.mkdir()
    after.mkdir()
    np.savez(before / "C.npz", qpos=np.zeros((10, 7)))
    np.savez(after / "C.npz", qpos=np.ones((10, 7)) if mode == "drift" else np.zeros((10, 7)))
    row = dict(
        course=[2.3, 0.1, 0.4],
        parameters={},
        experimental_fast_approach=False,
        entry_control=5,
        changed_controls=0 if mode == "rescue" else 1,
        healthy_full_horizon=True,
        outcome=dict(task_pass=mode != "rescue", safe=True),
        artifact=dict(path="C.npz", hash=digest(before / "C.npz")),
    )
    (before / "C.json").write_text(json.dumps(row))
    current = dict(row)
    current["outcome"] = dict(task_pass=mode != "loss", safe=True)
    current["controller_events"] = [
        dict(selected_basis="plastic" if mode == "rescue" else "backbone")
    ]
    if mode == "course":
        current["course"] = [2.31, 0.1, 0.4]
    (after / "C.json").write_text(json.dumps(current))
    cp = old / "complete.json"
    reference = dict(record="course-00/C.json", hash=digest(before / "C.json"))
    cp.write_text(json.dumps(dict(results=[dict(arms=dict(C=reference))])))
    pins = {str(p): digest(p) for p in (cp, before / "C.json", before / "C.npz")}
    if mode == "forged":
        (before / "C.npz").write_bytes(b"changed after qualification")
    if mode in ("drift", "course", "forged"):
        with pytest.raises(ValueError, match="drifted|same actual course|changed"):
            compare_champion(new, old, pins)
    else:
        losses, rescues = compare_champion(new, old, pins)
        assert losses == ([0] if mode == "loss" else [])
        assert rescues == ([0] if mode == "rescue" else [])
