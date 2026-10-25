import json

import numpy as np
import pytest

from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_stable_plastic_sources import (
    admit_development_gates,
    compare_champion,
    merge_pins,
)


def test_pins_never_silently_replace_an_earlier_qualified_input():
    pins = {"model": "old"}
    with pytest.raises(ValueError, match="conflicting"):
        merge_pins(pins, {"model": "new"})
    assert pins == {"model": "old"}


@pytest.mark.parametrize(
    "mode", ["qualified", "no_gain", "31passes", "lost_old", "missing", "wrong_model", "reuse_exam"]
)
def test_new_local_exam_requires_complete_verified_gains_on_both_consumed_banks(tmp_path, mode):
    paths, hashes = [], []
    for perturbed in (False, True):
        report = dict(
            status="MOTION_SYNERGY_TEACHER_RAW_CEM_FUTURES_AND_FULL_EXAM_VERIFIED",
            stable_plastic_development=True,
            expanded_public_validation=True,
            perturbed_public_validation=perturbed,
            new_physical_steps=0,
            actual_worlds=80,
            regressions=[],
            champion_regressions=[30] if mode == "lost_old" else [],
            champion_rescues=[] if mode == "no_gain" or not perturbed else [22],
            results=[
                dict(
                    course_index=i,
                    C=dict(task_pass=i < (31 if mode == "31passes" else 32)),
                    C_healthy=True,
                )
                for i in range(39 if mode == "missing" else 40)
            ],
            input_file_hashes={"model": "wrong" if mode == "wrong_model" else "exact-model"},
            independent_local_validation=mode == "reuse_exam",
        )
        path = tmp_path / f"{perturbed}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
        hashes.append(digest(path))
    pins = {}
    if mode == "qualified":
        result = admit_development_gates(paths, hashes, "exact-model", pins)
        assert result[1]["champion_rescues"] == [22]
        assert pins[str(paths[0])] == hashes[0]
    else:
        with pytest.raises(ValueError, match="verified new motor gain|development32/40"):
            admit_development_gates(paths, hashes, "exact-model", pins)


def test_new_exam_cannot_start_without_explicit_dual_gates():
    with pytest.raises(ValueError, match="two independently"):
        admit_development_gates(None, None, "model", {})


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
