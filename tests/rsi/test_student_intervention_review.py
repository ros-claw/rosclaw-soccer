"""Mock all-veto review accounting, not physical or neural recovery evidence."""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest


def fixture(tmp_path, monkeypatch, *, bad_charge=False, bad_sum=False):
    import mujoco

    from scripts import rsi_probe_student_intervention as probe
    from scripts.rsi_review_student_intervention import review_raw

    monkeypatch.setattr(probe, "validate_warm_pair", lambda *a: None)
    names = [
        s + t for s in ("left", "right") for t in ("_ankle_roll_link", "_ankle_pitch_link")
    ] + ["pelvis"]
    model = NS(
        nbody=6, body_rootid=np.array([0, 5, 5, 5, 5, 5]), body=lambda n: NS(id=names.index(n) + 1)
    )
    monkeypatch.setattr(mujoco.MjModel, "from_binary_path", lambda p: model)
    world = tmp_path / "world.mjb"
    world.write_bytes(b"fixture only not physical")
    q = np.zeros((30, 10, 43))
    q[:, :, 2], q[:, :, 3] = 0.4, 1.0
    prefix = dict(
        target_body_canonical_qpos_after_substep=q,
        ball_contact_force_by_body_n=np.zeros((30, 10, 6)),
    )
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix

    diag = diagnose_entry_prefix(model, prefix, frames=30)
    warm = tmp_path / "warm"
    warm.mkdir()
    collected = {}
    for part in range(8):
        rows = []
        for i in list(range(24))[part::8]:
            row = dict(
                status="PREINTERVENTION_HISTORY_VETO",
                course_index=i,
                prefix_diagnostics=diag,
                causal_qualified=False,
                candidates=[],
                selected=None,
                forecast_native_steps=0,
                executed_native_steps=300,
                new_native_steps=27300 if bad_charge and i == 0 else 300,
                denominator_retained=True,
            )
            rows.append(row)
            wd = warm / f"course{i:03d}"
            wd.mkdir()
            np.savez_compressed(wd / "actual-student30.npz", **prefix)
            np.savez_compressed(wd / "shadow-labels30.npz", fixture=np.zeros((30, 1), np.float32))
        for repeated in (False, True):
            root = tmp_path / f"part{part}-{repeated}"
            root.mkdir()
            for row in rows:
                folder = root / f"course{row['course_index']:03d}"
                folder.mkdir()
                (folder / "complete.json").write_text(json.dumps(row))
                np.savez_compressed(folder / "actual-student30.npz", **prefix)
                np.savez_compressed(
                    folder / "shadow-labels30.npz", fixture=np.zeros((30, 1), np.float32)
                )
            collected[part, repeated] = (
                root,
                dict(teacher_evidence_hash="fixture", runtime_file_hashes={str(world): "fixture"}),
                dict(
                    results=rows,
                    new_native_steps=901 if bad_sum else sum(r["new_native_steps"] for r in rows),
                ),
            )
    prior = dict(
        summary=dict(
            rows=[
                dict(
                    course_index=i,
                    stratum="fixture",
                    teacher_qualified=False,
                    actor_outcomes=dict(seed43=False),
                )
                for i in range(24)
            ]
        )
    )
    return review_raw(NS(warm=warm), dict(indices=list(range(24))), prior, collected, {})


def test_all_history_vetoes_keep_full24_and_charge_actual_warm_only(tmp_path, monkeypatch):
    result = fixture(tmp_path, monkeypatch)
    assert result["contexts"] == 24 and result["eligible_at_actual30"] == 0
    assert result["actual_new_native_steps_across_two_executions"] == 14400
    assert result["teacher_recoveries_from_student30"] == 0
    assert not result["learned_policy_growth_proven"]


def test_veto_cannot_manufacture_future_physics_charges(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="charge only warm"):
        fixture(tmp_path, monkeypatch, bad_charge=True)


def test_shard_total_cannot_diverge_from_raw_case_charges(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="accounting differs"):
        fixture(tmp_path, monkeypatch, bad_sum=True)
