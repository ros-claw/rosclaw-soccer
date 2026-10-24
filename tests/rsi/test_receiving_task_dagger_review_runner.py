"""Synthetic disk protocol checks; no native world, policy, or training executed."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_receiving_command_comparison import pair


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def publish(root, report):
    root.mkdir()
    p = root / "complete.json"
    p.write_text(json.dumps(report))
    return digest(p)


@pytest.mark.parametrize("fault", [None, "lineage", "checkpoint", "own", "initial", "repeat"])
def test_once_only_readback_all_three_then_no_simulator(tmp_path, monkeypatch, fault):
    from scripts import rsi_review_receiving_task_dagger as runner

    proot, croot = tmp_path / "parent-training", tmp_path / "child-training"
    proot.mkdir()
    croot.mkdir()
    pcs, ccs = [], []
    for seed in (732023, 732024, 732025):
        pc, cc = proot / f"cp{seed}.pt", croot / f"cp{seed}.pt"
        pc.write_bytes(f"synthetic-parent-{seed}".encode())
        cc.write_bytes(f"synthetic-child-{seed}".encode())
        pcs.append(
            dict(
                seed=seed,
                optimizer_updates=400,
                changed_parameter_tensors=list(range(6)),
                checkpoint_path=str(pc),
                checkpoint_hash=digest(pc),
            )
        )
        ccs.append(
            dict(
                seed=seed,
                optimizer_updates=400,
                changed_parameter_tensors=list(range(7)),
                checkpoint_path=str(cc),
                checkpoint_hash=digest(cc),
                parent_checkpoint_hash=digest(pc),
                exact_parent_function_before_updates=True,
                cumulative_prior_steps=800,
                optimizer_parameter_steps=[800.0] * 6 + [400.0],
            )
        )
    pd = dict(
        status="RECEIVING_COMMAND_BC_THREE_SEEDS_TRAINED_PENDING_OWN_NATIVE",
        optimizer_updates=1200,
        new_physical_steps=0,
        candidates=pcs,
        input_file_hashes={},
    )
    pp = proot / "complete.json"
    pp.write_text(json.dumps(pd))
    ph = digest(pp)
    cd = dict(
        status="RECEIVING_TASK_COMMAND_DAGGER_THREE_SEEDS_PENDING_NATIVE",
        optimizer_updates=1200,
        new_physical_steps=0,
        candidates=ccs,
        parent_bc_hash=ph,
        input_file_hashes={},
    )
    if fault == "lineage":
        cd["parent_bc_hash"] = "sha256:" + "e" * 64
    cp = croot / "complete.json"
    cp.write_text(json.dumps(cd))
    ch = digest(cp)
    args = [
        "review",
        "--parent-training-root",
        str(proot),
        "--parent-training-hash",
        ph,
        "--child-training-root",
        str(croot),
        "--child-training-hash",
        ch,
        "--expected-head",
        "a" * 40,
        "--output",
        str(tmp_path / "output"),
    ]
    for seed in (732023, 732024, 732025):
        roots = []
        for task in (False, True):
            s, o = pair(task)
            sr, ownroot = tmp_path / f"source-{seed}-{task}", tmp_path / f"own-{seed}-{task}"
            sr.mkdir()
            for row in s["records"]:
                directory = sr / f"trial{row['index']}"
                directory.mkdir()
                p = directory / "initial.npz"
                cluster = row["index"] // 6 + (row["index"] >= 10)
                p.write_bytes(f"synthetic-not-physics-initial-{cluster}".encode())
                row["artifacts"]["initial"] = dict(path=str(p.relative_to(sr)), hash=digest(p))
            for d in (s, o):
                d["command_seed"] = seed
                d["command_training_hash"] = ch if task else ph
                d["input_file_hashes"] = {}
            if fault == "initial" and task:
                s["records"][0]["artifacts"]["initial"]["hash"] = "sha256:" + "d" * 64
            o["records"] = [{k: v for k, v in r.items() if k != "artifacts"} for r in s["records"]]
            sp = sr / "complete.json"
            sp.write_text(json.dumps(s))
            o["source_hash"] = digest(sp)
            if fault == "own" and task:
                o["records"][0]["sampling_seed"] += 1
            oh = publish(ownroot, o)
            roots.extend([sr, ownroot, oh])
        args.extend(
            [
                "--cohort",
                str(seed),
                str(roots[0]),
                str(roots[1]),
                roots[2],
                str(roots[3]),
                str(roots[4]),
            ]
        )
    if fault == "checkpoint":
        (croot / "cp732023.pt").write_bytes(b"tampered")
    if fault == "repeat":
        (tmp_path / "output").mkdir()
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner, "clean", lambda _: None)
    monkeypatch.setattr(runner, "source_pins", lambda _: None)

    def check(pins):
        assert all(digest(Path(p)) == h for p, h in pins.items())

    monkeypatch.setattr(runner, "check_pins", check)
    if fault:
        with pytest.raises(ValueError):
            runner.main()
        assert not (tmp_path / "output" / "complete.json").exists()
    else:
        runner.main()
        out = json.loads((tmp_path / "output" / "complete.json").read_text())
        assert len(out["effects"]) == 3
        assert out["measured_prior_and_current_native_steps"] == 46680 * 12
        assert out["physical_steps_executed_by_review"] == 0
        assert out["all_three_local_precision_gain_without_loss"] is False
        assert out["goal_complete"] is False
