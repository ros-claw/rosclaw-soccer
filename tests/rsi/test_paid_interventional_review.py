"""Review orchestration tests with synthetic receipts, not physical evidence."""

import hashlib
import json
import sys

import pytest

from rosclaw_soccer.rsi.paid_interventional_student import learner_config, learner_status
from scripts import rsi_review_paid_full_entry as runner


def save(root, name, value):
    root.mkdir(parents=True, exist_ok=True)
    p = root / name
    p.write_text(json.dumps(value))
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def receipt(**kw):
    return dict(hardware_authorized=False, promotion_authorized=False, input_file_hashes={}, **kw)


def worlds(n, *, failed=False):
    return [
        dict(case=i, stable=True, quality_by_kick=[not (failed and i == 4), True, True])
        for i in range(n)
    ]


@pytest.mark.parametrize("failed_seed", [None, 0, 1, 2])
@pytest.mark.parametrize("generation", [1, 2, 3])
def test_all_three_seeds_and_hard_first_cases_remain_required(
    tmp_path, monkeypatch, failed_seed, generation
):
    reference = tmp_path / "reference"
    pins, ref_roots = {}, {}
    for bank, n in (("train20", 20), ("train16", 16), ("dev16", 16)):
        root = tmp_path / ("reference-" + bank)
        pins[str(root / "complete.json")] = save(root, "complete.json", dict(worlds=worlds(n)))
        ref_roots[bank] = root
    # The reviewer derives the two original bank paths from one pipeline root.
    pipeline = tmp_path / "pipeline"
    for child in ("train-exam", "dev-exam"):
        root = pipeline / child
        pins[str(root / "complete.json")] = save(root, "complete.json", dict(worlds=worlds(16)))
    save(
        reference,
        "declaration.json",
        dict(new_exams=[str(ref_roots["train20"])], pipelines=[str(pipeline)]),
    )
    reference_hash = save(
        reference,
        "complete.json",
        dict(
            **receipt(
                status="ROUTER_FRONTIER_PASS_PENDING_NEW_FRESH",
                all_three_seeds_pass=True,
                primary_seed=0,
            ),
        )
        | {"input_file_hashes": pins},
    )

    def training(root, seed, control=False):
        return save(
            root,
            "complete.json",
            receipt(
                source_head="producer",
                data_hash="data",
                warm_hash="warm",
                body_scene_hash="body",
                parent_hash="parent",
                status=learner_status(generation, control),
                config=learner_config(seed, control=control, generation=generation),
                counts=dict(optimizer_updates=0 if control else 2048),
            ),
        )

    def bank(exam, audit, student_hash, name, *, failed=False):
        n = 20 if name == "train20" else 16
        original = n == 16
        decl_hash = save(
            exam,
            "declaration.json",
            dict(
                source_head="producer",
                student_hash=student_hash,
                first_enabled=True,
                schedule=["student"] * 3,
            ),
        )
        summary = dict(synthetic=True)
        rh = save(
            exam,
            "complete.json",
            receipt(
                status="PAID_FULL_ENTRY_ORIGINAL16_WORLDS_CLOSED_PENDING_AUDIT"
                if original
                else "PAID_ROUTER_TRAIN20_WORLDS_CLOSED_PENDING_AUDIT",
                source_head="producer",
                declaration_hash=decl_hash,
                student_hash=student_hash,
                summary=summary,
                bank="dev" if name == "dev16" else "train",
                worlds=worlds(n, failed=failed),
                actual_native_steps=100,
                actual_neural_rows=10,
            ),
        )
        save(
            audit,
            "complete.json",
            receipt(
                status="PAID_FULL_ENTRY_ORIGINAL16_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS"
                if original
                else "PAID_ROUTER_TRAIN20_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS",
                source_head="producer",
                source_hash=rh,
                student_hash=student_hash,
                counts=dict(worlds=2 * n),
                own_native_steps=100,
                own_trajectory_neural_rows=10,
                first_kick_replayed=True,
                summary=summary,
            ),
        )

    for seed in range(3):
        th = training(tmp_path / f"family-s{seed}-trained-20261022", seed)
        for name in ("train20", "train16", "dev16"):
            bank(
                tmp_path / f"family-s{seed}-{name}-exam-20261022",
                tmp_path / f"family-s{seed}-{name}-audit-20261022",
                th,
                name,
                failed=name == "train20" and seed == failed_seed,
            )
    ch = training(tmp_path / "family-geometry-control-20261022", 0, True)
    bank(
        tmp_path / "family-geometry-control-train20-exam-20261022",
        tmp_path / "family-geometry-control-train20-audit-20261022",
        ch,
        "train20",
        failed=True,
    )
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda cmd, **kw: "frozen\n" if "rev-parse" in cmd else b"",
    )
    output = tmp_path / "review"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--family-root",
            str(tmp_path),
            "--family-prefix",
            "family",
            "--stamp",
            "20261022",
            "--reference-root",
            str(reference),
            "--reference-hash",
            reference_hash,
            "--producer-head",
            "producer",
            "--expected-head",
            "frozen",
            "--interventional",
            "--generation",
            str(generation),
            "--output",
            str(output),
        ],
    )
    runner.main()
    result = json.loads((output / "complete.json").read_text())
    assert result["all_three_seeds_pass"] is (failed_seed is None)
    assert not result["promotion_authorized"] and not result["goal_complete"]
    if failed_seed is None:
        assert result["next_action"] == "DECLARE_NEW_FRESH_V3_FOR_FIXED_SEED0"
        assert all(
            s["same_geometry_parameter_contrast"]["quality_gains"] == [[4, 0]]
            for s in result["seeds"]
        )
    else:
        assert result["status"] == "FULL_ENTRY_FRONTIER_REJECTED_KEEP_ROUTER"
