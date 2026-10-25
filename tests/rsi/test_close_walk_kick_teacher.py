import json

import pytest

from scripts.rsi_close_walk_kick_teacher import closed_shard_arguments, producer_finished


def roots(tmp_path):
    paths = []
    for i in range(4):
        p = tmp_path / str(i)
        p.mkdir()
        (p / "complete.json").write_text(
            json.dumps(
                dict(
                    status="PUBLIC16_TEACHER_CEM_SHARD_MEASURED_NOT_TRAINED",
                    source_head="frozen",
                    experimental_fast_approach=True,
                    worlds=76,
                    promotion_authorized=False,
                    results=[dict(course_index=k) for k in range(i, 16, 4)],
                )
            )
        )
        paths.append(p)
    return paths


def test_only_completed_exact_cohort_can_be_passed_to_audit(tmp_path):
    args = closed_shard_arguments(roots(tmp_path), "frozen")
    assert args.count("--shard") == 4
    assert sum(a.startswith("sha256:") for a in args) == 4


@pytest.mark.parametrize("defect", ["missing", "wronghead", "missingcourse", "duplicate"])
def test_bad_cohort_does_not_relaunch_or_silently_skip_physics(tmp_path, defect):
    paths = roots(tmp_path)
    if defect == "missing":
        paths[0] = tmp_path / "never-completed"
    elif defect == "wronghead":
        report = json.loads((paths[0] / "complete.json").read_text())
        report["source_head"] = "other"
        (paths[0] / "complete.json").write_text(json.dumps(report))
    elif defect == "missingcourse":
        report = json.loads((paths[0] / "complete.json").read_text())
        report["results"] = report["results"][:-1]
        (paths[0] / "complete.json").write_text(json.dumps(report))
    else:
        paths[1] = paths[0]
    with pytest.raises(ValueError):
        closed_shard_arguments(paths, "frozen")


def test_start_order_is_not_completion_of_simple_service():
    assert not producer_finished(
        dict(ActiveState="active", MainPID="123", Result="success", Job="")
    )
    assert not producer_finished(
        dict(ActiveState="inactive", MainPID="0", Result="success", Job="123")
    )
    assert producer_finished(dict(ActiveState="inactive", MainPID="0", Result="success", Job=""))
    with pytest.raises(ValueError, match="failed producer"):
        producer_finished(dict(ActiveState="failed", MainPID="0", Result="exit-code"))


def test_expanded_cohort_cannot_mix_banks_or_relax_baseline(tmp_path):
    paths = roots(tmp_path)
    for i, path in enumerate(paths):
        rp = path / "complete.json"
        r = json.loads(rp.read_text())
        r.update(
            status="PUBLIC40_TEACHER_CEM_SHARD_MEASURED_NOT_TRAINED",
            expanded_public_bank=True,
            baseline_uses_original_strict_codec=True,
            worlds=190,
            results=[dict(course_index=k) for k in range(i, 40, 4)],
        )
        rp.write_text(json.dumps(r))
    assert closed_shard_arguments(paths, "frozen").count("--shard") == 4
    rp = paths[0] / "complete.json"
    r = json.loads(rp.read_text())
    r["baseline_uses_original_strict_codec"] = False
    rp.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        closed_shard_arguments(paths, "frozen")
