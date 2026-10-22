import copy

import pytest

from rosclaw_soccer.rsi.paid_growth_probe import prepare_probe, probe_cases, probe_config


def test_continuous_probe_is_deterministic_balanced_new_and_never_resampled():
    from test_paid_neighborhood_study import sources

    from rosclaw_soccer.rsi.paid_neighborhood_study import contact_neighborhood_cases

    original = sources()
    before = copy.deepcopy(original)
    rows = probe_cases(original)
    assert rows == probe_cases(original) and original == before
    assert len(rows) == 32 and [r["case"] for r in rows] == list(range(32))
    assert [r["source_case"] for r in rows] == [4] * 16 + [5] * 16

    def keys(rows):
        return {(tuple(r["ball_world_initial"]), tuple(r["velocity_world_initial"])) for r in rows}

    assert len(keys(rows)) == 32 and not keys(rows).intersection(
        keys(contact_neighborhood_cases(original))
    )
    for row in rows:
        dx, dy, dvx, dvy = row["perturbation"]
        assert abs(dx) <= 0.03 and abs(dy) <= 0.03 and abs(dvx) <= 0.06 and abs(dvy) <= 0.06
        assert row["bank"] == "unseen_local_probe_not_Fresh" and row["not_fresh"]


def test_fixed_pair_hashes_are_in_config_and_not_mislabelled_fresh():
    hashes = {"learned": "sha256:" + "a" * 64, "zero-control": "sha256:" + "b" * 64}
    cfg = probe_config(hashes)
    assert cfg["locked_candidate_hashes"] == hashes and cfg["maximum_episodes"] == 128
    assert (
        cfg["maximum_native_steps"] == 128 * 3890
        and cfg["maximum_native_steps_per_arm"] == 64 * 3890
    )
    assert not cfg["consumed_for_training"] and not cfg["formal_Fresh_claim"]
    hashes["learned"] = "changed"
    assert cfg["locked_candidate_hashes"]["learned"] == "sha256:" + "a" * 64
    with pytest.raises(ValueError):
        probe_config(hashes)


def test_nested_probe_is_rejected_before_recursing_or_allocating(tmp_path):
    import json

    (tmp_path / "complete.json").write_text(
        json.dumps(dict(status="PAID_GROWTH_UNSEEN32_PROBE_PREREGISTERED_NO_ROLLOUT"))
    )
    with pytest.raises(ValueError, match="nested probe"):
        prepare_probe(dict(probe_base_root=str(tmp_path)))


@pytest.mark.parametrize("dirty", [True, False])
def test_invalid_probe_declaration_has_no_output_or_physics(tmp_path, monkeypatch, dirty):
    import sys

    from scripts import rsi_declare_paid_neighborhood as runner

    args = [
        "run",
        "--probe-base-root",
        str(tmp_path),
        "--expected-head",
        "frozen",
        "--output",
        str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda args, **k: "frozen\n" if "rev-parse" in args else (b"dirty" if dirty else b""),
    )
    with pytest.raises(ValueError):
        runner.main()
    assert not (tmp_path / "out").exists()
