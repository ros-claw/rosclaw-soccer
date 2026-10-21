import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_intervention_replan import (
    elite_update,
    replan_config,
    sample_latents,
)


def test_finite_protocol_is_current_student_prefix_not_fresh_or_sgd():
    c = replan_config(104, 289)
    assert c["switch_controls"] == [70, 80]
    assert c["max_episodes"] == 2 * (4 + 2 * 12 * 2 + 2 * 2)
    assert c["max_native_steps"] == c["max_episodes"] * (289 + 100) * 10
    assert c["neural_optimizer_updates"] == c["Fresh_rows_read"] == c["DEV_fit_rows"] == 0
    with pytest.raises(ValueError):
        replan_config(105, 289)


def test_cem_keeps_incumbent_bounded_and_updates_from_actual_elites():
    seed = replan_config(104, 289)["seed"]
    incumbent = np.full((2, 6), 0.2)
    mean, std = np.zeros((2, 6)), np.full((2, 6), 0.7)
    a, b = sample_latents(np.random.default_rng(seed), mean, std, incumbent)
    x, y = sample_latents(np.random.default_rng(seed), mean, std, incumbent)
    assert np.array_equal(a, x) and np.array_equal(b, y)
    assert np.array_equal(b[0], incumbent) and (np.abs(b) <= 1).all()
    ranks = [(int(i == 9), i) for i in range(12)]
    new_mean, new_std, ids = elite_update(a, ranks)
    assert ids == [9, 11, 10, 8]
    assert np.array_equal(new_mean, a[ids].mean(axis=0))
    assert (new_std >= 0.1).all()
    with pytest.raises(ValueError):
        elite_update(a, ranks[:-1])
    with pytest.raises(ValueError):
        sample_latents(np.random.default_rng(seed), mean, -std, incumbent)


@pytest.mark.parametrize("audit", [False, True])
def test_dirty_replan_or_auditor_never_imports_numerics_or_allocates_physics(
    tmp_path, monkeypatch, audit
):
    import sys

    from scripts import rsi_audit_paid_intervention_replan, rsi_replan_paid_interventions

    runner = rsi_audit_paid_intervention_replan if audit else rsi_replan_paid_interventions
    args = ["run", "--expected-head", "frozen", "--output", str(tmp_path / "out")]
    if audit:
        args += ["--root", str(tmp_path), "--producer-head", "producer"]
    else:
        for name in (
            "teacher",
            "teacher-audit",
            "student",
            "student-exam",
            "student-audit",
            "diagnosis",
        ):
            args += ["--" + name + "-root", str(tmp_path), "--" + name + "-hash", "hash"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: b"dirty")
    with pytest.raises(ValueError, match="clean frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()
