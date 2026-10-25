from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_mechanism_expansion import validate_public_course
from rosclaw_soccer.rsi.synergy_teacher_perturbations import declare_synergy_perturbations
from rosclaw_soccer.rsi.walk_kick_hybrid_teacher import expanded_validation_courses


def test_complete_disjoint_balanced_bank_is_deterministic_and_locally_bounded():
    original = expanded_validation_courses()
    first, report = declare_synergy_perturbations(original, seed=2026102519)
    second, replay = declare_synergy_perturbations(original, seed=2026102519)
    assert first == second and report == replay
    assert len(set(c.course for c in first)) == 40
    assert not set(c.course for c in first) & set(c.course for c in original)
    assert not report["private_fresh_accessed"] and report["all_courses_preflighted"]
    for parent, new in zip(original, first, strict=True):
        assert validate_public_course(parent.course) == validate_public_course(new.course)
        assert (abs(np.array(parent.course) - np.array(new.course)) <= [0.005, 0.01, 0.025]).all()


def test_invalid_entire_source_bank_rejected_before_random_generator(monkeypatch):
    original = list(expanded_validation_courses())
    original[-1] = replace(original[-1], course=(2.7, 0.161, 0.5))

    def unexpected(*args):
        raise AssertionError("must reject invalid whole declaration before sampling")

    monkeypatch.setattr(np.random, "default_rng", unexpected)
    with pytest.raises(ValueError, match="entire public"):
        declare_synergy_perturbations(tuple(original), seed=1)


def test_out_of_domain_generated_row_is_not_clipped_or_resampled(monkeypatch):
    original = expanded_validation_courses()

    class BadRandom:
        def uniform(self, lo, hi):
            return hi + 1

    monkeypatch.setattr(np.random, "default_rng", lambda seed: BadRandom())
    with pytest.raises(ValueError, match="entire public"):
        declare_synergy_perturbations(original, seed=1)


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError, match="uint32"):
        declare_synergy_perturbations(expanded_validation_courses(), seed=seed)


@pytest.mark.parametrize(
    "flags,reason",
    [
        (["--interface", "motion-synergy-full-window", "--perturb-40"], "local robustness"),
        (["--interface", "measured-pose", "--expand-40"], "full-window"),
    ],
)
def test_runner_rejects_unqualified_protocol_combinations_before_source_or_physics(
    monkeypatch, flags, reason
):
    import sys

    from scripts.rsi_run_task_space_kick_pilot import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            "--qualification",
            "unused",
            "--output",
            "unused",
            "--qualification-hash",
            "unused",
            "--expected-head",
            "unused",
            "--shard-root",
            "a",
            "b",
            "c",
            "d",
            *flags,
        ],
    )
    with pytest.raises(ValueError, match=reason):
        main()
