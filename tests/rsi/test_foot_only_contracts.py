import copy

import pytest

from rosclaw_soccer.rsi.foot_only_contracts import (
    require_foot_only_development,
    require_foot_only_learning,
)


def learner(warmup=False):
    return dict(
        status="FOOT_ONLY_VALUE_WARMUP_PPO_NOT_SKILL_QUALIFIED"
        if warmup
        else "FOOT_ONLY_PPO_PILOT_NOT_SKILL_QUALIFIED",
        gpu_world_steps=245760 if warmup else 122880,
        iterations=32 if warmup else 16,
        worlds=32,
        seed=4601,
        new_native_steps=0,
        optimizer_counts=dict(attempted=656 if warmup else 336, completed=656 if warmup else 336),
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
        value_warmup_boundary=dict(
            actor_bitexact=True, critic_changed=True, optimizer_updates=320, physical_substeps=3840
        ),
    )


@pytest.mark.parametrize("warmup", [False, True])
def test_explicit_completed_new_intent_only(warmup):
    d = learner(warmup)
    require_foot_only_learning(d)
    for key, bad in (
        ("status", "BALL_AWARE_GPU_PPO_COURSE_NOT_SKILL_QUALIFIED"),
        ("gpu_world_steps", 0),
        ("worlds", True),
        ("hardware_authorized", True),
        ("goal_complete", True),
    ):
        wrong = {**d, key: bad}
        with pytest.raises(ValueError):
            require_foot_only_learning(wrong)


def test_warmup_boundary_cannot_be_assumed():
    d = learner(True)
    for key, bad in (
        ("actor_bitexact", False),
        ("critic_changed", False),
        ("optimizer_updates", 319),
        ("physical_substeps", 3830),
    ):
        wrong = copy.deepcopy(d)
        wrong["value_warmup_boundary"][key] = bad
        with pytest.raises(ValueError, match="boundary"):
            require_foot_only_learning(wrong)


def development():
    return dict(
        status="FOOT_ONLY_PAIRED_PUBLIC_DEV_NOT_SKILL_QUALIFIED",
        all_initial_states_paired_exact=True,
        known_public_dev_not_blind_fresh=True,
        optimizer_updates=0,
        hardware_authorized=False,
        input_file_hashes={"receipt": "sha256:example"},
        results=[
            dict(source=f"soccer-standard-{s}", arm=a, completed_count=32)
            for s in ("007_left.npz", "010_right.npz")
            for a in ("initial", "final")
        ],
    )


def test_own_dev_binding_all_sources_and_pairing_are_required():
    d = development()
    require_foot_only_development(d, learner_path="receipt", learner_hash="sha256:example")
    for key, bad in (
        ("all_initial_states_paired_exact", False),
        ("results", d["results"][:2]),
        ("input_file_hashes", {"other": "sha256:example"}),
        ("known_public_dev_not_blind_fresh", False),
    ):
        with pytest.raises(ValueError):
            require_foot_only_development(
                {**d, key: bad}, learner_path="receipt", learner_hash="sha256:example"
            )
