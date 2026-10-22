import pytest

from rosclaw_soccer.rsi.motor_exploration_curriculum import (
    declared_motor_curriculum,
    motor_curriculum,
)


@pytest.mark.parametrize("name,worlds", [("run2", 8), ("retention-run64", 64)])
def test_fixed_curriculum_and_disjoint_noise_generations(name, worlds):
    c = motor_curriculum(name)
    assert c.worlds == worlds
    assert len(set(c.seeds(0) + c.seeds(1))) == 2 * c.seeds_per_context
    assert c.context_ids == ((20, 21) if name == "run2" else (4, 5, 20, 21))
    d = dict(
        curriculum=name,
        protocol=c.protocol,
        seeds=c.seeds(1),
        generation_index=1,
        contexts=[{}] * len(c.context_ids),
        maximum_worlds=worlds,
        maximum_native_steps=4000 * worlds,
        maximum_neural_decisions=400 * worlds,
    )
    assert declared_motor_curriculum(d) == c
    d["seeds"] = d["seeds"][:-1]
    with pytest.raises(ValueError):
        declared_motor_curriculum(d)


def test_legacy_two_case_commitment_stays_eight_not_inflated():
    c = motor_curriculum("run2")
    d = dict(
        protocol=c.protocol,
        seeds=c.seeds(0),
        contexts=[{}, {}],
        maximum_worlds=8,
        maximum_native_steps=32000,
        maximum_neural_decisions=3200,
    )
    assert declared_motor_curriculum(d).worlds == 8
    d["maximum_worlds"] = 64
    with pytest.raises(ValueError):
        declared_motor_curriculum(d)


def test_unreviewed_context_or_negative_generation_rejected():
    with pytest.raises(ValueError):
        motor_curriculum("holdout")
    with pytest.raises(ValueError):
        motor_curriculum("run2").seeds(-1)


def test_four_workers_partition_full_course_without_duplicates_or_qualification_of_a_shard():
    c = motor_curriculum("retention-run64")
    full = c.shard_pairs(3, 0, 1)
    parts = [c.shard_pairs(3, i, 4) for i in range(4)]
    assert [len(x) for x in parts] == [16] * 4
    assert set().union(*(set(x) for x in parts)) == set(full)
    assert sum(len(x) for x in parts) == len(set(full)) == 64
    d = dict(
        curriculum=c.name,
        protocol=c.protocol,
        seeds=c.seeds(3),
        generation_index=3,
        contexts=[{}] * 4,
        maximum_worlds=64,
        maximum_native_steps=256000,
        maximum_neural_decisions=25600,
        shard_count=4,
        shard_slot=0,
    )
    with pytest.raises(ValueError):
        declared_motor_curriculum(d)
