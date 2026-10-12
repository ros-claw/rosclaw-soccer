"""Public declarations bind whole banks, not private exam or physical truth."""

import argparse
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.sim.contracts import hash_json

MEAN = "sha256:" + "a" * 64


def plan():
    return PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), MEAN, 18)


def test_complete_balanced_bank_determinism_and_contract_ownership():
    p = plan()
    draws = p.draws()
    assert draws == plan().draws() and len(draws) == 160
    assert len({d.sampling_seed for d in draws}) == 160
    assert len({d.course for d in draws}) == 40
    assert {d.lane for d in draws} == set(range(16))
    counts = {(x, v): 0 for x in (False, True) for v in (False, True)}
    for i, draw in enumerate(draws):
        assert draw.group == i and draw.context_id == i // 4 and draw.sample_index == i % 4
        assert draw.course == sample_training_courses(draw.seed)[draw.lane]
        assert not 2.4 < draw.course[0] < 2.6
        counts[draw.course[0] >= 2.6, draw.course[2] > 0] += 1
        assert p.require_draw(group=i, seed=draw.seed, lane=draw.lane, course=draw.course) == draw
    assert set(counts.values()) == {40}
    contract = p.contract()
    seal = contract.pop("plan_hash")
    assert seal == hash_json(contract)
    contract["jobs"][0]["course"] = (9.0, 9.0, 9.0)
    contract["seeds"].clear()
    assert p.contract()["plan_hash"] == seal
    with pytest.raises(FrozenInstanceError):
        p.stream = 19
    other = replace(p, stream=19)
    assert {d.sampling_seed for d in draws}.isdisjoint(d.sampling_seed for d in other.draws())


@pytest.mark.parametrize("seeds", [[], (), (1, 1), (True,), (-1,), (2**32,), tuple(range(65))])
def test_bad_seed_declarations(seeds):
    with pytest.raises(ValueError):
        PublicFirstTouchTrainingPlan(seeds, MEAN, 18)


@pytest.mark.parametrize("value", [None, True, "sha256:" + "g" * 64, "a" * 64])
def test_bad_behavior_identity(value):
    with pytest.raises(ValueError):
        PublicFirstTouchTrainingPlan((1,), value, 18)


@pytest.mark.parametrize("value", [-1, 32, True, 18.0, None])
def test_bad_stream(value):
    with pytest.raises(ValueError):
        PublicFirstTouchTrainingPlan((1,), MEAN, value)


@pytest.mark.parametrize(
    "key,value",
    [
        ("group", None),
        ("group", True),
        ("group", -1),
        ("group", 160),
        ("seed", 1),
        ("lane", True),
        ("course", (2.45, 0.0, 0.5)),
    ],
)
def test_course_or_identity_substitution_rejected(key, value):
    p = plan()
    d = p.draws()[0]
    args = dict(group=0, seed=d.seed, lane=d.lane, course=d.course)
    args[key] = value
    with pytest.raises(ValueError):
        p.require_draw(**args)


def test_exact_sampling_view_binding():
    p = plan()
    draw = p.draws()[0]
    view = dict(
        schema="soccer.rsi.recurrent_motor_sampling.v1",
        seed=draw.sampling_seed,
        mean_model=dict(model_hash=MEAN),
    )
    assert p.require_sampling_view(draw, view) is None
    for bad in (
        None,
        dict(view, seed=True),
        dict(view, seed=draw.sampling_seed + 1),
        dict(view, schema="FRESH"),
        dict(view, mean_model=dict(model_hash="sha256:" + "b" * 64)),
    ):
        with pytest.raises(ValueError):
            p.require_sampling_view(draw, bad)


def test_public_partition_requires_plan_before_any_model_allocation(tmp_path):
    with pytest.raises(ValueError, match="public training declaration"):
        run_native_first_touch_episode(
            argparse.Namespace(),
            course=(2.2, 0.0, 0.4),
            partition="DECLARED_PUBLIC_TRAINING",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "change",
    [
        dict(approach_mode="continuous_lateral_diagnostic"),
        dict(swing_mode="memory_gate_both_directions_diagnostic"),
        dict(consumed_bank_hash=MEAN),
        dict(proposal_factory=object()),
    ],
)
def test_public_training_cannot_mix_interventions_or_other_factories(change):
    p = plan()
    d = p.draws()[0]
    args = dict(
        course=d.course,
        partition="DECLARED_PUBLIC_TRAINING",
        entry_source_path=Path(__file__),
        consumed_bank_hash=None,
        training_plan=p,
        training_group=0,
    )
    args.update(change)
    with pytest.raises(ValueError, match="unmixed public"):
        run_native_first_touch_episode(
            argparse.Namespace(
                seed=d.seed, lane=d.lane, foundation_only=False, cached_proposal_envelope=False
            ),
            **args,
        )


def test_wrong_course_rejected_before_physics_import():
    p = plan()
    d = p.draws()[0]
    with pytest.raises(ValueError, match="exact predeclared"):
        run_native_first_touch_episode(
            argparse.Namespace(
                seed=d.seed, lane=d.lane, foundation_only=False, cached_proposal_envelope=False
            ),
            course=(2.45, 0.0, 0.5),
            partition="DECLARED_PUBLIC_TRAINING",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            training_plan=p,
            training_group=0,
        )


@pytest.mark.parametrize(
    "partition",
    ["DECLARED_DEVELOPMENT_DIAGNOSTIC", "CONSUMED_TRANSFER_DIAGNOSTIC", "FRESH_CPU_EXAM"],
)
def test_training_plan_cannot_be_attached_to_other_partitions(partition):
    with pytest.raises(ValueError):
        run_native_first_touch_episode(
            argparse.Namespace(),
            course=(2.2, 0.0, 0.5),
            partition=partition,
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            training_plan=plan(),
            training_group=0,
        )
