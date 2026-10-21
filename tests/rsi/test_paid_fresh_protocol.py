import copy
from threading import Lock
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_fresh_protocol import fresh_lifecycle_cases, fresh_summary
from rosclaw_soccer.rsi.paid_research_bench import PaidResearchBench


def source_cases():
    return [
        dict(
            case=i,
            source=f"soccer-{i // 2}-right.npz",
            profile="cached-standard",
            ball_mass_kg=0.43,
            ball_world_initial=[0.65 if i % 2 == 0 else 1.0, -0.15, 0.11],
            goal_world=[6.0, 0.0, 0.11],
            obsolete_outcome="must not enter Fresh",
        )
        for i in range(16)
    ]


def test_fixed32_contexts_cover_all_four_classes_without_copying_outcomes():
    sources = source_cases()
    before = copy.deepcopy(sources)
    cases = fresh_lifecycle_cases(sources)
    assert cases == fresh_lifecycle_cases(sources) and sources == before
    assert len(cases) == 32 and [c["case"] for c in cases] == list(range(32))
    assert len({c["source"] for c in cases}) == 8
    assert all("obsolete_outcome" not in c for c in cases)
    for c in cases:
        velocity = np.array(c["velocity_world_initial"])
        assert (
            (-0.23 <= velocity[0] <= -0.17)
            if c["fresh_class"].endswith("toward")
            else (0.17 <= velocity[0] <= 0.23)
        )
        assert abs(velocity[1]) <= 0.04
        assert c["goal_world"] != sources[2 * c["source_cluster"]]["goal_world"]
    assert all(
        sum(c["fresh_class"] == label for c in cases) == 8
        for label in ("near_toward", "near_away", "far_toward", "far_away")
    )


@pytest.mark.parametrize("fault", ["missing", "source_pair", "mass", "height", "nan"])
def test_invalid_source_declaration_is_rejected_not_resampled(fault):
    sources = source_cases()
    if fault == "missing":
        sources.pop()
    elif fault == "source_pair":
        sources[1]["source"] = "other"
    elif fault == "mass":
        sources[0]["ball_mass_kg"] = 0.4
    elif fault == "height":
        sources[0]["ball_world_initial"][2] = 0.12
    else:
        sources[0]["goal_world"][0] = float("nan")
    with pytest.raises(ValueError):
        fresh_lifecycle_cases(sources)


def test_summary_includes_every_failure_and_does_not_count_repeats_as_contexts():
    cases = fresh_lifecycle_cases(source_cases())
    worlds = [dict(case=i, quality_by_kick=[True, False, True], stable=True) for i in range(32)]
    worlds[0].update(quality_by_kick=[False], stable=False)
    summary = fresh_summary(worlds, cases)
    assert summary["stable"] == 31 and summary["quality_by_kick"] == [31, 0, 31]
    assert summary["declared_contexts"] == 32 and summary["motion_source_clusters"] == 8
    with pytest.raises(ValueError):
        fresh_summary(worlds[:-1], cases)


def bench_fixture(monkeypatch):
    import rosclaw_soccer.rsi.paid_research_bench as module

    class Episode:
        def __init__(self, *_):
            self.work = SimpleNamespace(native_steps=0)

        def reset(self, **kwargs):
            self.actual = kwargs

    monkeypatch.setattr(module, "PaidNativeEpisode", Episode)
    bench = PaidResearchBench.__new__(PaidResearchBench)
    bench.template = SimpleNamespace(model=mujoco.MjModel.from_xml_string("<mujoco/>"))
    bench.lock, bench.counters = Lock(), []
    bench.clips = {"right": SimpleNamespace(body_quat_w=np.array([[[1.0, 0, 0, 0]]]))}
    bench.contract = SimpleNamespace(parameters={}, hg_from_policy=[])
    bench.effort, bench.build_observation, bench.torso_index = [], None, 0
    bench.quat_apply = lambda _q, v: v.copy()
    return bench


def test_explicit_world_velocity_is_applied_and_legacy_parity_stays_unchanged(monkeypatch):
    bench = bench_fixture(monkeypatch)
    case = dict(
        case=0,
        source="right",
        ball_world_initial=[0.7, 0, 0.11],
        goal_world=[6, 0, 0.11],
        velocity_world_initial=[0.12, -0.11],
    )
    ep = bench.make(case)
    np.testing.assert_array_equal(ep.actual["velocity"], [0.12, -0.11])
    del case["velocity_world_initial"]
    np.testing.assert_array_equal(bench.make(case).actual["velocity"], [0, 0])
    case["case"] = 1
    np.testing.assert_array_equal(bench.make(case).actual["velocity"], [-0.2, -0.1])


@pytest.mark.parametrize("velocity", [[0.7, 0], [np.nan, 0], [1], [[0, 0]]])
def test_bad_world_velocity_is_rejected_without_clipping(monkeypatch, velocity):
    bench = bench_fixture(monkeypatch)
    with pytest.raises(ValueError):
        bench.make(
            dict(
                case=0,
                source="right",
                ball_world_initial=[0.7, 0, 0.11],
                goal_world=[6, 0, 0.11],
                velocity_world_initial=velocity,
            )
        )
