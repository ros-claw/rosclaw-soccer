"""Mock native boundary only; no allocated Foundation, policy or physics."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi import native_first_touch_episode
from rosclaw_soccer.rsi.first_touch_campaign_worker import qualified_episode
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory


@pytest.mark.parametrize("new_world", [False, True])
def test_exact_original_public_draw_and_none_option_reaches_declared_native_boundary(
    tmp_path, monkeypatch, new_world
):
    mean = "sha256:" + "a" * 64
    old_world = "sha256:" + "b" * 64
    new_hash = "sha256:" + "c" * 64 if new_world else None
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), mean, 18)
    draws = plan.draws()
    records = [
        dict(
            draw=dict(course=list(d.course)),
            witnesses=dict(sampling_view=dict(path=str(tmp_path / "absent-view.json"))),
        )
        for d in draws
    ]
    # This bypasses allocation ONLY inside this plumbing fixture; never evidence
    # that an unallocated private factory could execute or pass qualification.
    factory = object.__new__(RecurrentSamplingEpisodeFactory)
    monkeypatch.setattr(RecurrentSamplingEpisodeFactory, "_stable", lambda self: None)
    calls = []

    def boundary(args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("mock native allocation boundary")

    monkeypatch.setattr(native_first_touch_episode, "run_native_first_touch_episode", boundary)
    assets = SimpleNamespace(
        WORLD=old_world,
        MEAN=mean,
        SCENE=tmp_path / "absent.xml",
        SONIC=tmp_path / "absent-sonic",
        LATE=tmp_path / "absent-late",
    )
    with pytest.raises(RuntimeError, match="mock native allocation"):
        qualified_episode(
            tmp_path / "output",
            4,
            None,
            plan=plan,
            manifest=dict(records=records),
            assets=assets,
            source=Path("absent-source"),
            core=Path("absent-core"),
            factory=factory,
            compiler=None,
            auditor=None,
            mesh_task_contact_world_hash=new_hash,
        )
    args, kw = calls[0]
    assert kw["course"] == draws[4].course
    assert kw["mesh_task_contact_world_hash"] == new_hash
    assert kw["record_default_planner_events"] is new_world
    assert kw["control_profile"] is None
    assert not (tmp_path / "output").exists()


def test_mesh_world_cannot_turn_untyped_source_into_public_data(tmp_path):
    with pytest.raises(ValueError, match="fixed public source"):
        qualified_episode(
            tmp_path / "output",
            0,
            None,
            plan=None,
            manifest={},
            assets=SimpleNamespace(WORLD="old", MEAN="mean"),
            source=Path("absent"),
            core=Path("absent"),
            factory=None,
            compiler=None,
            auditor=None,
            mesh_task_contact_world_hash="new",
        )
