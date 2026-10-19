"""Local24 admission/summary tests, not real G1 robustness evidence."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.rsi_local_impact_robustness import paired_summary, require_scope


def scope():
    return dict(
        profile="local-early-impact-perturbation24-v1",
        mode="SIM_ONLY",
        source_cases=6,
        new_rollouts=24,
        new_native_step_budget=72000,
        forecast_step_budget=21600,
        combined_step_budget=93600,
        official_horizon_frames=300,
        post_contact_frames=60,
        new_optimizer_updates=0,
        reuse_owned_early_updates=True,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        private_fresh_accessed=False,
        automatic_retry=False,
    )


def test_fixed_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_missing_or_changed_protocol_is_rejected(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    v = cfg[key]
    cfg[key] = int(v) if type(v) is bool else float(v) if type(v) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)


def rows():
    records = []
    for i in range(12):
        job = dict(index=i)
        arms = [
            dict(
                arm=arm,
                job=job,
                source_receipt_hash="source",
                original_policy_hash="policy",
                outcome=dict(task_pass=i % 2 == 0, safe=True),
            )
            for arm in ("original_control", "prospective_early_impact")
        ]
        records.append(dict(index=i, job=job, arms=arms))
    return records


def test_counts_are_primitive_rescue_regression_not_selected_successes():
    r = rows()
    r[1]["arms"][1]["outcome"]["task_pass"] = True
    r[3]["arms"][1]["outcome"]["task_pass"] = True
    r[0]["arms"][1]["outcome"]["task_pass"] = False
    d = paired_summary(r)
    assert (d["original_passes"], d["candidate_passes"], d["rescues"], d["regressions"]) == (
        6,
        7,
        2,
        1,
    )
    assert d["local_reproducible_net_gain"] is True
    assert d["teacher_qualified"] is False
    assert d["training_authorized"] is False
    assert d["promotion_authorized"] is False
    r[5]["arms"][1]["outcome"]["safe"] = False
    assert paired_summary(r)["local_reproducible_net_gain"] is False


@pytest.mark.parametrize(
    "mutate", ["missing", "order", "index", "arms", "source", "policy", "job", "bool"]
)
def test_incomplete_or_unpaired_primitive_evidence_cannot_be_summarized(mutate):
    r = copy.deepcopy(rows())
    if mutate == "missing":
        r.pop()
    elif mutate == "order":
        r.reverse()
    elif mutate == "index":
        r[1]["index"] = True
    elif mutate == "arms":
        r[1]["arms"].reverse()
    elif mutate in ("source", "policy", "job"):
        key = {"source": "source_receipt_hash", "policy": "original_policy_hash", "job": "job"}[
            mutate
        ]
        r[1]["arms"][1][key] = "other"
    else:
        r[0]["arms"][0]["outcome"]["safe"] = 1
    with pytest.raises(ValueError):
        paired_summary(r)


def test_preflight_returns_before_foundation_or_physics(monkeypatch, capsys):
    from scripts import rsi_local_impact_robustness as runner

    called = []
    monkeypatch.setattr(runner, "prepare", lambda path: called.append(path) or SimpleNamespace())
    monkeypatch.setattr(sys, "argv", ["local24", "--config", "fixture.json", "--preflight"])
    runner.main()
    assert called == [Path("fixture.json")]
    assert "NO_ALLOCATION_OR_PHYSICS" in capsys.readouterr().out


@pytest.mark.parametrize("fail_frame", [None, 5])
def test_complete24_arm_orchestration_with_fake_models_not_physics_evidence(
    monkeypatch, tmp_path, fail_frame
):
    """Exercise capture/count/prefix/report ordering without allocating weights."""
    import time

    import numpy as np

    from rosclaw_soccer.providers.g1 import sonic_torch
    from rosclaw_soccer.rsi import (
        causal_impact_replay,
        first_touch_outcome,
        json_artifact_io,
        recurrent_sampling_episode_factory,
    )
    from scripts import rsi_local_impact_robustness as runner

    jobs = [dict(source_index=i // 2, course=[2.7, 0.1, 0.3], variant=i % 2) for i in range(12)]
    cases = [
        dict(
            job=dict(heading_enabled=bool(i % 2)),
            physics=dict(canonical_pd_kp=[1.0] * 29, canonical_pd_kd=[0.1] * 29),
            receipt=dict(
                receipt_hash=f"source{i}",
                outcome=dict(task_pass=True, safe=True),
                witnesses=dict(
                    world=dict(path="fake-world"),
                    commitment=dict(path="fake-commitment"),
                    report=dict(path="fake-report"),
                ),
            ),
        )
        for i in range(6)
    ]
    p = SimpleNamespace(
        cfg=dict(
            mean_model_path="fake-mean", mean_model_hash="mean", deadline_unix=time.time() + 3600
        ),
        root=tmp_path / "output",
        pins={},
        plan=dict(jobs=jobs),
        cases=cases,
        model_root=Path("fake-models"),
        world_hash="world",
        declaration=dict(candidate=dict(parameters={})),
        reused_h300=dict(
            report_hash="h300",
            rows=[dict(causal_rollout_outcome=dict(task_pass=True, safe=True)) for _ in range(6)],
        ),
    )
    monkeypatch.setattr(runner, "prepare", lambda path: p)
    monkeypatch.setattr(sys, "argv", ["local24", "--config", "fixture.json"])
    assets = {
        "fake-mean": dict(model_hash="mean"),
        "fake-commitment": dict(foundation_hash="foundation", motor_policy_hash="policy"),
        "fake-report": dict(
            stripped_document=dict(
                executed_motor_policy=dict(step_motor_proof=dict(model=dict(seed=1)))
            )
        ),
    }
    monkeypatch.setattr(json_artifact_io, "load_json_artifact", lambda path: assets[str(path)])
    original_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: (
            "MemAvailable: 100000000 kB\n"
            if str(path) == "/proc/meminfo"
            else original_read(path, *a, **kw)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(MjModel=SimpleNamespace(from_binary_path=lambda path: object())),
    )

    class Factory:
        def __init__(self, mean):
            assert mean == dict(model_hash="mean")

        def generate_and_bind_seed(self, *, seed):
            return dict(policy_hash="policy"), object()

        def _stable(self):
            pass

    enabled_arms = []

    class Replay:
        def __init__(self, model, **kwargs):
            self.enabled = kwargs["early_impact_enabled"]
            assert kwargs["reuse_owned_early_updates"] is self.enabled
            enabled_arms.append(self.enabled)
            self.attempted_steps = self.integrated_steps = 0
            self.data = SimpleNamespace(qpos=np.zeros(43), qvel=np.zeros(41), ctrl=np.zeros(29))
            self.teacher = (
                SimpleNamespace(forecaster=SimpleNamespace(attempted_steps=0))
                if self.enabled
                else None
            )

        def step(self, frame):
            assert self.integrated_steps == frame * 10
            if frame == fail_frame:
                raise RuntimeError("synthetic injected fatal failure")
            self.attempted_steps += 10
            self.integrated_steps += 10
            a = dict(sample=np.array([[float(frame)]]))
            if self.enabled:
                a.update(
                    early_impact_feedback_requested_offset_nm=np.zeros((1, 29)),
                    early_impact_feedback_applied_offset_nm=np.zeros((1, 10, 29)),
                )
            return SimpleNamespace(arrays=a)

        def review(self):
            assert self.integrated_steps == 3000
            return dict(native_integrated_steps=3000, forecast_integrated_steps=0)

    monkeypatch.setattr(
        recurrent_sampling_episode_factory, "RecurrentSamplingEpisodeFactory", Factory
    )
    monkeypatch.setattr(sonic_torch, "FrozenSonicG1Torch", lambda *a, **kw: object())
    monkeypatch.setattr(causal_impact_replay, "CausalImpactReplay", Replay)
    monkeypatch.setattr(
        first_touch_outcome,
        "label_first_touch",
        lambda trace: SimpleNamespace(
            to_dict=lambda: dict(task_pass=False, safe=True, measurement_status="RIGHT_CENSORED")
        ),
    )
    import json

    if fail_frame is not None:
        with pytest.raises(RuntimeError, match="injected fatal"):
            runner.main()
        failure = json.loads((p.root / "failure.json").read_text())
        assert failure["automatic_retry"] is False
        assert failure["completed_pairs"] == 0
        assert failure["current_attempted_steps"] == fail_frame * 10
        assert not (p.root / "complete.json").exists()
        assert (p.root / "failure-body-state.npz").is_file()
        with np.load(p.root / "failure-partial-trace.npz", allow_pickle=False) as a:
            assert a["sample"].shape == (fail_frame, 1, 1)
        assert enabled_arms == [False]
        return
    runner.main()
    assert enabled_arms == [False, True] * 12
    result = json.loads((p.root / "complete.json").read_text())
    assert len(result["rows"]) == 12
    assert result["new_native_steps"] == 72000
    assert result["forecast_steps"] == result["new_optimizer_updates"] == 0
    assert result["teacher_qualified"] is False
    assert (result["rescues"], result["regressions"], result["added_unsafe"]) == (0, 0, 0)
    assert len(list(p.root.glob("pair-*-trace.npz"))) == 24
    assert all(
        a["outcome"]["measurement_status"] == "RIGHT_CENSORED"
        for row in result["rows"]
        for a in row["arms"]
    )
