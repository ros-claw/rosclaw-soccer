from copy import deepcopy

import pytest

from rosclaw_soccer.rsi.paid_intervention_replan import replan_config
from rosclaw_soccer.rsi.paid_replan_data import admitted_replan_windows


def witnesses():
    config = replan_config(104, 289)
    counts = dict(episodes=112, CEM_updates=4, R2_planner=32368, IK=32368)
    producer = dict(
        status="PAID_INTERVENTIONAL_OWN_PREFIX_REPLAN_CLOSED_PENDING_AUDIT",
        config=config,
        counts=counts,
        actual_native_steps=435680,
        actual_neural_rows=43568,
        optimizer_updates=0,
        Fresh_rows_read=0,
        hardware_authorized=False,
        promotion_authorized=False,
        searches=[],
    )
    audit = dict(
        status="PAID_INTERVENTIONAL_OWN_PREFIX_ALL_TRIAL_NN_IK_NATIVE_AUDIT_PASS",
        source_hash="source",
        counts=dict(**counts, native=435680, neural=43568),
        optimizer_updates=0,
        Fresh_rows_read=0,
        hardware_authorized=False,
        promotion_authorized=False,
        summaries=[],
    )
    for cut in config["switch_controls"]:
        rows = [
            dict(
                case=i,
                quality=True,
                healthy_success=True,
                switch_control=cut,
                latent=[[0] * 6] * 2,
                kick=dict(raw_path="kick.npz", raw_hash="raw"),
            )
            for i in (4, 5)
        ]
        outcomes = [
            dict(case=i, quality=True, healthy_success=True, stand_healthy=True, stand_stable=True)
            for i in (4, 5)
        ]
        producer["searches"].append(
            dict(
                switch_control=cut, latent=[[0] * 6] * 2, selected=[deepcopy(rows), deepcopy(rows)]
            )
        )
        audit["summaries"].append(
            dict(switch_control=cut, outcomes=[deepcopy(outcomes), deepcopy(outcomes)])
        )
    return producer, audit


def test_canonical_selected_windows_only_not_population_or_extra_repeat():
    p, a = witnesses()
    windows = admitted_replan_windows(p, a, "source")
    assert [(w["case"], w["switch_control"]) for w in windows] == [
        (4, 70),
        (5, 70),
        (4, 80),
        (5, 80),
    ]
    assert all(w["canonical_repeat"] == 0 and w["actual_physical_repeats"] == 2 for w in windows)


def test_honest_failed_expert_is_memory_not_positive_data():
    p, a = witnesses()
    for repeat in range(2):
        p["searches"][0]["selected"][repeat][1].update(quality=False, healthy_success=False)
        a["summaries"][0]["outcomes"][repeat][1].update(quality=False, healthy_success=False)
    windows = admitted_replan_windows(p, a, "source")
    assert len(windows) == 3
    a["summaries"][0]["outcomes"][1][1]["healthy_success"] = True
    with pytest.raises(ValueError, match="witness differs"):
        admitted_replan_windows(p, a, "source")


@pytest.mark.parametrize(
    "fault", ["native", "episodes", "source", "hardware", "missing_repeat", "latent"]
)
def test_incomplete_tampered_or_different_teacher_witness_cannot_admit(fault):
    p, a = witnesses()
    if fault == "native":
        a["counts"]["native"] -= 1
    elif fault == "episodes":
        a["counts"]["episodes"] -= 1
    elif fault == "source":
        a["source_hash"] = "other"
    elif fault == "hardware":
        a["hardware_authorized"] = True
    elif fault == "missing_repeat":
        a["summaries"][0]["outcomes"].pop()
    else:
        p["searches"][0]["selected"][1][0]["latent"] = [[1] * 6] * 2
    with pytest.raises(ValueError):
        admitted_replan_windows(p, a, "source")


def test_dirty_second_generation_curator_never_allocates(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_curate_paid_replans as runner

    args = ["run", "--expected-head", "frozen", "--output", str(tmp_path / "out")]
    for name in ("previous-root", "replan-root", "audit-root"):
        args += ["--" + name, str(tmp_path)]
    args += ["--previous-hash", "hash", "--producer-head", "producer"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="clean frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()


def test_second_generation_loader_does_not_relax_first_generation_contract(tmp_path):
    import hashlib
    import json

    from rosclaw_soccer.rsi.paid_replan_data import load_replanned_data

    (tmp_path / "declaration.json").write_text("{}")
    path = tmp_path / "complete.json"
    path.write_text(
        json.dumps(dict(status="PAID_INTERVENTIONAL_DAGGER_TRAIN_FEATURES_CAUSALLY_RECOMPUTED"))
    )
    h = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="explicit generation-two"):
        load_replanned_data(tmp_path, h, tmp_path / "never-open-policy.onnx")
