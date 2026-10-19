"""Synthetic whole12 reader/physics orchestration; zero real physics or NN."""

import json
import sys
from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.born_navigation_recovery import BornNavigationRecovery
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_verify_navigation_recovery as mod
from scripts.rsi_stage_release_contact_replay import digest
from tests.rsi.test_navigation_recovery_qualification import (
    test_whole_four_contexts_or_preserved_failure as make_fake_campaign,
)


def campaign(monkeypatch, tmp_path):
    # Reuse a FAKE producer execution, then give the synthetic artifacts the
    # complete production ABI. No real model, contact or body is allocated.
    make_fake_campaign(monkeypatch, tmp_path, None)
    root = tmp_path / "out"

    def seal(path, value):
        value = {k: v for k, v in value.items() if k != "report_hash"}
        value["report_hash"] = hash_json(value)
        path.write_text(json.dumps(value))
        return value

    complete = json.loads((root / "complete.json").read_text())
    declaration = json.loads((root / "declaration.json").read_text())
    world = tmp_path / "world.bin"
    world.write_bytes(b"synthetic-not-a-real-mujoco-world")
    commitment = tmp_path / "commitment.json"
    commitment.write_text(
        json.dumps(
            dict(
                motor_policy_hash="original-policy",
                physics=dict(canonical_pd_kp=[1.0] * 29, canonical_pd_kd=[0.1] * 29),
            )
        )
    )
    witnesses = dict(
        world=dict(path=str(world), sha256=digest(world)),
        commitment=dict(path=str(commitment), sha256=digest(commitment)),
    )
    native = tmp_path / "native.json"
    seal(native, dict(results=[dict(witnesses=witnesses)] * 6))
    stage = tmp_path / "stage.json"
    seal(stage, dict(effective_cohort_spec=dict(producer_complete=str(native))))
    declaration["config"].update(output_root=str(root), stage_readback=str(stage))
    for group, job in zip(complete["rows"], declaration["jobs"], strict=True):
        i = job["index"]
        job["gold_arrays"] = group["actuals"][0]["arrays"]
        job["original_policy_hash"] = "original-policy"
        start = job["comparison_start_after_measured_foot_bin"]
        for ai, row in enumerate(group["actuals"]):
            row["main_review"].update(
                constant_forward_approach=True, original_precontact_forward_law_unchanged=False
            )
            if ai:
                row["main_review"]["born_navigation_recovery"] = BornNavigationRecovery(
                    start, row["action_name"]
                ).contract()
            group["actuals"][ai] = seal(root / f"case-{i}-arm-{ai}-complete.json", row)
        for bi, pred in enumerate(group["predictions"]):
            pred["review"].update(
                schema="soccer.rsi.full_task_branch_review.v1",
                action_name=pred["action_name"],
                start_frame=start,
                new_planner_onnx_calls=sum(f % 20 == 0 for f in range(start, 300)),
                future_observations_from_own_simulation=True,
                recorded_future_targets_or_states_consumed=False,
                prefix_steps_counted_as_new=False,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
                new_optimizer_updates=0,
                recomputed_foundation_encoder_decoder_pairs=300 - start,
            )
            group["predictions"][bi] = seal(root / f"case-{i}-branch-{bi}-prediction.json", pred)
    declaration = seal(root / "declaration.json", declaration)
    complete["declaration_hash"] = declaration["report_hash"]
    seal(root / "complete.json", complete)
    return root, seal


@pytest.mark.parametrize("failure", [None, "physics", "row_promotion", "predicted_label"])
def test_whole12_physics_or_strict_rejection(monkeypatch, tmp_path, failure):
    import mujoco

    from rosclaw_soccer.physics import recorded_early_episode

    root, seal = campaign(monkeypatch, tmp_path)
    if failure in ("row_promotion", "predicted_label"):
        complete = json.loads((root / "complete.json").read_text())
        if failure == "row_promotion":
            row = complete["rows"][0]["actuals"][1]
            row["promotion_authorized"] = True
            complete["rows"][0]["actuals"][1] = seal(root / "case-0-arm-1-complete.json", row)
        else:
            pred = complete["rows"][0]["predictions"][2]
            pred["predicted_outcome"]["task_pass"] = True
            complete["rows"][0]["predictions"][2] = seal(
                root / "case-0-branch-2-prediction.json", pred
            )
        seal(root / "complete.json", complete)
    calls = []

    class Physics:
        def __init__(self, *a, **kw):
            self.attempted_steps = 0

        def verify_unforced_control(self, trace):
            self.attempted_steps = 3000
            calls.append(1)
            return dict(
                actual_physical_steps=3000,
                maximum_absolute_errors=dict(qpos=1.0 if failure == "physics" else 0.0),
            )

    monkeypatch.setattr(mod, "require_terminal", lambda _: None)
    # Algebra itself has separate strict43-field tests; here only test sealed
    # whole-bank orchestration with explicit synthetic4-field captures.
    monkeypatch.setattr(
        mod,
        "review_navigation_recovery",
        lambda trace, gold, action_name: dict(
            start_frame=int(trace["force_n"][:, 0, 0].argmax()) + 1, action_name=action_name
        ),
    )
    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(from_binary_path=lambda _: None))
    monkeypatch.setattr(recorded_early_episode, "RecordedEarlyEpisode", Physics)
    output = tmp_path / "verified"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reader",
            "--root",
            str(root),
            "--unit",
            "fake",
            "--complete-file-hash",
            digest(root / "complete.json"),
            "--output",
            str(output),
        ],
    )
    if failure:
        with pytest.raises(ValueError):
            mod.main()
        assert not (output / "complete.json").exists()
        if failure == "physics":
            assert len(calls) == 1
            assert json.loads((output / "failure.json").read_text())["automatic_retry"] is False
        else:
            assert not calls and not output.exists()
    else:
        mod.main()
        result = json.loads((output / "complete.json").read_text())
        assert len(calls) == 12 and result["actual_physical_steps"] == 36000
        assert result["new_nn_inferences"] == result["new_optimizer_updates"] == 0
        assert result["independent_neural_policy_qualified"] is False
