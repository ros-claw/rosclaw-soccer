"""Fake whole14 proof plumbing, zero real G1 physics or NN inference."""

import json
import sys
from types import SimpleNamespace

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_verify_born_release as entry
from scripts import rsi_verify_navigation_recovery as mod
from scripts.rsi_stage_release_contact_replay import digest
from tests.rsi.test_born_release_qualification import (
    test_whole14_or_preserved_partial as make_fake_campaign,
)


@pytest.mark.parametrize("failure", [None, "physics", "prefix_receipt", "label"])
def test_whole14_or_reject_before_false_qualification(monkeypatch, tmp_path, failure):
    import mujoco

    from rosclaw_soccer.physics import recorded_early_episode
    from rosclaw_soccer.rsi import born_release_parity

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
    world.write_bytes(b"synthetic-not-physical-world")
    commitment = tmp_path / "commitment.json"
    commitment.write_text(
        json.dumps(
            dict(
                motor_policy_hash="original-policy",
                physics=dict(canonical_pd_kp=[1.0] * 29, canonical_pd_kd=[0.1] * 29),
            )
        )
    )
    witness = dict(
        world=dict(path=str(world), sha256=digest(world)),
        commitment=dict(path=str(commitment), sha256=digest(commitment)),
    )
    native = tmp_path / "native.json"
    seal(native, dict(results=[dict(witnesses=witness)] * 6))
    stage = tmp_path / "stage.json"
    seal(stage, dict(effective_cohort_spec=dict(producer_complete=str(native))))
    declaration["config"].update(output_root=str(root), stage_readback=str(stage))
    group = complete["rows"][0]
    job = declaration["jobs"][0]
    import numpy as np

    with np.load(group["actuals"][0]["arrays"]["path"], allow_pickle=False) as archive:
        base = {k: a for k, a in dict(archive).items() if not k.startswith("born_release_")}
    gold = tmp_path / "gold.npz"
    np.savez_compressed(gold, **base)
    job.update(
        gold_arrays=dict(path=str(gold), sha256=digest(gold)),
        original_policy_hash="original-policy",
    )

    def control_info(action):
        return dict(
            action_name=action,
            start_frame=236,
            trigger_frame=None if action == "baseline" else 236,
            measured_contact_side=None if action == "baseline" else 0,
            action_available=None if action == "baseline" else True,
            cancelled=None if action == "baseline" else False,
        )

    def receipt(action):
        if action == "baseline":
            return None
        return dict(
            schema="soccer.rsi.born_postcontact_release_review.v1",
            action_name=action.split(":")[1],
            origin_frame=236,
            observed_command_frames=64,
            commands_before_origin_observed=0,
            trigger_frame=236,
            measured_contact_side=0,
            action_available=True,
            cancelled=False,
            maximum_action_duration_frames=3,
            selection_from_completed_contact_only=True,
            future_source_trace_consumed=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            new_physical_steps=0,
            new_nn_inferences=0,
        )

    for ai, row in enumerate(group["actuals"]):
        row["main_review"].update(
            constant_forward_approach=True, original_precontact_forward_law_unchanged=False
        )
        if ai:
            row["main_review"]["born_postcontact_release"] = receipt(row["action_name"])
        if failure == "prefix_receipt" and ai == 2:
            row["main_review"]["born_postcontact_release"]["observed_command_frames"] = 300
        group["actuals"][ai] = seal(root / f"case-8-arm-{ai}-complete.json", row)
    for bi, pred in enumerate(group["predictions"]):
        pred["review"].update(
            schema="soccer.rsi.full_task_branch_review.v1",
            action_name=pred["action_name"],
            start_frame=236,
            new_planner_onnx_calls=3,
            future_observations_from_own_simulation=True,
            recorded_future_targets_or_states_consumed=False,
            prefix_steps_counted_as_new=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            new_optimizer_updates=0,
            recomputed_foundation_encoder_decoder_pairs=64,
            born_postcontact_release=receipt(pred["action_name"]),
        )
        if failure == "label" and bi == 2:
            pred["predicted_outcome"]["task_pass"] = False
        group["predictions"][bi] = seal(root / f"case-8-branch-{bi}-prediction.json", pred)
    declaration = seal(root / "declaration.json", declaration)
    complete["declaration_hash"] = declaration["report_hash"]
    seal(root / "complete.json", complete)
    # Strict physical/capture algebra is tested separately; these deliberately
    # sparse7-field fake rows test whole-campaign binding and rejection only.
    monkeypatch.setattr(
        born_release_parity,
        "body_arrays",
        lambda a: {k: v for k, v in a.items() if not k.startswith("born_release_")},
    )
    monkeypatch.setattr(
        born_release_parity,
        "review_born_release",
        lambda trace, base, *, action_name, **kw: control_info(action_name),
    )
    monkeypatch.setattr(mod, "require_terminal", lambda _: None)
    calls = []

    class Physics:
        def __init__(self, *a, **kw):
            self.attempted_steps = 0

        def verify_born_release(self, trace):
            self.attempted_steps = 3000
            calls.append(1)
            return dict(
                actual_physical_steps=3000,
                maximum_absolute_errors=dict(qpos=1.0 if failure == "physics" else 0.0),
            )

    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(from_binary_path=lambda _: None))
    monkeypatch.setattr(recorded_early_episode, "RecordedEarlyEpisode", Physics)
    output = tmp_path / "verified"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
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
            entry.main()
        assert not (output / "complete.json").exists()
        if failure == "physics":
            assert len(calls) == 1
        else:
            assert not calls and not output.exists()
    else:
        entry.main()
        result = json.loads((output / "complete.json").read_text())
        assert result["actual_physical_steps"] == 42000 and len(calls) == 14
        assert result["new_nn_inferences"] == result["new_optimizer_updates"] == 0
        assert not result["teacher_qualified"]
