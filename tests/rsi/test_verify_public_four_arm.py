"""Synthetic160-arm verification orchestration; not actual body success."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_evidence import FIELDS, PREFIX
from rosclaw_soccer.rsi.foot_heading_parity import SHAPES
from rosclaw_soccer.rsi.postcontact_robustness_plan import declare_robustness40
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_verify_public_four_arm as mod
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_postcontact_exit_trace import check_exit_trace
from scripts.rsi_stage_release_contact_replay import digest
from tests.rsi.test_public_four_arm_collection import scope


@pytest.mark.parametrize("failure", [None, "partial", "heading", "physics"])
def test_whole160_or_fail_closed(monkeypatch, tmp_path, failure):
    from rosclaw_soccer.physics import recorded_early_episode

    root, output = tmp_path / "producer", tmp_path / "review"
    root.mkdir()

    def seal(path, data):
        data["report_hash"] = hash_json(data)
        write_once(path, data)
        return data

    mean = "sha256:" + "a" * 64
    plan = declare_robustness40(mean_model_hash=mean)
    world = tmp_path / "world.mjb"
    world.write_bytes(b"synthetic fixture")
    commitment = tmp_path / "commitment.json"
    write_once(
        commitment, dict(physics=dict(canonical_pd_kp=[1.0] * 29, canonical_pd_kd=[0.1] * 29))
    )
    native = tmp_path / "native.json"
    seal(
        native,
        dict(
            results=[
                dict(
                    receipt_hash="anchor",
                    witnesses=dict(
                        world=dict(path=str(world), sha256=digest(world)),
                        commitment=dict(path=str(commitment), sha256=digest(commitment)),
                    ),
                )
            ]
        ),
    )
    stage = tmp_path / "stage.json"
    seal(stage, dict(effective_cohort_spec=dict(producer_complete=str(native))))
    base = {
        k: np.zeros(
            (300, 1, *shape),
            dtype=np.bool_
            if k == "recurrent_sampling_performed"
            else np.float32
            if k.startswith("foundation_neural_")
            else np.float64,
        )
        for k, shape in SHAPES.items()
    }
    base.update({PREFIX + k: np.zeros((300, 1, width)) for k, width in FIELDS.items()})
    base["canonical_qpos"][:, :, 2] = 0.793
    base["pelvis_z_per_substep_m"][:] = 0.793
    rows, pairs = [], []
    for job in plan["records"]:
        ci = job["index"]
        originals = {}
        for ai, arm in enumerate(job["arms"]):
            teacher, heading = arm["postcontact_exit_enabled"], arm["heading_enabled"]
            a = {k: v.copy() for k, v in base.items()}
            a["foot_heading_state"][:, 0, 3] = int(heading)
            review = dict(
                native_integrated_steps=3000,
                native_attempted_steps=3000,
                forecast_integrated_steps=0,
                forecast_attempted_steps=0,
            )
            if teacher:
                a.update(
                    postcontact_exit_requested_offset_nm=np.zeros((300, 1, 29)),
                    postcontact_exit_applied_offset_nm=np.zeros((300, 1, 10, 29)),
                    postcontact_exit_forecast_steps=np.zeros((300, 1, 1), dtype=np.int64),
                )
                review.update(
                    postcontact_exit_owned_updates_reused=True,
                    postcontact_exit=dict(
                        selected_name=None,
                        trigger_frame=None,
                        measured_contact_side=None,
                        cancelled=False,
                        maximum_forecast_steps=560,
                        forecast_integrated_steps=0,
                    ),
                )
            control = check_exit_trace(a, review) if teacher else None
            if failure == "heading" and ci == 0 and ai == 0:
                a["foot_heading_state"][:, 0, 3] = 1.0
            path = root / f"course-{ci}-arm-{ai}-trace.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **a)
            row = dict(
                course_index=ci,
                arm_index=ai,
                arm=arm,
                course=job["course"],
                sampling_seed=job["sampling_seed"],
                source_anchor_receipt_hash="anchor",
                original_policy_hash=f"policy-{ci}",
                arrays=dict(path=str(path), sha256=digest(path)),
                outcome=label_first_touch(a).to_dict(),
                causal_review=review,
                control_review=control,
                new_optimizer_updates=0,
                teacher_qualified=False,
                training_authorized=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
            row = seal(root / f"course-{ci}-arm-{ai}-complete.json", row)
            rows.append(row)
            if teacher:
                old = originals[heading]
                pairs.append(
                    dict(
                        course_index=ci,
                        heading_enabled=heading,
                        baseline_hash=old["report_hash"],
                        teacher_hash=row["report_hash"],
                        baseline_outcome=old["outcome"],
                        teacher_outcome=row["outcome"],
                    )
                )
            else:
                originals[heading] = row
    cfg = dict(scope(), mean_model_hash=mean, output_root=str(root), stage_readback=str(stage))
    d = seal(
        root / "declaration.json",
        dict(
            schema="soccer.rsi.public_four_arm40_declaration.v1",
            config=cfg,
            plan=plan,
            input_file_hashes={},
        ),
    )
    c = dict(
        schema="soccer.rsi.public_four_arm40_complete.v1",
        declaration_hash=d["report_hash"],
        rows=rows if failure != "partial" else rows[:-1],
        pairs=pairs,
        summary=mod.summarize_pairs(pairs),
        unique_initial_states=40,
        generator_seed_clusters=10,
        new_native_rollouts=160,
        new_native_steps=480000,
        executed_control_foundation_pairs=48000,
        new_optimizer_updates=0,
        forecast_steps=0,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        private_fresh_accessed=False,
    )
    seal(root / "complete.json", c)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review",
            "--root",
            str(root),
            "--unit",
            "unit",
            "--source-head",
            "head",
            "--complete-file-hash",
            digest(root / "complete.json"),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(mod, "require_terminal", lambda name: None)
    monkeypatch.setattr(
        mod.subprocess, "check_output", lambda args, **kw: "head\n" if "rev-parse" in args else ""
    )
    decision_calls = []
    monkeypatch.setattr(
        mod, "read_exit_decisions", lambda trace, root, **kw: decision_calls.append(root) or []
    )
    physics_calls = []

    class Physics:
        attempted_steps = 0

        def __init__(self, *a, **kw):
            physics_calls.append(kw)

        def _verify(self, t):
            if failure == "physics":
                self.attempted_steps = 10
                raise ValueError("synthetic physics mismatch")
            self.attempted_steps = 3000
            return dict(actual_physical_steps=3000, maximum_absolute_errors=dict(qpos=0.0))

        verify_unforced_control = _verify
        verify_postcontact_exit = _verify

    monkeypatch.setattr(recorded_early_episode, "RecordedEarlyEpisode", Physics)
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(MjModel=SimpleNamespace(from_binary_path=lambda path: object())),
    )
    if failure:
        with pytest.raises(ValueError):
            mod.main()
        assert not (output / "complete.json").exists()
        if failure == "physics":
            f = json.loads((output / "failure.json").read_text())
            assert f["current_attempted_steps"] == 10 and not f["automatic_retry"]
            assert len(decision_calls) == 80 and len(physics_calls) == 1
        else:
            assert physics_calls == [] and not output.exists()
        return
    mod.main()
    c = json.loads((output / "complete.json").read_text())
    assert len(c["rows"]) == len(physics_calls) == 160 and len(decision_calls) == 80
    assert c["actual_physical_steps"] == 480000 and c["new_neural_inferences"] == 0
    assert (
        not c["training_authorized"]
        and not c["independently_recomputed_neural_or_search_qualified"]
    )
