"""Whole252 file protocol with synthetic arrays/mocked geometry, NO physics."""

import json
import sys

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_parity import SHAPES
from rosclaw_soccer.rsi.local_impact_perturbation_plan import declare_local_impact_perturbations
from scripts import rsi_read_postcontact_full_bank as mod
from scripts.rsi_postcontact_full_bank import summarize_context
from tests.rsi.test_local_impact_perturbation_plan import courses
from tests.rsi.test_postcontact_full_bank import scope


def test_whole252_file_protocol_exports_only_unadmitted_current121_features(monkeypatch, tmp_path):
    def sealed(file, value):
        d = dict(value)
        d["report_hash"] = mod.hash_json(d)
        file.write_text(json.dumps(d))
        return d

    def archive(file, a):
        np.savez_compressed(file, **a)
        return dict(path=str(file), sha256=mod.digest(file))

    def baseline(course):
        a = {
            k: np.zeros(
                (300, 1, *tail),
                dtype=np.bool_
                if k == "recurrent_sampling_performed"
                else np.float32
                if k.startswith("foundation_neural_")
                else np.float64,
            )
            for k, tail in SHAPES.items()
        }
        a["root_pose_xyzw_m"][:, :, 2] = 0.793
        a["root_pose_xyzw_m"][:, :, 6] = 1
        a["canonical_qpos"][:, :, 2] = 0.793
        a["canonical_qpos"][:, :, 3] = 1
        a["canonical_qpos"][:, :, 39] = 1
        a["canonical_qpos"][:, :, 36:39] = course
        a["pelvis_z_per_substep_m"][:] = 0.793
        a["force_n"][40, 0, 0] = 3
        a["ball_position_after_step_m"][:, 0, 0] = course[0] + np.arange(300) * 0.025
        a["ball_position_after_step_m"][:, 0, 2] = 0.13
        return a

    root = tmp_path / "bank"
    root.mkdir()
    local = tmp_path / "local"
    local.mkdir()
    authority = tmp_path / "authority"
    authority.mkdir()
    world = tmp_path / "world.mjb"
    world.write_bytes(b"synthetic geometry, never integrated")
    parents = []
    for i, course in enumerate(courses()):
        cm = tmp_path / f"commitment-{i}.json"
        cm.write_text(json.dumps(dict(course=list(course), motor_policy_hash="policy")))
        parents.append(
            dict(
                receipt_hash=f"receipt-{i}",
                witnesses=dict(
                    world=dict(path=str(world), sha256=mod.digest(world)),
                    commitment=dict(path=str(cm), sha256=mod.digest(cm)),
                    trace=archive(tmp_path / f"source-{i}.npz", baseline(course)),
                ),
            )
        )
    native = sealed(tmp_path / "native.json", dict(results=parents))
    stage = sealed(
        tmp_path / "stage.json",
        dict(effective_cohort_spec=dict(producer_complete=str(tmp_path / "native.json"))),
    )
    assert native and stage
    plan = declare_local_impact_perturbations(courses())
    pairs = []
    for i, job in enumerate(plan["jobs"]):
        a = baseline(job["course"])
        old = sealed(
            local / f"original-{i}.json",
            dict(arm="original_control", arrays=archive(local / f"original-{i}.npz", a)),
        )
        pairs.append(sealed(local / f"pair-{i}-complete.json", dict(job=job, arms=[old])))
    ld = sealed(local / "declaration.json", dict(plan=plan))
    lc = sealed(local / "complete.json", dict(declaration_hash=ld["report_hash"], rows=pairs))
    assert lc
    cfg = dict(
        scope(),
        output_root=str(root),
        stage_readback=str(tmp_path / "stage.json"),
        local_root=str(local),
        local_complete_file_hash=mod.digest(local / "complete.json"),
        authority_root=str(authority),
    )
    bank = {name: np.zeros(29) for name in cfg["action_names"]}
    for i, name in enumerate(cfg["action_names"][1:-1], start=1):
        bank[name][3] = 0.1 * i
    calls = []
    monkeypatch.setattr(mod, "current_named_bank", lambda *a: (calls.append(a), bank.copy())[1])
    contexts = []
    jobs = []
    authority_rows = []
    for ci in range(18):
        if ci < 12:
            si = plan["jobs"][ci]["source_index"]
            course = plan["jobs"][ci]["course"]
            bh = pairs[ci]["arms"][0]["report_hash"]
        else:
            si = ci - 12
            course = list(courses()[si])
            bh = parents[si]["receipt_hash"]
        base = baseline(course)
        outcome = label_first_touch(base).to_dict()
        job = dict(
            index=ci,
            source_index=si,
            course=course,
            baseline_hash=bh,
            baseline_outcome=outcome,
            partition="CONSUMED_LOCAL_DEVELOPMENT" if ci < 12 else "CONSUMED_ORIGINAL_RETENTION",
        )
        jobs.append(job)
        rows = []
        for ai, name in enumerate(cfg["action_names"]):
            t = {k: v.copy() for k, v in base.items()}
            action = np.zeros((300, 1, 29))
            action[41:44, 0] = bank[name]
            torque = np.tile(action[:, :, None], (1, 1, 10, 1))
            t.update(
                postcontact_release_requested_offset_nm=action,
                postcontact_release_applied_offset_nm=torque,
                postcontact_release_forecast_steps=np.zeros((300, 1, 1), dtype=np.int64),
                torque_nm=torque,
                actual_actuator_force_nm=torque.copy(),
            )
            r = dict(
                native_integrated_steps=3000,
                native_attempted_steps=3000,
                forecast_integrated_steps=0,
                forecast_attempted_steps=0,
                current_body_policy_recomputed=True,
                source_future_trace_or_targets_consumed=False,
                independently_implemented_neural_policy=False,
                postcontact_release=dict(
                    action_name=name,
                    maximum_action_duration_frames=3,
                    selection_from_completed_contact_only=True,
                    future_source_trace_consumed=False,
                    trigger_frame=41,
                    measured_contact_side=0,
                    cancelled=False,
                    action_available=True,
                ),
            )
            aw = archive(
                (authority / f"trial-{ai}-trace.npz")
                if ci == 1
                else (root / f"context-{ci}-action-{ai}-trace.npz"),
                t,
            )
            control = mod.check_release_trace(t, r, action_name=name)
            old = dict(
                arrays=aw,
                causal_review=r,
                control_review=control,
                outcome=outcome,
                original_policy_hash="policy",
                report_hash=f"old-{ai}",
            )
            if ci == 1:
                authority_rows.append(old)
            rows.append(
                sealed(
                    root / f"context-{ci}-action-{ai}-complete.json",
                    dict(
                        context_index=ci,
                        action_index=ai,
                        action_name=name,
                        course=course,
                        source_receipt_hash=parents[si]["receipt_hash"],
                        original_policy_hash="policy",
                        causal_review=r,
                        control_review=control,
                        arrays=aw,
                        outcome=outcome,
                        normalized_action_norm_squared=float(
                            np.max(np.sum((action[:, 0] / mod.CAPS) ** 2, axis=1))
                        ),
                        execution_reused=ci == 1,
                        reused_case_hash=f"old-{ai}" if ci == 1 else None,
                        new_optimizer_updates=0,
                        teacher_qualified=False,
                        training_authorized=False,
                        promotion_authorized=False,
                        hardware_authorized=False,
                    ),
                )
            )
        context = summarize_context(job, rows)
        sealed(
            root / f"context-{ci}-complete.json",
            {k: v for k, v in context.items() if k != "report_hash"},
        )
        contexts.append(context)
    ac = sealed(authority / "complete.json", dict(rows=authority_rows))
    cfg["authority_complete_file_hash"] = mod.digest(authority / "complete.json")
    d = sealed(
        root / "declaration.json",
        dict(
            schema="soccer.rsi.postcontact_full_h300_bank_declaration.v1",
            config=cfg,
            jobs=jobs,
            input_file_hashes={},
            reused_authority_hash=ac["report_hash"],
        ),
    )
    c = sealed(
        root / "complete.json",
        dict(
            schema="soccer.rsi.postcontact_full_h300_bank_complete.v1",
            declaration_hash=d["report_hash"],
            contexts=contexts,
            comparison_contexts=18,
            comparison_rollouts=252,
            new_rollouts=238,
            reused_execution_count=14,
            new_native_steps=714000,
            executed_control_foundation_pairs=71400,
            forecast_steps=0,
            new_optimizer_updates=0,
            original_baseline_passes=18,
            best_observed_safe_pass_contexts=18,
            selection_uses_full_simulated_future_outcomes=True,
            runtime_future_outcome_input_authorized=False,
            teacher_qualified=False,
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    assert c
    output = tmp_path / "readback.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "read",
            "--root",
            str(root),
            "--unit",
            "synthetic.service",
            "--source-head",
            "synthetic-head",
            "--complete-file-hash",
            mod.digest(root / "complete.json"),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(mod, "require_terminal", lambda unit: None)
    monkeypatch.setattr(
        mod.subprocess,
        "check_output",
        lambda argv, **kw: "synthetic-head" if "rev-parse" in argv else "",
    )
    mod.main()
    result = json.loads(output.read_text())
    assert len(calls) == 18 and len(result["candidate_supervision_not_admitted"]) == 18
    assert result["comparison_rollouts"] == 252 and result["new_rollouts_in_source"] == 238
    assert result["reused_execution_count"] == 14
    assert (
        result["new_physical_steps"]
        == result["new_neural_inferences"]
        == result["new_optimizer_updates"]
        == 0
    )
    assert not result["teacher_qualified"] and not result["training_authorized"]
    for candidate in result["candidate_supervision_not_admitted"]:
        assert (
            len(candidate["current_features"]) == 121
            and len(candidate["candidate_offsets_nm"]) == 13
        )
        assert (
            len(candidate["observed_recovery_proxies"]) == 13
            and not candidate["training_authorized"]
        )
