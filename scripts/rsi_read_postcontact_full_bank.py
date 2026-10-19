"""Whole252 offline outcome matrix and CURRENT-state learning candidates.

No new integration/NN, no training admission, no future-aware runtime actor.
All18 contexts remain in the matrix, including no-authority/late/unsafe cases.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_recovery_observations import observe_first_touch_recovery
from rosclaw_soccer.rsi.foot_heading_parity import original_arrays
from rosclaw_soccer.rsi.impact_torque_screen import CAPS
from rosclaw_soccer.rsi.local_impact_perturbation_plan import declare_local_impact_perturbations
from rosclaw_soccer.rsi.postcontact_learning_observation import (
    current_postcontact_features,
    observation_contract,
)
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_prospective_impact import require_prefix
from scripts.rsi_postcontact_full_bank import summarize_context
from scripts.rsi_postcontact_full_bank_inputs import require_scope
from scripts.rsi_postcontact_release_trace import check_release_trace
from scripts.rsi_read_local_impact_robustness import verify_original_control_torques
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def check_complete(c, d, root):
    require_scope(d["config"])
    if (
        c["schema"] != "soccer.rsi.postcontact_full_h300_bank_complete.v1"
        or d["schema"] != "soccer.rsi.postcontact_full_h300_bank_declaration.v1"
        or c["declaration_hash"] != d["report_hash"]
        or Path(d["config"]["output_root"]) != root
        or len(c["contexts"]) != 18
        or len(d["jobs"]) != 18
        or any(
            type(x.get("context_index")) is not int
            or x["context_index"] != i
            or len(x.get("rows", [])) != 14
            for i, x in enumerate(c["contexts"])
        )
        or any(
            type(c.get(k)) is not int or c[k] != v
            for k, v in dict(
                comparison_contexts=18,
                comparison_rollouts=252,
                new_rollouts=238,
                reused_execution_count=14,
                new_native_steps=714000,
                executed_control_foundation_pairs=71400,
                forecast_steps=0,
                new_optimizer_updates=0,
            ).items()
        )
        or c["selection_uses_full_simulated_future_outcomes"] is not True
        or any(
            c[k] is not False
            for k in (
                "runtime_future_outcome_input_authorized",
                "teacher_qualified",
                "training_authorized",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("whole252/238new/14reused no-learning offline bank required")


def current_named_bank(world, q, v, side):
    import mujoco

    from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.impact_torque_screen import impact_torque_candidates

    model = mujoco.MjModel.from_binary_path(world["path"])
    data = mujoco.MjData(model)
    joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
    qi = [int(model.jnt_qposadr[j]) for j in joints]
    vi = [int(model.jnt_dofadr[j]) for j in joints]
    ball = int(model.body_jntadr[int(model.body("ball").id)])
    bq, bv = int(model.jnt_qposadr[ball]), int(model.jnt_dofadr[ball])
    data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7] = q[:7], q[7:36], q[36:]
    data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6] = v[:6], v[6:35], v[35:]
    sampler = G1JointTaskKinematicSampler(model, expected_world_hash=world["sha256"])
    return dict(impact_torque_candidates(sampler.sample(data), side))


def check_row(
    row, trace, baseline, *, job, parent, commitment, context_index, action_index, action_name
):
    extra = {k for k in trace if k.startswith("postcontact_release_")}
    if extra != {
        "postcontact_release_requested_offset_nm",
        "postcontact_release_applied_offset_nm",
        "postcontact_release_forecast_steps",
    }:
        raise ValueError("exact fixed-release raw fields required")
    original_arrays({k: v for k, v in trace.items() if k not in extra})
    if (
        type(row["context_index"]) is not int
        or type(row["action_index"]) is not int
        or row["context_index"] != context_index
        or row["action_index"] != action_index
        or row["action_name"] != action_name
        or row["course"] != job["course"]
        or row["source_receipt_hash"] != parent["receipt_hash"]
        or row["original_policy_hash"] != commitment["motor_policy_hash"]
        or type(row["new_optimizer_updates"]) is not int
        or row["new_optimizer_updates"] != 0
        or any(
            row[k] is not False
            for k in (
                "teacher_qualified",
                "training_authorized",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("same original complete source policy/context/action identity required")
    review = row["causal_review"]
    if (
        any(
            type(review[k]) is not int or review[k] != v
            for k, v in dict(
                native_integrated_steps=3000,
                native_attempted_steps=3000,
                forecast_integrated_steps=0,
                forecast_attempted_steps=0,
            ).items()
        )
        or review["current_body_policy_recomputed"] is not True
        or review["source_future_trace_or_targets_consumed"] is not False
        or review["independently_implemented_neural_policy"] is not False
    ):
        raise ValueError("whole source3000-step zero-forecast action body required")
    control = check_release_trace(trace, review, action_name=action_name)
    outcome = label_first_touch(trace).to_dict()
    norm = float(
        np.max(np.sum((trace["postcontact_release_requested_offset_nm"][:, 0] / CAPS) ** 2, axis=1))
    )
    if (
        control != row["control_review"]
        or hash_json(outcome) != hash_json(row["outcome"])
        or norm != row["normalized_action_norm_squared"]
    ):
        raise ValueError("all raw controls/outcomes/action norms must rederive exactly")
    require_prefix(trace, baseline, control["first_physically_applied_correction_frame"])
    return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--complete-file-hash", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if (
        args.output.exists()
        or (args.root / "failure.json").exists()
        or digest(args.root / "complete.json") != args.complete_file_hash
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("closed whole bank bytes, unused output and clean frozen reader required")
    require_terminal(args.unit)
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts")
        for p in folder.rglob("*.py")
    }

    def pin(name, expected=None):
        file = Path(name)
        h = digest(file)
        if (
            file.is_symlink()
            or (expected is not None and h != expected)
            or (str(file) in pins and pins[str(file)] != h)
        ):
            raise ValueError("whole ordinary immutable offline source/input bytes required")
        pins[str(file)] = h

    def sealed(name):
        pin(name)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole offline bank source seal required")
        return d

    def arrays(w):
        pin(w["path"], w["sha256"])
        with np.load(w["path"], allow_pickle=False) as a:
            return dict(a)

    c, d = sealed(args.root / "complete.json"), sealed(args.root / "declaration.json")
    check_complete(c, d, args.root)
    for name, h in d["input_file_hashes"].items():
        pin(name, h)
    cfg = d["config"]
    stage = sealed(cfg["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    commitments = []
    for parent in native["results"]:
        w = parent["witnesses"]["commitment"]
        pin(w["path"], w["sha256"])
        commitments.append(json.loads(Path(w["path"]).read_text()))
    plan = declare_local_impact_perturbations(tuple(tuple(cm["course"]) for cm in commitments))
    lc = sealed(Path(cfg["local_root"]) / "complete.json")
    ld = sealed(Path(cfg["local_root"]) / "declaration.json")
    pin(Path(cfg["local_root"]) / "complete.json", cfg["local_complete_file_hash"])
    if lc["declaration_hash"] != ld["report_hash"] or ld["plan"] != plan or len(lc["rows"]) != 12:
        raise ValueError("same complete original public local context plan required")
    authority = sealed(Path(cfg["authority_root"]) / "complete.json")
    pin(Path(cfg["authority_root"]) / "complete.json", cfg["authority_complete_file_hash"])
    if authority["report_hash"] != d["reused_authority_hash"] or len(authority["rows"]) != 14:
        raise ValueError("all14 original reused authority bodies required")
    summaries, candidates = [], []
    new = reused = 0
    rescue_clusters = set()
    for ci, context in enumerate(c["contexts"]):
        if (
            context != sealed(args.root / f"context-{ci}-complete.json")
            or context["context_index"] != ci
        ):
            raise ValueError("all18 ordered whole context summaries required")
        if ci < 12:
            pair = lc["rows"][ci]
            if (
                pair != sealed(Path(cfg["local_root"]) / f"pair-{ci}-complete.json")
                or pair["job"] != plan["jobs"][ci]
            ):
                raise ValueError("same original local baseline job required")
            base = pair["arms"][0]
            if base["arm"] != "original_control":
                raise ValueError("original baseline required")
            si = plan["jobs"][ci]["source_index"]
            course = plan["jobs"][ci]["course"]
            baseline_hash = base["report_hash"]
            baseline = arrays(base["arrays"])
        else:
            si = ci - 12
            parent = native["results"][si]
            course = commitments[si]["course"]
            baseline_hash = parent["receipt_hash"]
            baseline = arrays(parent["witnesses"]["trace"])
        verify_original_control_torques(baseline)
        expected_job = dict(
            index=ci,
            source_index=si,
            course=course,
            baseline_hash=baseline_hash,
            baseline_outcome=label_first_touch(baseline).to_dict(),
            partition="CONSUMED_LOCAL_DEVELOPMENT" if ci < 12 else "CONSUMED_ORIGINAL_RETENTION",
        )
        job = d["jobs"][ci]
        if hash_json(job) != hash_json(expected_job):
            raise ValueError("original course/policy/baseline binding must rederive")
        parent = native["results"][si]
        world = parent["witnesses"]["world"]
        pin(world["path"], world["sha256"])
        rows = context["rows"]
        if len(rows) != 14:
            raise ValueError("all14 actions required for every context")
        features = None
        bank = None
        trigger = side = None
        targets, availability, recovery = [], [], []
        for ai, (row, name) in enumerate(zip(rows, cfg["action_names"], strict=True)):
            if row != sealed(args.root / f"context-{ci}-action-{ai}-complete.json"):
                raise ValueError("every ordered action row required")
            is_reused = ci == cfg["reused_authority_context_index"]
            if (
                type(row["execution_reused"]) is not bool
                or row["execution_reused"] is not is_reused
            ):
                raise ValueError("reused and new executions must remain distinct")
            if is_reused:
                old = authority["rows"][ai]
                if row["reused_case_hash"] != old["report_hash"] or any(
                    row[k] != old[k]
                    for k in (
                        "arrays",
                        "causal_review",
                        "control_review",
                        "outcome",
                        "original_policy_hash",
                    )
                ):
                    raise ValueError("whole exact original authority body required for reuse")
                reused += 1
            else:
                if row["reused_case_hash"] is not None or row["arrays"]["path"] != str(
                    args.root / f"context-{ci}-action-{ai}-trace.npz"
                ):
                    raise ValueError("new actual native body trace required")
                new += 1
            trace = arrays(row["arrays"])
            control = check_row(
                row,
                trace,
                baseline,
                job=job,
                parent=parent,
                commitment=commitments[si],
                context_index=ci,
                action_index=ai,
                action_name=name,
            )
            f = control["observed_trigger_frame"]
            if ai == 0:
                trigger = f
                side = control["measured_contact_side"]
                if f is not None:
                    features = current_postcontact_features(
                        trace["canonical_qpos"][f, 0],
                        trace["canonical_qvel"][f, 0],
                        trace["joint_target_rad"][f, 0],
                        frame=f,
                        previous_contact_forces=trace["force_n"][f - 1, 0],
                    )
                    bank = current_named_bank(
                        world, trace["canonical_qpos"][f, 0], trace["canonical_qvel"][f, 0], side
                    )
            if f != trigger or control["measured_contact_side"] != side:
                raise ValueError(
                    "all actions must share the same original causal observation boundary"
                )
            requested = trace["postcontact_release_requested_offset_nm"][:, 0]
            expected = np.zeros(29)
            available = None
            if f is not None:
                x = current_postcontact_features(
                    trace["canonical_qpos"][f, 0],
                    trace["canonical_qvel"][f, 0],
                    trace["joint_target_rad"][f, 0],
                    frame=f,
                    previous_contact_forces=trace["force_n"][f - 1, 0],
                )
                if x.tobytes() != features.tobytes():
                    raise ValueError("current learning input differs before the action")
                available = name in bank
                expected = bank.get(name, np.zeros(29))
                if not np.array_equal(requested[f], expected):
                    raise ValueError(
                        "actual named action differs from independently measured current Jacobian"
                    )
            if row["causal_review"]["postcontact_release"]["action_available"] is not available:
                raise ValueError("current action availability differs from actual source capture")
            if ai < 13:
                targets.append(expected.tolist())
                availability.append(available is True)
                measured_recovery = observe_first_touch_recovery(trace)
                recovery.append(
                    dict(
                        contact_to_plus60=measured_recovery["contact_to_plus60"],
                        contact_to_episode_end=measured_recovery["contact_to_episode_end"],
                        naturalness_or_successor_readiness_qualified=False,
                    )
                )
        rebuilt = summarize_context(job, rows)
        if rebuilt != context:
            raise ValueError("whole offline label selection must rederive from raw outcomes")
        preferred = context["offline_preferred_action_index"]
        if (
            not job["baseline_outcome"]["task_pass"]
            and context["offline_selected_outcome"]["task_pass"]
            and preferred
        ):
            rescue_clusters.add(si // 2)
        summaries.append(
            dict(
                context_index=ci,
                source_index=si,
                baseline_outcome=job["baseline_outcome"],
                offline_selected_outcome=context["offline_selected_outcome"],
                offline_preferred_action_index=preferred,
                source_context_hash=context["report_hash"],
            )
        )
        candidates.append(
            dict(
                context_index=ci,
                partition=job["partition"],
                source_cluster=si // 2,
                observed_trigger_frame=trigger,
                current_features=None if features is None else features.tolist(),
                observation_available=features is not None,
                unavailable_reason="NO_ELIGIBLE_POSTCONTACT_TRIGGER" if features is None else None,
                candidate_offsets_nm=targets,
                candidate_available=availability,
                observed_recovery_proxies=recovery,
                observed_action_task_pass=[r["outcome"]["task_pass"] for r in rows[:13]],
                observed_action_safe=[r["outcome"]["safe"] for r in rows[:13]],
                observed_action_quality_measured=[
                    r["outcome"]["quality_observed"] for r in rows[:13]
                ],
                observed_action_training_reward=[
                    r["outcome"]["training_reward"] for r in rows[:13]
                ],
                offline_preferred_action_index=preferred,
                training_authorized=False,
            )
        )
    if (
        new != 238
        or reused != 14
        or c["original_baseline_passes"]
        != sum(r["baseline_outcome"]["task_pass"] for r in summaries)
        or c["best_observed_safe_pass_contexts"]
        != sum(r["offline_selected_outcome"]["task_pass"] for r in summaries)
        or any(digest(n) != h for n, h in pins.items())
    ):
        raise ValueError("whole immutable252 matrix and primitive counts required")
    result = dict(
        schema="soccer.rsi.postcontact_full_bank_readback.v1",
        source_head=args.source_head,
        complete_hash=c["report_hash"],
        contexts=summaries,
        observation_contract=observation_contract(),
        candidate_supervision_not_admitted=candidates,
        independent_current_jacobian_banks_rebuilt=sum(
            r["observation_available"] for r in candidates
        ),
        nonzero_safe_rescue_source_clusters=sorted(rescue_clusters),
        input_file_hashes=pins,
        comparison_contexts=18,
        comparison_rollouts=252,
        new_rollouts_in_source=238,
        reused_execution_count=14,
        actual_source_native_steps=714000,
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        independent_neural_or_physical_qualified=False,
        teacher_qualified=False,
        training_authorized=False,
        runtime_future_outcome_input_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print("WHOLE252_POSTCONTACT_OFFLINE_BANK_READBACK", result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
