"""One proposed source-isolated DAgger-style numeric batch, never data admission.

Keep the sixteen original bodies and thirteen original student-exam bodies.
The counterfactual expert action is queried at the ACTUAL behaviour-policy
state. It is not a verified successful continuation from that state. Only a
caller-authenticated original successful teacher domain may be supervised.
"""

import numpy as np

from rosclaw_soccer.rsi.reference_student_exam_scope import exam_scope
from rosclaw_soccer.rsi.reference_student_imitation import prepare_numeric_batch
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

LABEL_CEILING = "RESEARCH_ONLY_COUNTERFACTUAL_EXPERT_ACTIONS_NOT_VERIFIED_SUCCESS"
GROUPS = (12, 13, 28, 29, 76, 77, 92, 93)


def prepare_proposed_aggregation(
    original_x,
    original_y,
    own_x,
    queried_y,
    *,
    original_receipts,
    original_metadata,
    own_jobs,
    own_receipt_hashes,
    query_receipt_hashes,
    parent_model_hash,
    acknowledged_label_ceiling,
):
    """Proposed offline supervision, not physics-qualified labels or success.

    x for student bodies MUST be the original recorded actor inputs, including
    its own executed past actions; never reconstruct x from oracle targets.
    The current no-past-reference student ignores those twelve columns.
    File/source/trace/query authentication is the caller's responsibility.
    """
    expected_jobs = exam_scope(parent_model_hash)["jobs"]
    if (
        acknowledged_label_ceiling != LABEL_CEILING
        or own_jobs != expected_jobs
        or len(original_receipts) != 16
        or len(original_metadata) != 16
        or len(own_receipt_hashes) != 13
        or own_receipt_hashes != query_receipt_hashes
        or len(set(own_receipt_hashes)) != 13
        or len({r["receipt_hash"] for r in original_receipts}) != 16
        or set(own_receipt_hashes).intersection(r["receipt_hash"] for r in original_receipts)
    ):
        raise ValueError(
            "whole fixed 16+13 batch, query alignment and explicit research ceiling required"
        )
    original_x, original_y, own_x, queried_y = [
        np.asarray(a) for a in (original_x, original_y, own_x, queried_y)
    ]
    if any(
        a.shape != shape or a.dtype != np.float64
        for a, shape in (
            (original_x, (16, 300, 135)),
            (original_y, (16, 300, 12)),
            (own_x, (13, 300, 135)),
            (queried_y, (13, 300, 12)),
        )
    ):
        raise ValueError("all original measured actor inputs and aligned expert proposals required")
    seeds, splits, selected, lineage, domain = [], [], [], [], {}
    for i, (r, meta) in enumerate(zip(original_receipts, original_metadata, strict=True)):
        role, group = ("control" if i % 2 == 0 else "teacher"), GROUPS[i // 2]
        if (
            r["role"] != role
            or r["group"] != group
            or type(meta["eligible"]) is not bool
            or (
                meta["eligible"]
                and (
                    role != "teacher"
                    or r["outcome"]["task_pass"] is not True
                    or r["outcome"]["safe"] is not True
                )
            )
        ):
            raise ValueError("original ordered teacher/control eligibility must be preserved")
        seeds.append(meta["source_seed"])
        splits.append(meta["partition"])
        selected.append(meta["eligible"])
        lineage.append(
            dict(
                kind="original_teacher_or_control",
                group=group,
                role=role,
                source_receipt_hash=r["receipt_hash"],
                experimental_numeric_supervision=meta["eligible"],
                counterfactual_label=False,
            )
        )
        if role == "teacher":
            domain[group] = dict(meta)
        elif (meta["source_seed"], meta["partition"]) != (
            original_metadata[i + 1]["source_seed"],
            original_metadata[i + 1]["partition"],
        ):
            raise ValueError("both controllers of each source must keep original split")
    if (
        sum(selected) != 4
        or sum(e and p == "train" for e, p in zip(selected, splits, strict=True)) != 3
        or sum(e and p == "validation" for e, p in zip(selected, splits, strict=True)) != 1
    ):
        raise ValueError("original three train teachers and one validation teacher required")
    for job, receipt_hash in zip(own_jobs, own_receipt_hashes, strict=True):
        if job["plan"] == "positive19":
            m = domain[job["group"]]
            seed, split = m["source_seed"], m["partition"]
            use = job["role"] == "student" and m["eligible"]
        else:
            # The two original-plan18 retention sources are never optimization
            # examples, and remain separate from all prospective plan19 seeds.
            seed, split, use = {4: 20261060, 32: 20261062}[job["group"]], "train", False
        seeds.append(seed)
        splits.append(split)
        selected.append(use)
        lineage.append(
            dict(
                kind="recorded_student_exam_body",
                job=dict(job),
                source_receipt_hash=receipt_hash,
                experimental_numeric_supervision=use,
                counterfactual_label=True,
                expert_success_from_state_verified=False,
            )
        )
    batch = prepare_numeric_batch(
        np.concatenate((original_x, own_x)),
        np.concatenate((original_y, queried_y)),
        partitions=splits,
        source_seeds=seeds,
        eligible=selected,
    )
    if int(batch["train"].sum()) != 6 or int(batch["validation"].sum()) != 2:
        raise ValueError("exact six train and two held-out proposed supervision episodes required")
    report = dict(
        schema="soccer.rsi.proposed_reference_state_aggregation.v1",
        acknowledged_label_ceiling=LABEL_CEILING,
        rows=lineage,
        original_body_episodes=16,
        original_student_exam_body_episodes=13,
        whole_episode_count=29,
        total_recorded_frames=8700,
        proposed_train_episodes=6,
        proposed_validation_episodes=2,
        proposed_train_positive_weight_frames=int(np.count_nonzero(batch["weights"])),
        proposed_train_source_seeds=sorted(
            {s for s, t in zip(seeds, batch["train"], strict=True) if t}
        ),
        proposed_validation_source_seeds=sorted(
            {s for s, t in zip(seeds, batch["validation"], strict=True) if t}
        ),
        actual_measured_inputs_hash=hash_bytes(batch["original_observations"].tobytes()),
        same_body_label_proposals_hash=hash_bytes(batch["targets"].tobytes()),
        own_input_rebuilt_from_expert_actions=False,
        original_failures_retained_as_zero_weight_not_negative_gradients=True,
        validation_excluded_from_normalization_and_gradient=True,
        expert_success_from_student_states_verified=False,
        original_teacher_domain_authentication_performed_here=False,
        source_file_authentication_performed_here=False,
        new_optimizer_updates=0,
        new_physical_steps=0,
        dataset_admission_authorized=False,
        teacher_qualified=False,
        runtime_execution_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    return batch, report
