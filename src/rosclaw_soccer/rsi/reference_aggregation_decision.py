"""Finite research recommendation from caller-authenticated whole exam pairs."""


def recommend_aggregation(pairs, *, positive_dense_nonfoot):
    if type(positive_dense_nonfoot) is not int or not 0 <= positive_dense_nonfoot <= 8:
        raise ValueError("verified finite positive-student dense nonfoot count required")
    positive = pairs["positive_teacher_vs_student"]
    retention = [pairs[k] for k in ("retention_zero_vs_student", "retention_legacy_vs_student")]
    for study, n in ((positive, 8), *((r, 2) for r in retention)):
        keys = (
            "baseline_passes",
            "evaluated_passes",
            "rescued_pairs",
            "regressed_pairs",
            "newly_unsafe_pairs",
        )
        if (
            type(study.get("pairs")) is not int
            or study["pairs"] != n
            or any(type(study.get(k)) is not int or not 0 <= study[k] <= n for k in keys)
            or study["baseline_passes"] + study["rescued_pairs"] - study["regressed_pairs"]
            != study["evaluated_passes"]
            or study.get("net_success_gain") != study["evaluated_passes"] - study["baseline_passes"]
        ):
            raise ValueError(
                "whole finite original eight-course and two retention pairings required"
            )
    if positive["baseline_passes"] != 4:
        raise ValueError("unchanged original four-passing-teacher benchmark required")
    if (
        positive["regressed_pairs"]
        or positive["newly_unsafe_pairs"]
        or positive["evaluated_passes"] < positive["baseline_passes"]
    ):
        action = "FIT_ONCE_ON_ORIGINAL_STUDENT_STATES"
    elif positive_dense_nonfoot:
        action = "REVIEW_DENSE_CONTACT_COLLISIONS_WITHOUT_AUTOMATIC_FIT"
    elif any(r["regressed_pairs"] or r["newly_unsafe_pairs"] for r in retention):
        action = "REVIEW_APPROACH_APPLICABILITY_WITHOUT_CONTACT_REFIT"
    else:
        action = "MATCHED_ZERO_AND_THREE_SEED_PHYSICAL_VERIFICATION"
    return dict(
        schema="soccer.rsi.reference_aggregation_recommendation.v1",
        action=action,
        finite_numeric_fit_recommended=action == "FIT_ONCE_ON_ORIGINAL_STUDENT_STATES",
        maximum_new_numeric_candidates=1,
        maximum_optimizer_steps_per_candidate=128,
        query_body_distribution="original unablated student's actually recorded states",
        recommendation_only=True,
        data_admission_authorized=False,
        neural_training_authorized=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
