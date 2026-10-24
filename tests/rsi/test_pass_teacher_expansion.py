"""Synthetic transcript contract, not physical teacher evidence."""

from copy import deepcopy

import pytest

from scripts.rsi_pass_teacher_expansion_common import verify_expansion


def outcome(error, speed, strict=False):
    return dict(
        whole_body_healthy=True,
        passer_full100NNrecovery=True,
        passer_clean_foot=True,
        B_excluded_contact=False,
        B_legal_functional_receive_before_deadline=True,
        combined_precision_and_functional_receive=strict,
        incoming_arrival_speed_mps=speed,
        incoming=dict(
            A_first_foot_strike_substep=900,
            B_first_ball_contact_substep=1100,
            incoming_target_error_3d_before_B_contact_m=error,
        ),
    )


def fixture():
    cal = dict(records=[])
    plans = []
    report = dict(records=[], selections=[])
    for task in (0, 1):
        course = dict(
            state_id=21, task=task, context=20, assisted=True, latent=[[0.0] * 6 for _ in range(2)]
        )
        zero = dict(course=course, student_prefix_controls=0, outcome=outcome(0.1, 3.4))
        pulse = dict(
            course=course | dict(latent=[[0.2] * 6 for _ in range(2)]),
            student_prefix_controls=80,
            outcome=outcome(0.2, 3.8),
        )
        cal["records"].extend([zero, pulse])
        plan = dict(
            state_id=21,
            task=task,
            donor_state=23,
            donor_controlled=True,
            course=course | dict(latent=[[0.4] * 6 for _ in range(2)]),
        )
        plans.append(plan)
        o = outcome(0.8, 3.8) if task == 0 else outcome(0.05, 2.9, True)
        proposal = dict(
            index=len(report["records"]),
            course=plan["course"],
            tag=f"t{task}-transfer",
            task=task,
            role="transfer",
            proposal_donor_state=23,
            proposal_donor_controlled=True,
            prefix_identity_fields=27,
            neural_actor_optimizer_updates=0,
            outcome=o,
        )
        report["records"].append(proposal)
        best = zero if task == 0 else proposal
        for repeat in range(2):
            report["records"].append(
                proposal
                | dict(
                    index=len(report["records"]),
                    course=best["course"],
                    tag=f"t{task}-repeat{repeat}",
                    role="selected_repeat",
                    outcome=deepcopy(best["outcome"]),
                )
            )
        report["selections"].append(
            dict(
                task=task,
                selected="cal-t0-p0" if task == 0 else "t1-transfer",
                latent=best["course"]["latent"],
                outcome=best["outcome"],
                transfer_selected=task == 1,
            )
        )
    return report, plans, cal


def test_reconstructs_both_transfer_and_honest_fallback():
    report, plans, cal = fixture()
    verify_expansion(report, plans, cal, 21)


@pytest.mark.parametrize(
    "defect", ["missing", "donor", "fallback", "latent", "repeat", "optimizer"]
)
def test_incomplete_or_fabricated_expansion_rejected(defect):
    report, plans, cal = fixture()
    if defect == "missing":
        report["records"].pop()
    elif defect == "donor":
        report["records"][0]["proposal_donor_state"] = 29
    elif defect == "fallback":
        report["selections"][0]["transfer_selected"] = True
    elif defect == "latent":
        report["records"][0]["course"] = dict(
            report["records"][0]["course"], latent=[[0.0] * 6 for _ in range(2)]
        )
    elif defect == "repeat":
        report["records"][1]["outcome"]["incoming_arrival_speed_mps"] = 2.0
    else:
        report["records"][0]["neural_actor_optimizer_updates"] = 1
    with pytest.raises(ValueError):
        verify_expansion(report, plans, cal, 21)
