"""Synthetic refinement transcripts; not actual native evidence."""

from copy import deepcopy

import pytest

from rosclaw_soccer.rsi.two_body_pass_response import refinement_rank
from scripts.rsi_two_body_response_common import verify_refinement


def outcome(error, speed, strict=False):
    return dict(
        whole_body_healthy=True,
        passer_full100NNrecovery=True,
        passer_clean_foot=True,
        B_legal_functional_receive_before_deadline=True,
        B_excluded_contact=False,
        combined_precision_and_functional_receive=strict,
        incoming_arrival_speed_mps=speed,
        incoming=dict(
            A_first_foot_strike_substep=100,
            B_first_ball_contact_substep=200,
            incoming_target_error_3d_before_B_contact_m=error,
        ),
    )


def fixture():
    plans = []
    anchors = []
    report = dict(records=[], selections=[])
    for task in (0, 1):
        course = dict(context=20, state_id=20, task=task, latent=[[0.0] * 6 for _ in range(2)])
        initial = dict(
            row=dict(course=course, outcome=outcome(0.008, 3.8)),
            tag=f"anchor{task}",
            latent=course["latent"],
        )
        anchors.append(initial)
        candidates = [
            dict(
                candidate_index=i,
                latent=[[0.01 * (i + 1)] * 6 for _ in range(2)],
                predicted_response=[0.0, 0.0, 0.0, 0.6],
            )
            for i in range(6)
        ]
        plan = dict(task=task, new_distinct_candidates=candidates)
        plans.append(plan)

        def row(z, tag, role, index, prediction, o, task=task, course=course):
            r = dict(
                index=len(report["records"]),
                course=course | dict(latent=z),
                tag=tag,
                task=task,
                role=role,
                model_candidate_index=index,
                predicted_response=prediction,
                actual_response=[0.0, 0.0, 0.0, 0.6],
                prefix_identity_fields=38,
                neural_actor_optimizer_updates=0,
                outcome=deepcopy(o),
            )
            report["records"].append(r)
            return dict(row=r, tag=tag, latent=z)

        values = [initial]
        for p in candidates:
            i = p["candidate_index"]
            o = (
                outcome(0.12, 2.8)
                if i == 0
                else outcome(0.09, 2.9, True)
                if i == 1
                else outcome(0.001, 2.8, True)
                if i == 2
                else outcome(0.3, 3.4)
            )
            values.append(
                row(p["latent"], f"t{task}-model{i}", "proposal", i, p["predicted_response"], o)
            )
        selected = max(
            enumerate(values), key=lambda kv: (refinement_rank(kv[1]["row"]["outcome"]), -kv[0])
        )[1]
        for repeat in range(2):
            row(
                selected["latent"],
                f"t{task}-repeat{repeat}",
                "selected_repeat",
                None,
                None,
                selected["row"]["outcome"],
            )
        report["selections"].append(
            dict(
                task=task,
                selected=selected["tag"],
                latent=selected["latent"],
                rank=list(refinement_rank(selected["row"]["outcome"])),
                outcome=deepcopy(selected["row"]["outcome"]),
            )
        )
    return report, plans, anchors


def test_complete_all_proposal_and_repeat_transcript_reconstructs_actual_joint_selection():
    r, p, a = fixture()
    verify_refinement(r, p, a)
    assert [s["selected"] for s in r["selections"]] == ["t0-model2", "t1-model2"]


@pytest.mark.parametrize(
    "fault", ["missing", "latent", "prediction", "selection", "repeat", "prefix", "optimizer"]
)
def test_modified_incomplete_or_relabelled_refinement_cannot_pass(fault):
    r, p, a = fixture()
    if fault == "missing":
        r["records"].pop()
    elif fault == "latent":
        r["records"][0]["course"]["latent"] = [[0.99] * 6 for _ in range(2)]
    elif fault == "prediction":
        r["records"][0]["predicted_response"] = [9.0] * 4
    elif fault == "selection":
        r["selections"][0]["selected"] = "invented"
    elif fault == "repeat":
        r["records"][6]["outcome"]["combined_precision_and_functional_receive"] = False
    elif fault == "prefix":
        r["records"][0]["prefix_identity_fields"] = 0
    else:
        r["records"][0]["neural_actor_optimizer_updates"] = 1
    with pytest.raises(ValueError):
        verify_refinement(r, p, a)
