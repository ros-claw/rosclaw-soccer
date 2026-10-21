import copy

import pytest
import torch

from rosclaw_soccer.rsi.paid_distilled_actor import validate_distilled_state
from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH


def evidence():
    initial = dict(
        mean=torch.zeros(1, 160), denominator=torch.ones(1, 160), weight=torch.zeros(29, 128)
    )
    config = dict(epochs=128, teacher_episodes=16, DEV_episodes_used=0, freeze_rnn=False)
    counts = dict(
        optimizer_updates=512, teacher_forward_rows=3810 * 128, rehearsal_forward_rows=3810 * 128
    )
    flags = dict(
        hardware_authorized=False, promotion_authorized=False, commercial_publicity_authorized=False
    )
    state = dict(
        schema="soccer.paid_cartesian_distilled.v1",
        epoch=128,
        config=config,
        counts=counts,
        network=copy.deepcopy(initial),
        source_actor_hash=PUBLISHED_HASH,
        source_head="source",
        declaration_hash="decl",
        body_scene_hash="scene",
        teacher_hash="teacher",
        teacher_audit_hash="audit",
        optimizer=dict(
            param_groups=[dict(params=list(range(16)))],
            state={
                i: dict(step=torch.tensor(512), exp_avg=torch.zeros(1), exp_avg_sq=torch.zeros(1))
                for i in range(16)
            },
        ),
        **flags,
    )
    receipt = dict(
        status="PAID_CARTESIAN_DISTILLATION_TRAINED_NOT_PHYSICALLY_EXAMINED",
        config=config,
        counts=counts,
        source_head="source",
        native_steps=0,
        CPU_neural_only_exam_required=True,
        input_file_hashes={"teacher": "teacher", "audit": "audit"},
        **flags,
    )
    return state, receipt, initial


def test_completed_research_candidate_admitted_not_promoted():
    state, receipt, initial = evidence()
    validate_distilled_state(state, receipt, initial, declaration_hash="decl", scene_hash="scene")
    assert receipt["promotion_authorized"] is False


@pytest.mark.parametrize(
    "change", ["smoke", "norm", "nan", "shape", "body", "counts", "audit", "authority", "Adam"]
)
def test_incomplete_corrupt_cross_body_or_self_authorized_candidates_rejected(change):
    state, receipt, initial = evidence()
    if change == "smoke":
        state["epoch"] = 1
    if change == "norm":
        state["network"]["mean"] += 0.01
    if change == "nan":
        state["network"]["weight"][0, 0] = float("nan")
    if change == "shape":
        state["network"]["weight"] = torch.zeros(28, 128)
    if change == "body":
        state["body_scene_hash"] = "another"
    if change == "counts":
        state["counts"] = dict(state["counts"], optimizer_updates=0)
    if change == "audit":
        state["teacher_audit_hash"] = "unbound"
    if change == "authority":
        state["hardware_authorized"] = True
    if change == "Adam":
        state["optimizer"]["state"][0]["step"] = torch.tensor(0)
    with pytest.raises(ValueError):
        validate_distilled_state(
            state, receipt, initial, declaration_hash="decl", scene_hash="scene"
        )
