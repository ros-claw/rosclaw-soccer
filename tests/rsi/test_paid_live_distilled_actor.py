import copy

import pytest
import torch

from rosclaw_soccer.rsi.paid_live_distilled_actor import validate_live_state
from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH


def evidence():
    initial = dict(
        mean=torch.zeros(1, 160),
        denominator=torch.ones(1, 160),
        **{f"p{i}": torch.zeros(i + 1) for i in range(16)},
    )
    config = dict(
        epochs=256,
        variant=0,
        seed=20261022,
        lr=0.0001,
        rehearsal=1.0,
        freeze_rnn=False,
        teacher_episodes=12,
        rehearsal_episodes=16,
        exam_continuous_cases=16,
        DEV_episodes_used=0,
        unique_teacher_neural_rows=2898,
        unique_rehearsal_neural_rows=3810,
        full_episode_BPTT=True,
        teacher_hidden_states_not_inputs=True,
        normalizer_frozen=True,
        active_weight=5.0,
        other_weight=1.0,
        max_grad_norm=1.0,
        loss_units="joint_target_radians_squared",
        teacher_minibatches_per_epoch=3,
        rehearsal_minibatch_sizes=[6, 5, 5],
    )
    counts = dict(
        optimizer_updates=768, teacher_forward_rows=2898 * 256, rehearsal_forward_rows=3810 * 256
    )
    flags = dict(
        hardware_authorized=False, promotion_authorized=False, commercial_publicity_authorized=False
    )
    state = dict(
        schema="soccer.paid_live_distilled.v1",
        epoch=256,
        config=config,
        counts=counts,
        source_actor_hash=PUBLISHED_HASH,
        declaration_hash="decl",
        body_scene_hash="scene",
        source_head="src",
        base_checkpoint_hash="base",
        base_training_receipt_hash="base_receipt",
        teacher_hash="teacher",
        teacher_audit_hash="audit",
        network=copy.deepcopy(initial),
        optimizer=dict(
            param_groups=[dict(params=list(range(16)), lr=0.0001)],
            state={
                i: dict(
                    step=torch.tensor(768.0),
                    exp_avg=torch.zeros(i + 1),
                    exp_avg_sq=torch.zeros(i + 1),
                )
                for i in range(16)
            },
        ),
        **flags,
    )
    receipt = dict(
        status="PAID_LIVE_DISTILLATION_TRAINED_NOT_PHYSICALLY_EXAMINED",
        config=copy.deepcopy(config),
        counts=copy.deepcopy(counts),
        source_head="src",
        native_steps=0,
        CPU_neural_only_full_sequence_exam_required=True,
        input_file_hashes={x: x for x in ("base", "base_receipt", "teacher", "audit")},
        **flags,
    )
    return state, receipt, initial


def test_live_protocol_admits_final_candidate_not_authority():
    s, r, i = evidence()
    validate_live_state(s, r, i, declaration_hash="decl", scene_hash="scene")
    assert not r["promotion_authorized"]


@pytest.mark.parametrize(
    "fault",
    [
        "smoke",
        "old_schema",
        "counts",
        "base",
        "seed",
        "rehearsal",
        "DEV",
        "normalizer",
        "nan",
        "Adam_shape",
        "Adam_step",
        "authority",
        "body",
    ],
)
def test_corrupt_unbound_or_incompatible_candidate_rejected(fault):
    s, r, i = evidence()
    if fault == "smoke":
        s["epoch"] = 1
    elif fault == "old_schema":
        s["schema"] = "soccer.paid_cartesian_distilled.v1"
    elif fault == "counts":
        s["counts"]["optimizer_updates"] = 1024
    elif fault == "base":
        s["base_checkpoint_hash"] = "unbound"
    elif fault == "seed":
        s["config"]["seed"] = 42
    elif fault == "rehearsal":
        s["config"]["rehearsal"] = 0
    elif fault == "DEV":
        s["config"]["DEV_episodes_used"] = 1
    elif fault == "normalizer":
        s["network"]["mean"] += 0.1
    elif fault == "nan":
        s["network"]["p0"][0] = float("nan")
    elif fault == "Adam_shape":
        s["optimizer"]["state"][1]["exp_avg"] = torch.zeros(1)
    elif fault == "Adam_step":
        s["optimizer"]["state"][0]["step"] = torch.tensor(0.0)
    elif fault == "body":
        s["body_scene_hash"] = "other"
    else:
        r["hardware_authorized"] = True
    with pytest.raises(ValueError):
        validate_live_state(s, r, i, declaration_hash="decl", scene_hash="scene")
