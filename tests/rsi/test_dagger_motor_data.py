import numpy as np
import pytest

from rosclaw_soccer.rsi.dagger_motor_data import same_state_actor_mask, same_state_teacher_sequence
from rosclaw_soccer.rsi.motor_training_protocol import checkpoint_step_pair
from rosclaw_soccer.rsi.neural_motor_student import recorded_student_sequence


def sample():
    q = np.zeros((300, 43), dtype=np.float64)
    q[:, 2], q[:, 3], q[:, 39] = 0.8, 1, 1
    actions = np.arange(300, dtype=np.float32)[:, None] * np.ones((1, 29), dtype=np.float32) / 300
    raw = dict(canonical_qpos=q, canonical_qvel=np.zeros((300, 41)), consumed_action_isaac=actions)
    x, _ = recorded_student_sequence(dict(**raw, neural_action_isaac=actions))
    raw.update(
        student_features=x,
        mixed_consumed_action_isaac=actions.copy(),
        teacher_advice_action_isaac=np.full((300, 29), 2, dtype=np.float32),
    )
    return raw


def test_advice_is_label_not_previous_executed_observation():
    raw = sample()
    x, y = same_state_teacher_sequence(raw)
    np.testing.assert_array_equal(y, raw["teacher_advice_action_isaac"])
    np.testing.assert_array_equal(x[1:, 79:], raw["consumed_action_isaac"][:-1])
    assert not np.shares_memory(y, raw["teacher_advice_action_isaac"])


@pytest.mark.parametrize("key", ["student_features", "mixed_consumed_action_isaac"])
def test_wrong_history_or_consumed_action_rejected(key):
    raw = sample()
    raw[key][30, 1] += 0.01
    with pytest.raises(ValueError, match="binding differs"):
        same_state_teacher_sequence(raw)


def test_protocols_do_not_relabel_previous_training_as_new_updates():
    assert checkpoint_step_pair("bc3000") == (0, 3000)
    assert checkpoint_step_pair("dagger4500") == (3000, 4500)
    with pytest.raises(ValueError):
        checkpoint_step_pair("anything")


def test_all40_split_and_failure_masks_are_preserved():
    rows = [
        dict(
            course=dict(course_index=i),
            split="train" if i < 32 else "validation",
            score=dict(qualified=True),
            prefix_diagnostics=dict(
                prefix_physical_safe=True, post_boundary_control_can_cause_first_touch=True
            ),
        )
        for i in range(40)
    ]
    originals = [dict(course=r["course"], split=r["split"]) for r in rows]
    rows[2]["score"]["qualified"] = False
    rows[3]["prefix_diagnostics"]["prefix_physical_safe"] = False
    rows[4]["prefix_diagnostics"]["post_boundary_control_can_cause_first_touch"] = False
    mask = same_state_actor_mask(rows, originals)
    assert mask.shape == (40,) and mask.sum() == 37
    rows[33]["split"] = "train"
    with pytest.raises(ValueError, match="original whole-course split"):
        same_state_actor_mask(rows, originals)


def test_truthy_text_is_not_safety_evidence():
    rows = [
        dict(
            course=dict(course_index=i),
            split="train",
            score=dict(qualified=True),
            prefix_diagnostics=dict(
                prefix_physical_safe="false", post_boundary_control_can_cause_first_touch=True
            ),
        )
        for i in range(40)
    ]
    with pytest.raises(ValueError, match="typed safety"):
        same_state_actor_mask(rows, rows)
