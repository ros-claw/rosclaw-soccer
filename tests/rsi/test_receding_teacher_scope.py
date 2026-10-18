import numpy as np
import pytest

from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import planning_contexts
from rosclaw_soccer.rsi.receding_teacher_scope import (
    SOURCE_GROUPS,
    receding_contexts,
    receding_jobs,
    reproduce_zero_control,
    scope_contract,
)


def plan():
    return PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), "sha256:" + "a" * 64, 18)


def test_whole_six_course_domain_and_original_c6_c7_identity_are_preserved():
    p = plan()
    contexts = receding_contexts(p)
    assert [c.source_group for c in contexts] == list(SOURCE_GROUPS)
    assert contexts[:2] == planning_contexts(p)[6:]
    assert all(2.6 <= c.course[0] <= 2.8 and 0.25 <= c.course[2] <= 0.7 for c in contexts)
    assert len({c.context_hash for c in contexts}) == 6


def test_ten_bodies_reuse_only_two_old_none_controls_and_reproduce_archived_c7_first():
    jobs = receding_jobs()
    assert jobs[0]["group"] == 77 and jobs[0]["option_index"] == 2
    assert sum(j["role"] == "teacher" for j in jobs) == 6
    assert sum(j["role"] == "control" for j in jobs) == 4
    assert {j["group"] for j in jobs if j["role"] == "control"} == {108, 109, 140, 141}
    report = scope_contract(plan())
    assert report["combined_body_substeps"] == 60000
    assert not report["teacher_qualified"] and not report["fresh_exam"]
    assert not report["training_authorized"]


def test_changed_source_generator_does_not_silently_redefine_courses():
    with pytest.raises(ValueError):
        receding_contexts(
            PublicFirstTouchTrainingPlan(plan().seeds, plan().behavior_model_hash, 17)
        )


def test_reproduction_is_full_original_array_equality_not_only_outcome_or_qpos():
    source = dict(qpos=np.zeros((300, 1, 43)), targets=np.zeros((300, 1, 29)))
    old = dict(source, cerebellar_sampling_mean=np.zeros((300, 1, 10, 29)))
    assert reproduce_zero_control(source, old)["whole_original_arrays_bit_equal"]
    changed = dict(source, targets=source["targets"].copy())
    changed["targets"][299, 0, 28] = 1e-12
    with pytest.raises(ValueError):
        reproduce_zero_control(changed, old)


@pytest.mark.parametrize("tamper", ["shape", "dtype", "missing", "extra", "scalar"])
def test_no_partial_reproduction_or_extra_capture(tamper):
    old = dict(qpos=np.zeros((300, 1, 43)), target=np.zeros((300, 1, 29)))
    new = dict(old)
    if tamper == "shape":
        new["qpos"] = new["qpos"][:299]
    elif tamper == "dtype":
        new["target"] = new["target"].astype(np.float32)
    elif tamper == "missing":
        new.pop("target")
    elif tamper == "extra":
        new["unregistered"] = np.zeros((300, 1, 1))
    else:
        new["target"] = np.array(0.0)
    with pytest.raises(ValueError):
        reproduce_zero_control(new, old)
