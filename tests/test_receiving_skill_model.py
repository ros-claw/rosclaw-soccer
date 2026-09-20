from dataclasses import replace

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_context import ReceivingContext
from rosclaw_soccer.training.receiving_skill_model import ReceivingSkillModel, ReceivingSkillSample


def model():
    context = ReceivingContext(0.6, 0.3, 0.1, 0.115, 0.5, -0.5, 0, 0.05)
    samples = tuple(
        ReceivingSkillSample(
            hash_json(["course", i]),
            hash_json(["evidence", i]),
            hash_json("policy"),
            hash_json("physics"),
            "red.playmaker",
            context,
            i < 4,
        )
        for i in range(16)
    )
    return ReceivingSkillModel(samples)


def query(m, **changes):
    s = m.samples[0]
    return m.query(
        **{
            **dict(
                context=s.context,
                agent_id=s.agent_id,
                policy_hash=s.policy_hash,
                physics_hash=s.physics_hash,
            ),
            **changes,
        }
    )


def test_frequency_comes_from_measured_outcomes_not_readiness():
    m = model()
    result = query(m)
    assert result["status"] == "OBSERVED" and result["observed_fraction"] == 0.25
    assert result["samples"] == 16 and len(set(result["evidence_hashes"])) == 16
    assert result["skill_duration_sec"] is None and result["readiness_after"] is None
    assert not result["transition_model_ready"] and not result["promotion_authorized"]
    assert ReceivingSkillModel(m.samples).model_hash == m.model_hash


@pytest.mark.parametrize("field", ["policy_hash", "physics_hash", "agent_id", "context"])
def test_unmatched_context_is_unknown_not_zero_or_success(field):
    m = model()
    value = {
        "policy_hash": hash_json("other"),
        "physics_hash": hash_json("other"),
        "agent_id": "blue.playmaker",
        "context": replace(m.samples[0].context, ball_speed_mps=3),
    }[field]
    result = query(m, **{field: value})
    assert result["status"] == "UNKNOWN" and result["observed_fraction"] is None


@pytest.mark.parametrize(
    "fault", ["duplicate_course", "duplicate_evidence", "policy", "physics", "time"]
)
def test_model_does_not_pool_different_worlds_or_replays(fault):
    rows = list(model().samples)
    if fault.startswith("duplicate"):
        key = "course_hash" if fault.endswith("course") else "evidence_hash"
        rows[-1] = replace(rows[-1], **{key: getattr(rows[0], key)})
    elif fault in {"policy", "physics"}:
        rows[-1] = replace(rows[-1], **{fault + "_hash": hash_json("other")})
    else:
        rows[-1] = replace(rows[-1], context=replace(rows[-1].context, time_sec=0.8))
    with pytest.raises(ValueError):
        ReceivingSkillModel(tuple(rows))
