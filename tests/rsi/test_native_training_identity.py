import copy
import json
from dataclasses import asdict

import pytest

from rosclaw_soccer.rsi.native_training_identity import require_native_training_identity
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.sim.contracts import hash_json


def case():
    plan = PublicFirstTouchTrainingPlan((20261060,), "sha256:" + "a" * 64, 18)
    draw = plan.draws()[0]
    view = dict(
        schema="soccer.rsi.recurrent_motor_sampling.v1",
        seed=draw.sampling_seed,
        mean_model=dict(model_hash=plan.behavior_model_hash),
        model_hash="sha256:" + "b" * 64,
    )
    body = dict(
        seed=draw.seed,
        lane=draw.lane,
        course=list(draw.course),
        partition="DECLARED_PUBLIC_TRAINING",
        entry_source_hash="entry",
        source_hash="kernel",
        step_model_hash=view["model_hash"],
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
        public_training_declaration=json.loads(json.dumps(plan.contract())),
    )
    envelope = dict(
        schema="rosclaw.growth.shared_json_payload.v1",
        location=["mean_model"],
        stripped_document=dict(model_hash=view["model_hash"]),
    )
    envelope["envelope_hash"] = hash_json(envelope)
    return body, view, envelope, plan


def check(body, view, envelope, plan):
    require_native_training_identity(
        body, view, envelope, plan, 0, entry_source_hash="entry", kernel_source_hash="kernel"
    )


def test_json_round_trip_does_not_create_false_fatal():
    body, view, envelope, plan = case()
    assert body["public_training_declaration"] != plan.contract()
    assert hash_json(body["public_training_declaration"]) == hash_json(plan.contract())
    assert asdict(plan.draws()[0])["course"] != body["course"]
    check(body, view, envelope, plan)


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", False),
        ("lane", True),
        ("source_hash", "wrong"),
        ("entry_source_hash", "wrong"),
        ("step_model_hash", "wrong"),
        ("partition", "FRESH"),
        ("activation_ceiling", "REAL"),
        ("promotion_authorized", 0),
        ("hardware_authorized", True),
        ("course", [1.0, 2.0, 3.0]),
    ],
)
def test_identity_mutations_still_rejected(key, value):
    body, view, envelope, plan = case()
    body[key] = value
    with pytest.raises(ValueError):
        check(body, view, envelope, plan)


def test_plan_and_sampling_seed_changes_rejected():
    body, view, envelope, plan = case()
    changed = copy.deepcopy(body)
    changed["public_training_declaration"]["jobs"][0]["course"][0] += 0.001
    with pytest.raises(ValueError):
        check(changed, view, envelope, plan)
    view["seed"] += 1
    with pytest.raises(ValueError):
        check(body, view, envelope, plan)


def test_envelope_tampering_rejected():
    body, view, envelope, plan = case()
    envelope["location"] = ["other"]
    with pytest.raises(ValueError):
        check(body, view, envelope, plan)
