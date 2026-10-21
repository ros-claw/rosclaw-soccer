"""Synthetic new40 admission and finite clock tests, no robot or physics."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_native_contact_adapter import Base

from rosclaw_soccer.rsi.approach_option_checkpoint import (
    IDENTITY_FIELDS,
    NativeApproachContactSession,
    validate_approach40_state,
)
from rosclaw_soccer.rsi.native_contact_adapter import NativeResidualContactActor


def fixture():
    parent = NativeResidualContactActor(Base()).state_dict()
    identities = {k: k + "-expected" for k in IDENTITY_FIELDS}
    actor = {k: v.clone() for k, v in parent.items()}
    actor["adapter.4.bias"] += 0.01
    state = dict(
        **identities,
        schema="soccer.rsi.native_approach_contact_recovery_ppo.v1",
        protocol="native-approach-measured-contact-recovery60-ppo-v1",
        episode_count=40,
        mode="train40",
        training_seed=7608,
        actor_observation_dim=177,
        critic_observation_dim=177,
        native_pd_position_targets=29,
        activation_ceiling="SIM_ONLY",
        gamma=0.995,
        GAE_lambda=0.95,
        maximum_normalized_residual=0.2,
        fresh_critic=True,
        fresh_optimizer=True,
        optimizer_resumed=False,
        hardware_authorized=False,
        promotion_authorized=False,
        counts=dict(
            value_only_steps=128,
            PPO_steps=123,
            optimizer_completed=251,
            optimizer_attempted=251,
            actor_update_steps=123,
            head_policy_rows=100000,
            head_critic_rows=198720,
            recovery_foundation_rows=76800,
        ),
        native_steps=1768000,
        actor_state_dict=actor,
        policy_state_dict={
            **{"actor." + k: v.clone() for k, v in actor.items()},
            **{"baseline." + k: v.clone() for k, v in parent.items()},
            "log_std": torch.zeros(29),
        },
    )
    return state, identities, parent


def test_admits_new40_parent132_baseline_not_legacy_zero_adapter_baseline():
    state, ids, parent = fixture()
    parent["adapter.4.bias"] += 0.02
    state["policy_state_dict"]["baseline.adapter.4.bias"] = parent["adapter.4.bias"].clone()
    validate_approach40_state(state, identities=ids, parent_actor=parent)


def test_new_exam_does_not_read_missing_legacy_receipt_fields():
    from scripts.rsi_exam_native_contact_policy import legacy_session_options

    assert legacy_session_options({}, option=True, continued=False, approach=True) == {}
    source = {
        k: k
        for k in (
            "recovery_proof_hash",
            "recovery_fault_hash",
            "resume_checkpoint_hash",
            "resume_source_hash",
            "resume_proof_hash",
        )
    }
    result = legacy_session_options(source, option=True, continued=True, approach=False)
    assert result == {"expected_" + k: v for k, v in source.items()}


@pytest.mark.parametrize(
    "mutation",
    [
        "episode",
        "schema",
        "warmup",
        "identity",
        "base",
        "baseline",
        "copy",
        "nan",
        "count",
        "hardware",
    ],
)
def test_rejects_partial_legacy_mismatched_or_unbound_candidates(mutation):
    state, ids, parent = fixture()
    if mutation == "episode":
        state["episode_count"] = 24
    elif mutation == "schema":
        state["schema"] = "soccer.rsi.native_contact24_recovery60_continuation_ppo.v1"
    elif mutation == "warmup":
        state["counts"]["value_only_steps"] = 64
    elif mutation == "identity":
        state["course_proof_hash"] = "another"
    elif mutation == "base":
        state["actor_state_dict"]["base.head.weight"] += 1
        state["policy_state_dict"]["actor.base.head.weight"] += 1
    elif mutation == "baseline":
        state["policy_state_dict"]["baseline.adapter.4.bias"] += 1
    elif mutation == "copy":
        state["policy_state_dict"]["actor.adapter.4.bias"] += 1
    elif mutation == "nan":
        state["actor_state_dict"]["adapter.4.bias"][0] = float("nan")
    elif mutation == "count":
        state["native_steps"] += 10
    else:
        state["hardware_authorized"] = True
    with pytest.raises(ValueError):
        validate_approach40_state(state, identities=ids, parent_actor=parent)


def test_actual_clock_can_include_recovery_past300_but_never_replay_or_extend360():
    session = NativeApproachContactSession.__new__(NativeApproachContactSession)
    session.params = SimpleNamespace(
        defaults=np.zeros(29), action_scale=np.ones(29) * 0.25, action_offset=np.zeros(29)
    )
    session.model = torch.nn.Linear(177, 29)
    session.last_action = np.zeros(29, dtype=np.float32)
    session.phase = np.float32(0.2)
    session.next_frame = 300
    session.neural_calls = session.attempted_neural_calls = 0
    session.faulted = False
    q, v, omega = np.zeros(43), np.zeros(41), np.zeros(3)
    q[3] = 1.0
    command, ref = np.zeros(3, dtype=np.float32), np.zeros(79, dtype=np.float32)
    session._propose(q, v, omega, command, ref, frame=300)
    assert session.next_frame == 301 and session.neural_calls == 1
    at_end = copy.copy(session)
    at_end.next_frame = 360
    with pytest.raises(ValueError, match="actual head"):
        at_end._propose(q, v, omega, command, ref, frame=360)
    with pytest.raises(ValueError, match="actual head"):
        session._propose(q, v, omega, command, ref, frame=300)
    assert session.faulted
