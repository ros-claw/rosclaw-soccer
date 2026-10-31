import hashlib
from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.skills.team.navigation_option import NavigationObservation, NavigationSlot
from rosclaw_soccer.training.live_navigation_sampler import (
    DeterministicNavigationCandidate,
    LiveNavigationSampler,
)
from rosclaw_soccer.training.local_navigation import build_local_navigation_actor_critic

torch = pytest.importorskip("torch")
HASH = "sha256:" + "a" * 64


def observation(committed):
    return NavigationObservation(
        agent_id="red.finisher",
        frame=0,
        time_sec=0.0,
        role="finisher",
        intent="receive",
        body_pose=(0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0),
        body_velocity=(0.0, 0.0, 0.0),
        ball_position=(0.5, 0.0, 0.11),
        ball_velocity=(-0.4, 0.0, 0.0),
        task_target=(0.5, 0.0, 0.11),
        steering_target=(0.5, 0.0),
        baseline_command=(0.0, 0.0, 0.0),
        previous_command=(0.0, 0.0, 0.0),
        neighbors=(),
        committed_receiver=committed,
    )


def sampler(tmp_path, seed=1, gate="committed_receiver"):
    torch.manual_seed(3)
    model = build_local_navigation_actor_critic()
    path = tmp_path / "actor.npz"
    if not path.exists():
        np.savez_compressed(path, **{k: v.detach().numpy() for k, v in model.state_dict().items()})
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return LiveNavigationSampler(
        path,
        agent_id="red.finisher",
        actor_hash=digest,
        foundation_hash=HASH,
        foundation_config_hash=HASH,
        seed=seed,
        gate=gate,
    )


def test_seeded_proposal_is_bounded_replayable_and_on_policy(tmp_path):
    first, second = sampler(tmp_path), sampler(tmp_path)
    a, b = first.propose(observation(True)), second.propose(observation(True))
    assert a == b
    assert np.linalg.norm(a.velocity_delta[:2]) <= 0.25
    assert np.array_equal(first.samples[0]["raw"], second.samples[0]["raw"])
    record = first.samples[0]
    with torch.no_grad():
        mean, value = first._actor(torch.from_numpy(record["obs"][None]))
        normal = torch.distributions.Normal(mean, first._actor.logstd.clamp(-2.5, -0.3).exp())
        logp = normal.log_prob(torch.from_numpy(record["raw"][None])).sum(1)
    assert np.isclose(record["logp"], float(logp[0]))
    assert np.isclose(record["value"], float(value[0]))


def test_gate_and_slot_do_not_claim_uncommitted_ball(tmp_path):
    model = sampler(tmp_path)
    slot = NavigationSlot(model)
    assert slot.propose(observation(False)) == (0.0, 0.0, 0.0)
    assert not model.samples and not slot.active and not slot.faulted
    slot.propose(replace(observation(True), frame=1, time_sec=0.02))
    assert len(model.samples) == 1 and slot.active and not slot.faulted


def test_invalid_identity_is_rejected(tmp_path):
    model = sampler(tmp_path)
    with pytest.raises(ValueError):
        model.propose(replace(observation(True), agent_id="blue.finisher"))


def test_frozen_candidate_uses_mean_and_distinct_contract(tmp_path):
    exploratory = sampler(tmp_path)
    path = tmp_path / "actor.npz"
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    model = DeterministicNavigationCandidate(
        path,
        agent_id="red.finisher",
        actor_hash=digest,
        foundation_hash=HASH,
        foundation_config_hash=HASH,
    )
    assert model.contract_hash != exploratory.contract_hash
    assert model.propose(observation(False)) is None
    assert model.propose(observation(True)).velocity_delta == (0.0, 0.0, 0.0)
    assert not model.samples


@pytest.mark.parametrize("intent", ["receive", "intercept", "press"])
def test_explicit_ball_intent_gate_only_changes_uncommitted_target(tmp_path, intent):
    default = sampler(tmp_path)
    expanded = sampler(tmp_path, gate="ball_intent")
    state = replace(observation(False), intent=intent)
    assert default.propose(state) is None
    assert expanded.propose(state) is not None
    assert expanded.contract_hash != default.contract_hash


@pytest.mark.parametrize("intent", ["shoot", "recover", "hold"])
def test_intent_gate_does_not_override_other_tasks(tmp_path, intent):
    expanded = sampler(tmp_path, gate="ball_intent")
    assert expanded.propose(replace(observation(False), intent=intent)) is None
    assert not expanded.samples


def test_unknown_gate_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        sampler(tmp_path, gate="everything")
