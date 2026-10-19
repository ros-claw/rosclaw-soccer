"""Actual toy integration with synthetic neural weights, NOT G1 skill evidence."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.rsi import causal_impact_replay as mod
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
from rosclaw_soccer.rsi.recurrent_sampling_motor import (
    CompiledRecurrentSamplingMotor,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.rsi.recurrent_success_motor import make_model
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.skills.team.navigation_envelope import SimulationNavigationEnvelope
from tests.physics.test_g1_feedback_forecast import world
from tests.providers.test_sonic_branch_state import tracker as toy_tracker
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def build(monkeypatch, parent_model, *, early=False, reuse_owned_early_updates=False):
    sim, _, _ = world()
    sim.opt.gravity[:] = 0
    sim.dof_armature[6:35] = 0.02
    sim.jnt_range[:] = [-2.0, 2.0]
    binary = np.zeros(mujoco.mj_sizeModel(sim), dtype=np.uint8)
    mujoco.mj_saveModel(sim, None, binary)
    foundation, _, _ = toy_tracker()
    foundation.model.qualification.qualification_hash = "synthetic"
    foundation.model.qualification.require_eligible = lambda: None

    class Navigation:
        def __init__(self, root, agent, config):
            assert config.replan_frames == 20 and config.maximum_frames == 300
            self.agent_id = agent
            self.navigation_envelope = SimulationNavigationEnvelope(
                agent, "sha256:" + "a" * 64, 1.5
            )
            reference = np.zeros((512, 36), dtype=np.float32)
            reference[:, 2] = 0.793
            reference[:, 3] = 1
            reference[:, 7:] = G1SonicRunupController.default_angles
            self.backend = SimpleNamespace(
                kp=np.full(29, 10.0),
                kd=np.full(29, 0.2),
                default_angles=G1SonicRunupController.default_angles.copy(),
                qualification=SimpleNamespace(require_eligible=lambda: None),
                reference=reference,
                events=[],
            )
            self.observations = []

        def start_from_observation(self, observation):
            assert observation.frame == 0

        def prepare_batched_proposal(self, observation):
            self.observations.append(observation)

        def commit_batched_action(self, action):
            # Synthetic target law, not an official SONIC decoding contract.
            return SimpleNamespace(
                target_rad=np.asarray(G1SonicRunupController.default_angles) + action * 0.01
            )

    monkeypatch.setattr(mod, "G1SonicNavigation", Navigation)
    policy = make_preview(make_sampling_view(make_model(parent_model), seed=901))
    return mod.CausalImpactReplay(
        sim,
        world_hash=hash_bytes(binary.tobytes()),
        kp=np.full(29, 10.0),
        kd=np.full(29, 0.2),
        model_root=Path("synthetic"),
        foundation=foundation.model,
        expected_foundation_hash="synthetic",
        policy=policy,
        decoder=CompiledRecurrentSamplingMotor(policy),
        expected_policy_hash=policy["policy_hash"],
        course=(2.5, 0.0, 0.2),
        profile=BoundedFirstTouchControl(),
        heading=FootHeadingContactProbe(False),
        impact_enabled=False,
        early_impact_enabled=early,
        early_evidence_root=Path("synthetic-output") if early else None,
        reuse_owned_early_updates=reuse_owned_early_updates,
    )


def test_current_body_full300_loop_integrates_and_recomputes_neural_inputs(
    monkeypatch,
    imitation_parent,  # noqa: F811
):  # noqa: F811
    replay = build(monkeypatch, imitation_parent)
    frames = [replay.step(f) for f in range(300)]
    result = replay.review()
    assert result["native_attempted_steps"] == result["native_integrated_steps"] == 3000
    assert result["forecast_attempted_steps"] == 0
    assert result["current_body_policy_recomputed"]
    assert not result["source_future_trace_or_targets_consumed"]
    assert not result["teacher_qualified"] and not result["independently_implemented_neural_policy"]
    assert len(replay.navigation.observations) == 300
    assert replay.tracker._next_frame == 300
    assert replay.decoder._memory.last_frame == 299
    assert frames[30].arrays["recurrent_sampling_performed"].all()
    assert not np.array_equal(
        frames[30].arrays["foundation_neural_decoder_input"],
        frames[31].arrays["foundation_neural_decoder_input"],
    )
    assert frames[-1].arrays["torque_nm"].shape == (1, 10, 29)
    assert replay.data.time == pytest.approx(6.0)


@pytest.mark.parametrize("frame", [1, -1, True, 0.0, 300])
def test_wrong_clock_fails_without_physics_or_silent_retry(frame):
    replay = object.__new__(mod.CausalImpactReplay)
    replay.next_frame = replay.attempted_steps = 0
    replay.failed = False
    with pytest.raises(ValueError, match="next causal frame"):
        replay.step(frame)
    assert replay.attempted_steps == 0 and replay.failed
    with pytest.raises(ValueError):
        replay.step(0)


def test_integration_failure_is_terminal_and_charges_attempt(monkeypatch, imitation_parent):  # noqa: F811
    replay = build(monkeypatch, imitation_parent)

    def fail(*args):
        raise RuntimeError("synthetic integrator failure")

    monkeypatch.setattr(mujoco, "mj_step", fail)
    with pytest.raises(RuntimeError):
        replay.step(0)
    assert replay.attempted_steps == 1 and replay.integrated_steps == 0 and replay.failed
    with pytest.raises(ValueError):
        replay.step(0)


def test_no_recorded_or_future_state_input_in_driver_api():
    forbidden = {"trace", "future_target", "source_contact_frame", "callback", "reference_trace"}
    assert not forbidden.intersection(inspect.signature(mod.CausalImpactReplay).parameters)
    assert list(inspect.signature(mod.CausalImpactReplay.step).parameters) == ["self", "frame"]


def test_owned_continuation_is_opt_in_and_propagated_only_to_early_adapter(
    monkeypatch,
    imitation_parent,  # noqa: F811
):
    captured = []

    class Early:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs["reuse_owned_updates"])

    monkeypatch.setattr(mod, "EarlyImpactController", Early)
    monkeypatch.setattr(mod, "EarlyImpactEvidence", lambda root: object())
    build(monkeypatch, imitation_parent, early=True)
    build(monkeypatch, imitation_parent, early=True, reuse_owned_early_updates=True)
    assert captured == [False, True]


@pytest.mark.parametrize("early,value", [(False, True), (True, 1), (True, "yes")])
def test_owned_option_requires_boolean_and_early_mode_before_model_allocation(early, value):
    # Deliberately no model assets: invalid options fail before any allocation.
    with pytest.raises(ValueError, match="fixed study"):
        mod.CausalImpactReplay(
            None,
            world_hash="unused",
            kp=None,
            kd=None,
            model_root=Path("unused"),
            foundation=None,
            expected_foundation_hash="unused",
            policy={},
            decoder=None,
            expected_policy_hash="unused",
            course=(2.7, 0.1, 0.3),
            profile=None,
            heading=None,
            impact_enabled=False,
            early_impact_enabled=early,
            early_evidence_root=Path("unused") if early else None,
            reuse_owned_early_updates=value,
        )


def test_early_adapter_applies_three_frame_torque_and_preserves_distinct_trace_namespace(
    monkeypatch,
    imitation_parent,  # noqa: F811
):  # noqa: F811
    from rosclaw_soccer.rsi.early_impact_controller import EarlyImpactCommand

    recorded = []

    class Evidence:
        def __init__(self, root):
            assert root == Path("synthetic-output")

        def record(self, frame, command):
            assert frame == len(recorded)
            recorded.append(frame)

    class Early:
        def __init__(self, *a, **kw):
            self.forecaster = SimpleNamespace(attempted_steps=0, integrated_steps=0)

        def plan(self, data, target, **kw):
            frame = kw["frame"]
            assert data.time == pytest.approx(frame * 0.02)
            offset = np.zeros(29)
            if 30 <= frame <= 32:
                offset[3] = 3.0
            return EarlyImpactCommand(offset, {"forecast_substeps": 0}, ())

    monkeypatch.setattr(mod, "EarlyImpactController", Early)
    monkeypatch.setattr(mod, "EarlyImpactEvidence", Evidence)
    native_step = mujoco.mj_step

    def checked_step(engine_model, data):
        assert recorded[-1] == round(data.time / 0.002) // 10
        native_step(engine_model, data)

    monkeypatch.setattr(mujoco, "mj_step", checked_step)
    replay = build(monkeypatch, imitation_parent, early=True)
    frames = [replay.step(f) for f in range(300)]
    report = replay.review()
    assert recorded == list(range(300))
    assert report["native_integrated_steps"] == 3000
    assert report["forecast_integrated_steps"] == 0 and not report["teacher_qualified"]
    for frame in frames:
        values = frame.arrays
        assert "impact_feedback_requested_offset_nm" not in values
        expected = np.zeros((1, 29))
        if 30 <= frame.frame <= 32:
            expected[0, 3] = 3.0
        np.testing.assert_array_equal(values["early_impact_feedback_requested_offset_nm"], expected)
        raw = values["pd_requested_torque_nm"]
        limits = replay.torque_limits
        np.testing.assert_array_equal(
            values["torque_nm"], np.clip(raw + expected[:, None], -limits, limits)
        )
