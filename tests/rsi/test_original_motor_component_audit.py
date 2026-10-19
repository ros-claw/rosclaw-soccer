"""Actual toy dynamics/synthetic policy, NOT real G1 motor qualification."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.recorded_early_episode import RecordedEarlyEpisode
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.feedback_branch_control import BODY_SHAPES
from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
from rosclaw_soccer.rsi.foot_heading_evidence import FootHeadingCpuReconstruction
from rosclaw_soccer.rsi.original_motor_component_audit import OriginalMotorComponentAudit
from rosclaw_soccer.rsi.recurrent_sampling_motor import CompiledRecurrentSamplingMotor
from rosclaw_soccer.sim.contracts import hash_bytes
from tests.rsi.test_causal_impact_replay import build
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.mark.parametrize(
    "bad,heading",
    [
        (None, False),
        (None, True),
        ("recurrent_residual_hidden_state", False),
        ("foot_heading_state", False),
        ("joint_target_rad", False),
    ],
)
def test_composed_original_motor_is_recomputed_from_only_own_past(
    monkeypatch,
    imitation_parent,  # noqa: F811
    bad,
    heading,
):  # noqa: F811
    producer = build(monkeypatch, imitation_parent)
    if heading:
        producer.ik = FootHeadingCpuReconstruction(producer.model, FootHeadingContactProbe(True))
    records = [producer.step(f).arrays for f in range(300)]
    trace = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    trace["early_impact_feedback_requested_offset_nm"] = np.zeros((300, 1, 29))
    trace["early_impact_feedback_applied_offset_nm"] = np.zeros((300, 1, 10, 29))
    decoder = CompiledRecurrentSamplingMotor(producer.policy)
    audit = OriginalMotorComponentAudit(
        producer.model,
        policy=producer.policy,
        decoder=decoder,
        heading=FootHeadingContactProbe(heading),
        profile=BoundedFirstTouchControl(),
    )
    seen = []
    original = CompiledRecurrentSamplingMotor.delta_at_frame

    def checked(self, policy, body, **kw):
        f = kw["frame"]
        assert set(body) == set(BODY_SHAPES)
        assert all(v.shape == (f + 1, 1, *BODY_SHAPES[k]) for k, v in body.items())
        assert all(not np.shares_memory(v, trace[k]) for k, v in body.items())
        seen.append(f)
        return original(self, policy, body, **kw)

    monkeypatch.setattr(CompiledRecurrentSamplingMotor, "delta_at_frame", checked)
    binary = np.empty(mujoco.mj_sizeModel(producer.model), dtype=np.uint8)
    mujoco.mj_saveModel(producer.model, None, binary)
    physics = RecordedEarlyEpisode(
        producer.model,
        world_hash=hash_bytes(binary.tobytes()),
        kp=producer.kp,
        kd=producer.kd,
        course=(2.5, 0.0, 0.2),
        defaults=trace["joint_position_rad"][0, 0].copy(),
    )
    if bad is not None:
        trace[bad][31, 0, 0] += 0.01
        with pytest.raises(ValueError, match="motor differs"):
            physics.verify(trace, motor_audit=audit)
        assert audit.failed and physics.failed and physics.attempted_steps == 310
        with pytest.raises(ValueError, match="once-only"):
            physics.verify(trace, motor_audit=audit)
    else:
        r = physics.verify(trace, motor_audit=audit)
        assert (
            r["actual_physical_steps"] == 3000 and r["original_ik_and_recurrent_motor_recomputed"]
        )
        assert (
            r["motor_review"]["motor_decisions"] == 300
            and r["motor_review"]["recurrent_updates"] == 270
        )
        assert max(r["motor_review"]["maximum_absolute_errors"].values()) == 0
        assert seen == list(range(300))
        assert not r["motor_review"]["action_search_reconstructed"] and not r["teacher_qualified"]


@pytest.mark.parametrize("bad", ["frame", "pending", "failed"])
def test_wrong_before_clock_is_terminal_without_accessing_a_body(bad):
    a = object.__new__(OriginalMotorComponentAudit)
    a.failed = bad == "failed"
    a.pending = bad == "pending"
    a.next_frame = 0
    with pytest.raises(ValueError, match="no retry"):
        a.before_frame(None, 1 if bad == "frame" else 0, {})
    assert a.failed


def test_invalid_completed_forces_cannot_be_retried():
    a = object.__new__(OriginalMotorComponentAudit)
    a.pending = True
    a.failed = True
    a.next_frame = 0
    calls = []
    a.ik = SimpleNamespace(after_step=lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="no retry"):
        a.after_frame(0, np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError, match="no retry"):
        a.after_frame(0, np.zeros(6))
    assert a.failed and not a.pending and calls == []
