import numpy as np
import pytest
from test_ball_surface_tasks import fixture

from rosclaw_soccer.rsi.contact_transition_diagnostics import diagnose_contact_transition


def test_timing_uses_native_force_and_geometry_uses_previous_native_state():
    model, data, layout = fixture()
    q = np.repeat(data.qpos[None], 50, axis=0)
    q[:, layout.bq] += np.arange(50) * 0.001
    forces = np.zeros((50, 1, 3))
    forces[30, 0, 0], forces[36, 0, 2] = 2.0, 4.0
    report = diagnose_contact_transition(
        model, layout, dict(qpos=q, qvel=np.zeros((50, model.nv)), ball_forces=forces)
    )
    assert report["foot_to_nonfoot_ms"] == 12
    assert report["first_foot_native_index"] == 30
    assert report["observed_nonfoot_samples"] == 1
    assert report["observed_nonfoot_peak_n"] == 4
    for state in report["states"]:
        n = state["native_index"]
        assert state["geometry_state_native_index"] == n - 1
        np.testing.assert_array_equal(state["ball_position_m"], q[n - 1, layout.bq : layout.bq + 3])
        assert state["nearest_nonfoot_surfaces"]
    assert data.time == 0 and report["new_physical_steps"] == 0
    assert not report["collision_body_force_identity_reconstructed"]


def test_first_sample_has_no_fabricated_preintegration_pose():
    model, data, layout = fixture()
    forces = np.zeros((1, 1, 3))
    forces[0, 0, 2] = 3
    result = diagnose_contact_transition(
        model,
        layout,
        dict(qpos=data.qpos[None].copy(), qvel=data.qvel[None].copy(), ball_forces=forces),
    )
    assert result["first_foot_native_index"] is None
    assert result["states"] == [
        dict(event="first_nonfoot", native_index=0, geometry_status="INITIAL_STATE_NOT_SAVED")
    ]


def test_negative_contact_evidence_is_rejected():
    model, data, layout = fixture()
    with pytest.raises(ValueError, match="complete finite"):
        diagnose_contact_transition(
            model,
            layout,
            dict(
                qpos=data.qpos[None].copy(),
                qvel=data.qvel[None].copy(),
                ball_forces=-np.ones((1, 1, 3)),
            ),
        )
