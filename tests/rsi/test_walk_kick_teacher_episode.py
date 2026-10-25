from types import SimpleNamespace

import numpy as np
from test_shared_native_motor_world import source_xml

from rosclaw_soccer.rsi import walk_kick_teacher_episode as episode
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene


def test_actual_physical_failure_is_kept_without_fake_padding_or_abort(monkeypatch):
    import mujoco

    class Stream:
        def __init__(self, *args, **kwargs):
            self.params = SimpleNamespace(
                defaults=np.zeros(29),
                stiffness=np.full(29, 20.0),
                damping=np.ones(29),
            )
            self.neural_calls = 0

        def propose(self, q, v, angular, command, *, frame):
            self.neural_calls += 1
            return np.zeros(29)

    monkeypatch.setattr(episode, "ContinuousVelocityStream", Stream)
    xml = source_xml().replace('gravity="0 0 0"', 'gravity="0 0 -9.81"')
    xml = xml.replace('name="ball"', 'name="soccer_ball"')
    model = mujoco.MjModel.from_xml_string(shared_native_scene(xml, prefixes=("a/",)))
    row, arrays = episode.evaluate_episode(
        model=model,
        assets=dict(
            velocity_model=None,
            velocity_config=None,
            velocity_model_hash="unused",
            velocity_config_hash="unused",
        ),
        policy_types=(None, None, None, np.arange(29)),
        course=(2.25, 0.08, 0.4),
        parameters=dict(
            speed_mps=0.5, entry_gap_m=0.9, reference_frame=200, root_lateral_offset_m=0.0
        ),
        hard_limits=np.full(29, 25.0),
    )
    assert not row["healthy_full_horizon"]
    assert row["controls"] < 300 and row["native_steps"] == row["controls"] * 10
    assert arrays["qpos"].shape[0] == row["native_steps"]
    assert arrays["pelvis_z_per_substep_m"].shape[0] == row["controls"]
    assert row["outcome"]["event_status"] == "PHYSICALLY_TERMINATED"
    assert row["outcome"]["forward_60_m"] is None
    assert row["optimizer_updates"] == 0
