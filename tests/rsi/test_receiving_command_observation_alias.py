"""Record a known GENERALIZATION limitation, not a fixed or physical success.

Within the current96 bank the direction is fixed: this counterexample does NOT
by itself explain current native precision loss. It proves that the present112
interface cannot represent the teacher for all legitimate communicated tasks.
"""

import numpy as np
from test_continuous_velocity_stream import fake_policy

from rosclaw_soccer.providers.g1.velocity_policy import phase_delta
from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
from rosclaw_soccer.rsi.receiving_command_actor import command_inputs


def test_same112_can_hide_different_communicated_pass_plane_and_capture_target():
    q, v = np.zeros(43), np.zeros(41)
    q[2], q[3] = 0.78, 1
    q[36:39], v[35:38] = [-3, 3, 0.11], [3, -3, 0]
    anchor = np.array([0.25, -0.15, 0.11])
    inputs, commands = [], []
    for offset in (0.0, 0.5):
        angle = -np.pi / 4 + offset
        forward = np.array([np.cos(angle), np.sin(angle)])
        lateral = np.array([-forward[1], forward[0]])
        target = anchor[:2] - 0.25 * forward - 0.12 * lateral
        x, _, _, _ = command_inputs(
            q,
            v,
            np.zeros(3),
            params=fake_policy().params,
            last_action=np.zeros(29, np.float32),
            phase=phase_delta(),
            home_xy=np.zeros(2),
            pass_direction=forward,
            anchor_world=anchor,
            launch_seen=True,
            foot_contact_seen=False,
            other_body_contact_seen=False,
        )
        teacher = receiving_intent(
            root_xy=q[:2],
            yaw=0.0,
            ball_xy=q[36:38],
            ball_velocity_xy=v[35:37],
            home_xy=np.zeros(2),
            pass_direction=forward,
            launch_seen=True,
            ready_anchor_xy=anchor[:2],
            capture_target_xy=target,
            mode="projected-capture-v3",
        )
        inputs.append(x)
        commands.append(np.array(teacher.command_body))
    np.testing.assert_array_equal(inputs[0], inputs[1])
    assert not np.array_equal(commands[0], commands[1])
    assert commands[0][0] > 0 and commands[1][0] == -0.5
