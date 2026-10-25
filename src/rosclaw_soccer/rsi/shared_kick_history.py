"""Read an owned shared-world boundary, including the actually applied target."""

from rosclaw_soccer.providers.g1.kick_measured_history import KickMeasuredSample
from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld


def measured_kick_sample(world: SharedNativeMotorWorld, prefix: str) -> KickMeasuredSample:
    """Capture after a committed control; do not infer an old policy action.

    One boundary sample is not a full policy history. The existing installer
    requires five consecutive samples and verifies the actual entry state.
    """
    world._require_current_boundary()
    if prefix not in world.layouts or world.control_steps < 1:
        raise ValueError("own player after a committed actual control required")
    layout = world.layouts[prefix]
    q, v = layout.canonical(world.data)
    # Verify angular-frame semantics, then record the exact codec consumed by
    # SharedRoboNaldoPlayer._observe (Jacobian roundoff is not bit-identical).
    layout.angular_velocity_body(world.data)
    return KickMeasuredSample(
        frame=world.control_steps,
        time_sec=float(world.data.time),
        previous_target_frame=world.control_steps - 1,
        joint_position=tuple(map(float, q[7:36])),
        joint_velocity=tuple(map(float, v[6:35])),
        angular_velocity=tuple(map(float, v[3:6])),
        pelvis_position=tuple(map(float, q[:3])),
        pelvis_quaternion=tuple(map(float, q[3:7])),
        ball_position=tuple(map(float, world.data.xpos[layout.ball])),
        previous_pd_target=tuple(map(float, world.last_applied_target(prefix))),
    )
