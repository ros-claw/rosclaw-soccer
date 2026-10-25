"""An explicit absolute episode clock over an owned future's NEW step counter.

No world counter, body state, phase or time is rewritten. Nonzero origins are
limited to the H360 SIM experiment and bound to the native physical timestamp.
"""

from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld


def absolute_control_frame(world: SharedNativeMotorWorld, *, control_origin: int = 0) -> int:
    world._require_current_boundary()
    if type(control_origin) is not int or not 0 <= control_origin <= 360:
        raise ValueError("explicit bounded integer SIM episode control origin required")
    frame = control_origin + world.control_steps
    if control_origin and (frame > 360 or abs(world.data.time - 0.02 * frame) > 1e-8):
        raise ValueError("owned future origin must match the uninterrupted native episode time")
    return frame
