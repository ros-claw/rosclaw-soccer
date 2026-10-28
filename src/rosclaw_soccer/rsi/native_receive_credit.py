"""SIM-only online credit for contact followed by sustained measured ball control."""

import numpy as np


class NativeReceiveCredit:
    """Count actual native contacts, never turn proximity alone into possession.

    Each call is one 20ms control with ten 2ms force samples. A capture credit
    requires 10 consecutive *post-contact* controls within the declared bounds.
    It is a shaping signal, not a match success or safety certification.
    """

    def __init__(self, worlds: int):
        if type(worlds) is not int or worlds < 1:
            raise ValueError("positive explicit world count required")
        self.worlds = worlds
        self.next_step = 0
        self.foot_seen = np.zeros(worlds, dtype=bool)
        self.nonfoot_seen = np.zeros(worlds, dtype=bool)
        self.capture_emitted = np.zeros(worlds, dtype=bool)
        self.stable_controls = np.zeros(worlds, dtype=np.int64)

    def step(self, *, step, forces, active, ball_root_distance_m, relative_speed_mps, healthy):
        f, a, dist, speed, good = map(
            np.asarray, (forces, active, ball_root_distance_m, relative_speed_mps, healthy)
        )
        if (
            type(step) is not int
            or step != self.next_step
            or f.shape != (self.worlds, 10, 3)
            or a.shape != (self.worlds,)
            or a.dtype != np.bool_
            or good.shape != (self.worlds,)
            or good.dtype != np.bool_
            or dist.shape != (self.worlds,)
            or speed.shape != (self.worlds,)
            or not all(np.isfinite(x).all() for x in (f, dist, speed))
            or np.any(f < 0)
            or np.any(dist < 0)
            or np.any(speed < 0)
        ):
            raise ValueError("ordered finite actual native receiving data required")
        first_foot = np.zeros(self.worlds, dtype=np.float32)
        first_nonfoot = np.zeros(self.worlds, dtype=np.float32)
        foot_seen_before_control = self.foot_seen.copy()
        for j in range(10):
            bad = a & (f[:, j, 2] > 1) & ~self.nonfoot_seen
            self.nonfoot_seen |= bad
            first_nonfoot[bad] = 1
            foot = a & (np.max(f[:, j, :2], axis=1) > 10) & ~self.foot_seen
            self.foot_seen |= foot
            first_foot[foot & ~self.nonfoot_seen] = 1
        eligible = (
            a
            & foot_seen_before_control
            & ~self.nonfoot_seen
            & good
            & (dist <= 0.65)
            & (speed <= 0.5)
        )
        self.stable_controls[a] = np.where(eligible[a], self.stable_controls[a] + 1, 0)
        capture = eligible & (self.stable_controls >= 10) & ~self.capture_emitted
        self.capture_emitted |= capture
        self.next_step += 1
        return dict(
            first_clean_foot=first_foot,
            first_nonfoot=first_nonfoot,
            sustained_capture=capture.astype(np.float32),
            foot_seen=self.foot_seen.copy(),
            contaminated=self.nonfoot_seen.copy(),
            stable_controls=self.stable_controls.copy(),
        )
