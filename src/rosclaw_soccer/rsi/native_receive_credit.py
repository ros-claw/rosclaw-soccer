"""SIM-only online credit for contact followed by sustained measured ball control."""

import numpy as np


def planar_ball_root_distance(ball: np.ndarray, root: np.ndarray) -> np.ndarray:
    """Measured horizontal separation; both positions still must be valid 3-D."""
    if (
        not isinstance(ball, np.ndarray)
        or not isinstance(root, np.ndarray)
        or ball.ndim != 2
        or ball.shape[1] != 3
        or root.shape != ball.shape
        or not np.isfinite(ball).all()
        or not np.isfinite(root).all()
    ):
        raise ValueError("aligned finite measured 3-D ball and root positions required")
    return np.linalg.norm(ball[:, :2] - root[:, :2], axis=1)


class NativeReceiveCredit:
    """Count actual native contacts, never turn proximity alone into possession.

    Each call is one 20ms control with ten 2ms force samples. A capture credit
    requires 10 consecutive *post-contact* controls within the declared bounds.
    Distance is horizontal ball-center to pelvis, not a 3-D distance that
    would incorrectly penalize a standing robot for its own pelvis height.
    It is a shaping signal, not a match success or safety certification.
    """

    def __init__(self, worlds: int, *, strict_native: bool = False):
        if type(worlds) is not int or worlds < 1 or type(strict_native) is not bool:
            raise ValueError("positive explicit world count required")
        self.worlds = worlds
        self.strict_native = strict_native
        self.next_step = 0
        self.foot_seen = np.zeros(worlds, dtype=bool)
        self.nonfoot_seen = np.zeros(worlds, dtype=bool)
        self.capture_emitted = np.zeros(worlds, dtype=bool)
        self.stable_controls = np.zeros(worlds, dtype=np.int64)
        self.stable_native_samples = np.zeros(worlds, dtype=np.int64)

    def step(
        self,
        *,
        step,
        forces,
        active,
        ball_root_planar_distance_m,
        relative_speed_mps,
        healthy,
        native_planar_distance_m=None,
        native_relative_speed_mps=None,
    ):
        f, a, dist, speed, good = map(
            np.asarray, (forces, active, ball_root_planar_distance_m, relative_speed_mps, healthy)
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
        if self.strict_native:
            native_dist = np.asarray(native_planar_distance_m)
            native_speed = np.asarray(native_relative_speed_mps)
            if (
                native_dist.shape != (self.worlds, 10)
                or native_speed.shape != (self.worlds, 10)
                or not np.isfinite(native_dist).all()
                or not np.isfinite(native_speed).all()
                or np.any(native_dist < 0)
                or np.any(native_speed < 0)
            ):
                raise ValueError("all finite 10-substep receiving measurements required")
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
        if self.strict_native:
            native_base = a & foot_seen_before_control & ~self.nonfoot_seen & good
            capture = np.zeros(self.worlds, dtype=bool)
            for j in range(10):
                sample_eligible = (
                    native_base & (native_dist[:, j] <= 0.65) & (native_speed[:, j] <= 0.5)
                )
                self.stable_native_samples[a] = np.where(
                    sample_eligible[a], self.stable_native_samples[a] + 1, 0
                )
                capture |= sample_eligible & (self.stable_native_samples >= 100)
            capture &= ~self.capture_emitted
        else:
            capture = eligible & (self.stable_controls >= 10) & ~self.capture_emitted
        self.capture_emitted |= capture
        self.next_step += 1
        result = dict(
            first_clean_foot=first_foot,
            first_nonfoot=first_nonfoot,
            sustained_capture=capture.astype(np.float32),
            foot_seen=self.foot_seen.copy(),
            contaminated=self.nonfoot_seen.copy(),
            stable_controls=self.stable_controls.copy(),
        )
        if self.strict_native:
            result["stable_native_samples"] = self.stable_native_samples.copy()
        return result
