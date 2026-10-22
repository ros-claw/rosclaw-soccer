"""Past-only SIM receiving boundary; NOT the stricter clean-foot task grade."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.receiving_handoff_opportunity import body_contact_kind


class OnlineReceivingBoundary:
    def __init__(self, world: Any, prefix: str) -> None:
        self.world, self.prefix = world, prefix
        self.last_control = world.control_steps
        self.contact_seen = False
        self.excluded_contact = False
        self.consecutive = 0
        self.first_ready_control: int | None = None

    def update(self) -> bool:
        w = self.world
        if w.control_steps != self.last_control + 1:
            raise ValueError("one new actual shared control per receiving observation required")
        for name, force in w.last_ball_contact_bodies[self.prefix].items():
            if not np.isfinite(force) or force <= 1:
                raise ValueError("actual native >1N contact witnesses required")
            if body_contact_kind(name, prefix=self.prefix) == "excluded_or_unknown":
                self.excluded_contact = True
            else:
                self.contact_seen = True
        q, v = w.layouts[self.prefix].canonical(w.data)
        delta = q[36:39] - q[:3]
        qw, x, y, z = q[3:7]
        front = float(
            np.array([1 - 2 * (y * y + z * z), 2 * (x * y + qw * z), 2 * (x * z - qw * y)]) @ delta
        )
        good = bool(
            self.contact_seen
            and not self.excluded_contact
            and not w.failed
            and np.linalg.norm(delta[:2]) <= 0.65
            and np.linalg.norm(v[35:37] - v[:2]) <= 0.6
            and q[38] <= 0.3
            and front >= 0.05
            and q[2] >= 0.65
            and 1 - 2 * (q[4] ** 2 + q[5] ** 2) >= 0.9
        )
        self.consecutive = self.consecutive + 1 if good else 0
        self.last_control = w.control_steps
        if self.consecutive >= 15 and self.first_ready_control is None:
            self.first_ready_control = w.control_steps
        return good and self.consecutive >= 15
