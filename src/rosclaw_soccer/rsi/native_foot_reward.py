"""Causal near-foot task reward; not a replacement for the unchanged exam gate."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FootStepEvidence:
    frame: int
    gap_before_m: float
    gap_after_m: float
    ball_x_before_m: float
    ball_x_after_m: float
    ball_y_before_m: float
    ball_y_after_m: float
    ball_vx_before_mps: float
    ball_vx_after_mps: float
    foot_force_n: float
    nonfoot_force_n: float
    minimum_height_m: float
    minimum_upright: float


class NearFootReward:
    """No football progress reward until measured native FOOT contact occurs.

    Nonfoot exclusion belongs to this narrow foot-only experiment, not all
    football roles/rules. Falls remain data; this never terminates physics.
    """

    def __init__(self, *, initial_ball_y_m: float) -> None:
        if type(initial_ball_y_m) not in (int, float) or not math.isfinite(initial_ball_y_m):
            raise ValueError("finite own boundary ball required")
        self.initial_y = float(initial_ball_y_m)
        self.next_frame = 20
        self.touched = False
        self.illegal = False
        self.fallen = False
        self.finalized = False
        self.anchor_x = 0.0
        self.forward = 0.0
        self.lateral = 0.0

    def step(self, e: FootStepEvidence) -> tuple[float, dict[str, float]]:
        if (
            type(e.frame) is not int
            or e.frame != self.next_frame
            or not 20 <= e.frame < 300
            or self.finalized
        ):
            raise ValueError("consecutive own frame20..299 evidence required")
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v)
                for k, v in e.__dict__.items()
                if k != "frame"
            )
            or min(e.foot_force_n, e.nonfoot_force_n, e.gap_before_m, e.gap_after_m) < 0
        ):
            raise ValueError("finite native metric and nonnegative gap/contact required")
        terms = dict(
            time=-0.001,
            approach=0.0,
            contact=0.0,
            momentum=0.0,
            forward=0.0,
            lateral=0.0,
            nonfoot=0.0,
            balance=0.0,
        )
        if e.nonfoot_force_n > 1 and not self.illegal:
            self.illegal = True
            terms["nonfoot"] = -5.0
        if (e.minimum_height_m < 0.65 or e.minimum_upright < 0.9) and not self.fallen:
            self.fallen = True
            terms["balance"] = -5.0
        if not self.illegal and not self.fallen:
            if not self.touched:
                terms["approach"] = 2 * float(np.clip(e.gap_before_m - e.gap_after_m, -0.05, 0.05))
                if e.foot_force_n > 1:
                    self.touched = True
                    self.anchor_x = e.ball_x_before_m
                    self.lateral = min(abs(e.ball_y_before_m - self.initial_y), 1.0)
                    terms["contact"] = 0.5
                    terms["momentum"] = min(
                        max(e.ball_vx_after_mps - e.ball_vx_before_mps, 0.0), 1.0
                    )
            if self.touched:
                forward = float(np.clip(e.ball_x_after_m - self.anchor_x, -1, 2))
                lateral = min(abs(e.ball_y_after_m - self.initial_y), 1.0)
                terms["forward"] = 2 * (forward - self.forward)
                terms["lateral"] = -2 * (lateral - self.lateral)
                self.forward, self.lateral = forward, lateral
        self.next_frame += 1
        return float(sum(terms.values())), terms

    def terminal(self, score: dict[str, Any], *, causal_first_touch_possible: bool) -> float:
        if (
            self.next_frame != 300
            or self.finalized
            or type(score["qualified"]) is not bool
            or type(causal_first_touch_possible) is not bool
        ):
            raise ValueError("full H300 original terminal gate required once")
        if score["qualified"] and (not self.touched or self.illegal or self.fallen):
            raise ValueError("reward contact/balance evidence disagrees with strict task gate")
        self.finalized = True
        return (
            5.0
            if score["qualified"] and causal_first_touch_possible
            else -3.0
            if self.illegal or self.fallen
            else 0.0
        )


class NativeFootRewardObserver:
    """FK on PRIVATE data: never forward/reset the live world to query a gap."""

    def __init__(self, body: Any) -> None:
        import mujoco

        self.model = body.model
        self.kinematic = mujoco.MjData(body.model)
        self.ball_geom = body.bg
        self.feet = body.contacts[:4]
        root = int(self.model.body_rootid[body.pelvis])
        self.nonfeet = np.flatnonzero(
            (self.model.body_rootid == root) & ~np.isin(np.arange(self.model.nbody), self.feet)
        )
        self.geoms = [
            i
            for i in range(self.model.ngeom)
            if int(self.model.geom_bodyid[i]) in self.feet
            and (
                (int(self.model.geom_contype[i]) & int(self.model.geom_conaffinity[self.ball_geom]))
                or (
                    int(self.model.geom_contype[self.ball_geom])
                    & int(self.model.geom_conaffinity[i])
                )
            )
        ]
        if not self.geoms or not len(self.nonfeet):
            raise ValueError("native solid foot geometry/body classification required")

    def gap(self, body: Any) -> float:
        import mujoco

        self.kinematic.qpos[:] = body.data.qpos
        mujoco.mj_kinematics(self.model, self.kinematic)
        distance = min(
            mujoco.mj_geomDistance(self.model, self.kinematic, self.ball_geom, g, 3.0, None)
            for g in self.geoms
        )
        if not math.isfinite(distance):
            raise ValueError("nonfinite native closest foot/ball geometry")
        return max(0.0, float(distance))

    def evidence(
        self, row: dict[str, np.ndarray], *, frame: int, gap_before: float, gap_after: float
    ) -> FootStepEvidence:
        q = row["target_body_canonical_qpos_after_substep"][0]
        v = row["target_body_canonical_qvel_after_substep"][0]
        forces = row["ball_contact_force_by_body_n"][0]
        before = row["canonical_qpos"][0]
        return FootStepEvidence(
            frame,
            gap_before,
            gap_after,
            float(before[36]),
            float(q[-1, 36]),
            float(before[37]),
            float(q[-1, 37]),
            float(row["canonical_qvel"][0, 35]),
            float(v[-1, 35]),
            float(forces[:, self.feet].max()),
            float(forces[:, self.nonfeet].max()),
            float(q[:, 2].min()),
            float((1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)).min()),
        )
