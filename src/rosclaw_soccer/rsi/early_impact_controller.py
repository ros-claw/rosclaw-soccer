"""Prospective80ms SIM proposals with a strictly bounded60ms action lease.

No future source event or trajectory input, training, policy qualification or
hardware entry. Actual H300/retention evidence is still required separately.
"""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches
from rosclaw_soccer.physics.g1_early_feedback_forecast import (
    G1EarlyFeedbackForecast,
    G1EarlyFeedbackForecaster,
)
from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
from rosclaw_soccer.rsi.early_feedback_impact_score import early_feedback_impact_score
from rosclaw_soccer.rsi.feedback_branch_control import G1FeedbackBranchControl
from rosclaw_soccer.rsi.impact_torque_screen import impact_torque_candidates

MAX_BASELINE_WINDOWS = 32
MAX_FORECAST_STEPS = 1800  # 32 baselines+12 nonzero actions+one repeatedzero, each80ms.


@dataclass(frozen=True)
class EarlyImpactCandidate:
    name: str
    offset_nm: np.ndarray
    forecast: G1EarlyFeedbackForecast
    score: dict[str, Any]


@dataclass(frozen=True)
class EarlyImpactCommand:
    offset_nm: np.ndarray
    report: dict[str, Any]
    candidates: tuple[EarlyImpactCandidate, ...]


def early_trigger_status(
    frame: int,
    *,
    side: int,
    contact_frame: int | None,
    gap_m: float,
    pelvis_z_m: float,
    screened: bool,
    baseline_windows: int,
) -> str | None:
    if (
        type(frame) is not int
        or not 0 <= frame < 300
        or type(side) is not int
        or side not in (-1, 0, 1)
        or (
            contact_frame is not None
            and (type(contact_frame) is not int or not 0 <= contact_frame < frame)
        )
        or type(screened) is not bool
        or type(baseline_windows) is not int
        or not 0 <= baseline_windows <= MAX_BASELINE_WINDOWS
        or not np.isfinite([gap_m, pelvis_z_m]).all()
    ):
        raise ValueError("finite actual current-phase trigger inputs required")
    if screened:
        return "ALREADY_SCREENED_THIS_EPISODE"
    if baseline_windows == MAX_BASELINE_WINDOWS:
        return "BASELINE_SEARCH_BUDGET_EXHAUSTED"
    if not 30 <= frame <= 294 or frame // 20 != (frame + 3) // 20:
        return "OUTSIDE_NO_REPLAN80MS_WINDOW"
    if side < 0 or contact_frame is not None:
        return "NO_CURRENT_PRECONTACT_SWING_FOOT"
    if pelvis_z_m < 0.65:
        return "CURRENT_PELVIS_BELOW_TASK_FLOOR"
    if not -0.02 <= gap_m <= 0.18:
        return "OUTSIDE_FIXED_CURRENT_SURFACE_GAP"
    return None


def same_arrays(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    return set(left) == set(right) and all(
        left[k].shape == right[k].shape
        and left[k].dtype == right[k].dtype
        and left[k].tobytes() == right[k].tobytes()
        for k in left
    )


class EarlyImpactController:
    def __init__(self, model: Any, *, world_hash: str, kp: np.ndarray, kd: np.ndarray) -> None:
        self.model = model
        self.forecaster = G1EarlyFeedbackForecaster(
            model, world_hash=world_hash, kp=kp, kd=kd, max_forecast_steps=MAX_FORECAST_STEPS
        )
        self.sampler = G1JointTaskKinematicSampler(model, expected_world_hash=world_hash)
        self.next_frame = self.baseline_windows = 0
        self.screened = self.failed = False
        self.held_offset = np.zeros(29)
        self.action_end_frame = 0
        self.last_candidates: list[EarlyImpactCandidate] = []
        self.ball_geom = int(model.geom("ball_geom").id)
        self.feet = [int(model.body(s + "_ankle_roll_link").id) for s in ("left", "right")]

    def plan(
        self,
        data: Any,
        current_target_rad: np.ndarray,
        *,
        frame: int,
        tracker: Any,
        decoder: Any,
        ik: Any,
        policy: dict[str, Any],
        body_prefix: dict[str, np.ndarray],
        previous_motor_delta_rad: np.ndarray,
    ) -> EarlyImpactCommand:
        if (
            self.failed
            or type(frame) is not int
            or frame != self.next_frame
            or not 0 <= frame < 300
        ):
            self.failed = True
            raise ValueError("one contiguous current-frame decision required; no retry")
        self.failed = True
        if (
            not isinstance(data, mujoco.MjData)
            or data.model is not self.model
            or data is self.forecaster._scratch
            or not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or not np.isclose(data.time, frame * 0.02, rtol=0, atol=1e-8)
            or not isinstance(current_target_rad, np.ndarray)
            or current_target_rad.shape != (29,)
            or current_target_rad.dtype != np.float64
            or not np.isfinite(current_target_rad).all()
            or np.max(abs(current_target_rad)) > 10
        ):
            raise ValueError("finite actual current same-world body/target required")
        self.last_candidates = []
        candidates = self.last_candidates
        before = self.forecaster.attempted_steps

        def finish(status: str, selected: int | None = None) -> EarlyImpactCommand:
            self.next_frame += 1
            self.failed = False
            offset = self.held_offset.copy() if frame < self.action_end_frame else np.zeros(29)
            return EarlyImpactCommand(
                offset,
                dict(
                    schema="soccer.rsi.prospective_early80ms_impact_plan.v1",
                    frame=frame,
                    status=status,
                    selected_index=selected,
                    selected_name=None if selected is None else candidates[selected].name,
                    action_frames_remaining_including_current=max(0, self.action_end_frame - frame),
                    requested_offset_nm=offset.tolist(),
                    baseline_windows=self.baseline_windows,
                    forecast_substeps=self.forecaster.attempted_steps - before,
                    accumulated_forecast_substeps=self.forecaster.attempted_steps,
                    future_source_frame_or_target_used=False,
                    full_h300_gain_proven=False,
                    neural_policy_trained=False,
                    teacher_qualified=False,
                    hardware_authorized=False,
                    promotion_authorized=False,
                ),
                tuple(candidates),
            )

        nonfoot = any(
            int(self.model.geom_bodyid[w.other_geom_id]) not in [0, *self.feet]
            and float(np.linalg.norm(w.force_world_n)) > 1
            for w in read_geom_contact_wrenches(self.model, data, geom_id=self.ball_geom)
        )
        if nonfoot or data.qpos[2] < 0.65:
            self.held_offset.fill(0)
            self.action_end_frame = frame
            # Never resume an interrupted action or search a replacement bank.
            self.screened = True
            return finish("CANCELLED_CURRENT_UNSAFE_CONTACT_OR_PELVIS")
        if frame < self.action_end_frame:
            return finish("HELD_BOUNDED60MS_ACTION")
        if self.screened:
            return finish("ALREADY_SCREENED_THIS_EPISODE")
        sample = self.sampler.sample(data)
        side = ik.side
        gap = (
            float(sample.foot_ball_signed_gap_m[side])
            if type(side) is int and side in (0, 1)
            else 0.0
        )
        reason = early_trigger_status(
            frame,
            side=side,
            contact_frame=ik.contact,
            gap_m=gap,
            pelvis_z_m=float(data.qpos[2]),
            screened=self.screened,
            baseline_windows=self.baseline_windows,
        )
        if reason is not None:
            return finish(reason)

        def predict(name: str, offset: np.ndarray) -> EarlyImpactCandidate:
            control = G1FeedbackBranchControl(
                self.model,
                frame=frame,
                tracker=tracker,
                decoder=decoder,
                ik=ik,
                policy=policy,
                body_prefix=body_prefix,
                previous_motor_delta_rad=previous_motor_delta_rad,
            )
            forecast = self.forecaster.forecast(
                data, current_target_rad, control=control, early_offset_nm=offset
            )
            result = EarlyImpactCandidate(
                name, offset.copy(), forecast, early_feedback_impact_score(forecast, offset)
            )
            candidates.append(result)
            return result

        self.baseline_windows += 1
        baseline = predict("baseline", np.zeros(29))
        if not baseline.score["feasible_predicted_impact"]:
            return finish("BASELINE_HAS_NO_COMPLETE_SAFE_PREDICTED_IMPACT")
        self.screened = True
        for name, offset in impact_torque_candidates(sample, side)[1:]:
            predict(name, offset)
        repeat = predict("repeat_zero", np.zeros(29))
        for candidate in candidates:
            if (
                candidate.forecast.donor_integration_state_hash
                != baseline.forecast.donor_integration_state_hash
            ):
                raise ValueError("all early candidates must copy same current donor")
        b, r = baseline.forecast, repeat.forecast
        if (
            b.end_integration_state.tobytes() != r.end_integration_state.tobytes()
            or not same_arrays(b.arrays, r.arrays)
            or len(b.next_target_records) != 3
            or len(r.next_target_records) != 3
            or len(b.next_neural_records) != 3
            or len(r.next_neural_records) != 3
            or any(
                not same_arrays(left, right)
                for aa, bb in (
                    (b.next_target_records, r.next_target_records),
                    (b.next_neural_records, r.next_neural_records),
                )
                for left, right in zip(aa, bb, strict=True)
            )
        ):
            raise ValueError("whole early zero repeat dynamics/three neural decisions differ")
        selected, best = min(
            ((i, c) for i, c in enumerate(candidates[:-1]) if c.score["feasible_predicted_impact"]),
            key=lambda row: row[1].score["proxy_cost"],
        )
        if baseline.score["proxy_cost"] - best.score["proxy_cost"] <= 1e-6:
            selected = 0
        if selected:
            self.held_offset = candidates[selected].offset_nm.copy()
            self.action_end_frame = frame + 3
        return finish("PROSPECTIVE_LOCAL_PROXY_SCREENED_NOT_TEACHER_QUALIFIED", selected)
