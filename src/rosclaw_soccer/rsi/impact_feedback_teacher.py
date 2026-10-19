"""Prospective bounded SIM-only impact proposals, not a qualified teacher.

No source contact frame, future target, private exam, training or hardware API
is accepted. Native full-H300 execution and independent retention exams still
have to qualify this mechanism before any neural distillation.
"""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_feedback_forecast import G1FeedbackForecast, G1FeedbackForecaster
from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
from rosclaw_soccer.rsi.feedback_branch_control import G1FeedbackBranchControl
from rosclaw_soccer.rsi.feedback_impact_score import feedback_impact_score
from rosclaw_soccer.rsi.impact_torque_screen import impact_torque_candidates

MAX_BASELINE_WINDOWS = 32
MAX_FORECAST_STEPS = 900  # 32 baseline windows plus12 candidates andone repeated zero.


@dataclass(frozen=True)
class ImpactCandidate:
    name: str
    offset_nm: np.ndarray
    forecast: G1FeedbackForecast
    score: dict[str, Any]
    next_neural_capture: dict[str, np.ndarray]


@dataclass(frozen=True)
class ImpactFeedbackPlan:
    offset_nm: np.ndarray
    report: dict[str, Any]
    candidates: tuple[ImpactCandidate, ...]


def trigger_status(
    frame: int,
    *,
    side: int,
    contact_frame: int | None,
    gap_m: float,
    pelvis_z_m: float,
    already_screened: bool,
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
        or type(already_screened) is not bool
        or type(baseline_windows) is not int
        or not 0 <= baseline_windows <= MAX_BASELINE_WINDOWS
        or not np.isfinite([gap_m, pelvis_z_m]).all()
    ):
        raise ValueError("finite causal current-phase trigger inputs required")
    if already_screened:
        return "ALREADY_SCREENED_THIS_EPISODE"
    if baseline_windows == MAX_BASELINE_WINDOWS:
        return "BASELINE_SEARCH_BUDGET_EXHAUSTED"
    if not 30 <= frame < 298 or (frame + 1) % 20 == 0:
        return "OUTSIDE_CALIBRATED_NO_REPLAN_WINDOW"
    if contact_frame is not None or side < 0:
        return "NO_CURRENT_PRECONTACT_SWING_FOOT"
    if pelvis_z_m < 0.65:
        return "CURRENT_PELVIS_BELOW_TASK_FLOOR"
    if not -0.02 <= gap_m <= 0.06:
        return "OUTSIDE_FIXED_CURRENT_SURFACE_GAP"
    return None


class ImpactFeedbackTeacher:
    def __init__(self, model: Any, *, world_hash: str, kp: np.ndarray, kd: np.ndarray) -> None:
        self.model = model
        self.forecaster = G1FeedbackForecaster(
            model, world_hash=world_hash, kp=kp, kd=kd, max_forecast_steps=MAX_FORECAST_STEPS
        )
        self.sampler = G1JointTaskKinematicSampler(model, expected_world_hash=world_hash)
        self.baseline_windows = 0
        self.screened = False
        self.last_frame = -1
        self.last_candidates: list[ImpactCandidate] = []

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
    ) -> ImpactFeedbackPlan:
        if type(frame) is not int or not self.last_frame < frame < 300:
            raise ValueError("one strictly increasing current-frame decision required")
        if (
            not isinstance(data, mujoco.MjData)
            or data.model is not self.model
            or data is self.forecaster._scratch
            or not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or not np.isclose(data.time, frame * 0.02, rtol=0, atol=1e-8)
        ):
            raise ValueError("actual current caller-owned same-world body required")
        if (
            not isinstance(current_target_rad, np.ndarray)
            or current_target_rad.shape != (29,)
            or current_target_rad.dtype != np.float64
            or not np.isfinite(current_target_rad).all()
            or np.max(abs(current_target_rad)) > 10
        ):
            raise ValueError("bounded finite actual current joint target required")
        self.last_frame = frame  # Failures consume this decision too; never retry it silently.
        sample = self.sampler.sample(data)
        side = ik.side
        gap = (
            float(sample.foot_ball_signed_gap_m[side])
            if type(side) is int and side in (0, 1)
            else 0.0
        )
        reason = trigger_status(
            frame,
            side=side,
            contact_frame=ik.contact,
            gap_m=gap,
            pelvis_z_m=float(data.qpos[2]),
            already_screened=self.screened,
            baseline_windows=self.baseline_windows,
        )
        candidates: list[ImpactCandidate] = []
        self.last_candidates = (
            candidates  # Completed branches remain inspectable on a later failure.
        )
        before = self.forecaster.attempted_steps

        def finish(status: str, *, selected_index: int | None = None) -> ImpactFeedbackPlan:
            offset = (
                np.zeros(29)
                if selected_index is None
                else candidates[selected_index].offset_nm.copy()
            )
            return ImpactFeedbackPlan(
                offset,
                dict(
                    schema="soccer.rsi.prospective_impact_feedback_plan.v1",
                    frame=frame,
                    causal_swing_side=side,
                    current_surface_gap_m=gap,
                    status=status,
                    selected_index=selected_index,
                    forecast_substeps=self.forecaster.attempted_steps - before,
                    accumulated_forecast_substeps=self.forecaster.attempted_steps,
                    baseline_windows=self.baseline_windows,
                    selected_name=None
                    if selected_index is None
                    else candidates[selected_index].name,
                    candidates=[
                        dict(
                            name=c.name,
                            offset_nm=c.offset_nm.tolist(),
                            score=c.score,
                            donor_hash=c.forecast.donor_integration_state_hash,
                        )
                        for c in candidates
                    ],
                    future_source_frame_or_target_used=False,
                    full_h300_gain_proven=False,
                    teacher_qualified=False,
                    neural_policy_trained=False,
                    hardware_authorized=False,
                    promotion_authorized=False,
                ),
                tuple(candidates),
            )

        if reason is not None:
            return finish(reason)

        def predict(name: str, offset: np.ndarray) -> ImpactCandidate:
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
                data, current_target_rad, control=control, first_offset_nm=offset
            )
            neural = {
                k: v.cpu().numpy().copy() for k, v in control.tracker.neural_observation().items()
            }
            result = ImpactCandidate(
                name, offset.copy(), forecast, feedback_impact_score(forecast, offset), neural
            )
            candidates.append(result)
            return result

        self.baseline_windows += 1
        baseline = predict("baseline", np.zeros(29))
        if not baseline.score["feasible_predicted_impact"]:
            return finish("BASELINE_HAS_NO_COMPLETE_SAFE_PREDICTED_IMPACT")
        # Once per episode. Never repeatedly search a window to obtain a lucky result.
        self.screened = True
        bank = impact_torque_candidates(sample, side)
        for name, offset in bank[1:]:
            predict(name, offset)
        repeat = predict("repeat_zero", np.zeros(29))
        for candidate in candidates:
            if (
                candidate.forecast.donor_integration_state_hash
                != baseline.forecast.donor_integration_state_hash
            ):
                raise ValueError("all prospective candidates must use one unchanged actual donor")
        for left, right in (
            (baseline.forecast.arrays, repeat.forecast.arrays),
            (baseline.forecast.next_target_record, repeat.forecast.next_target_record),
            (baseline.next_neural_capture, repeat.next_neural_capture),
        ):
            if set(left) != set(right) or any(
                left[k].shape != right[k].shape
                or left[k].dtype != right[k].dtype
                or left[k].tobytes() != right[k].tobytes()
                for k in left
            ):
                raise ValueError("prospective zero repeat changed full feedback or neural bits")
        if (
            baseline.forecast.end_integration_state.tobytes()
            != repeat.forecast.end_integration_state.tobytes()
        ):
            raise ValueError("prospective zero repeat changed final integration state")
        eligible = [
            (index, c)
            for index, c in enumerate(candidates[:-1])
            if c.score["feasible_predicted_impact"]
        ]
        selected, best = min(eligible, key=lambda row: row[1].score["proxy_cost"])
        if baseline.score["proxy_cost"] - best.score["proxy_cost"] <= 1e-6:
            selected = 0
        return finish(
            "PROSPECTIVE_LOCAL_PROXY_SCREENED_NOT_TEACHER_QUALIFIED", selected_index=selected
        )
