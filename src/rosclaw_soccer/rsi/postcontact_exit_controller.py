"""One current-body post-touch80ms bank, not a qualified teacher or real control.

Own measured past contact gates the search, never a future contact frame.
Predicts three subsequent Foundation/IK/RNN decisions per candidate using
the existing verified branch machinery. No planner refresh, retry, external
candidate name, source trajectory or policy-weight update is admitted.
"""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_early_feedback_forecast import (
    G1EarlyFeedbackForecast,
    G1EarlyFeedbackForecaster,
)
from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
from rosclaw_soccer.rsi.early_impact_controller import same_arrays
from rosclaw_soccer.rsi.feedback_branch_control import G1FeedbackBranchControl
from rosclaw_soccer.rsi.impact_torque_screen import CAPS, impact_torque_candidates
from rosclaw_soccer.rsi.postcontact_exit_score import postcontact_exit_score

MAX_FORECAST_STEPS = 560  # 13 named candidates plus one independent repeated zero.


@dataclass(frozen=True)
class PostContactExitCandidate:
    name: str
    offset_nm: np.ndarray
    forecast: G1EarlyFeedbackForecast
    score: dict[str, Any]


@dataclass(frozen=True)
class PostContactExitCommand:
    offset_nm: np.ndarray
    report: dict[str, Any]
    candidates: tuple[PostContactExitCandidate, ...]


def select_exit_candidate(candidates: list[PostContactExitCandidate]) -> int:
    if (
        len(candidates) < 2
        or len(candidates) > 14
        or candidates[0].name != "baseline"
        or candidates[-1].name != "repeat_zero"
        or np.any(candidates[0].offset_nm)
        or np.any(candidates[-1].offset_nm)
    ):
        raise ValueError("bounded whole baseline-first repeat-zero-last exit bank required")
    baseline, repeat = candidates[0].forecast, candidates[-1].forecast
    for c in candidates:
        if (
            type(c) is not PostContactExitCandidate
            or type(c.forecast) is not G1EarlyFeedbackForecast
            or type(c.offset_nm) is not np.ndarray
            or c.offset_nm.shape != (29,)
            or c.offset_nm.dtype != np.float64
            or not np.isfinite(c.offset_nm).all()
            or np.any(abs(c.offset_nm) > CAPS)
            or type(c.score.get("feasible_predicted_exit")) is not bool
            or c.score.get("schema") != "soccer.rsi.postcontact80ms_exit_proxy.v1"
        ):
            raise ValueError("typed bounded exit candidate and exact proxy profile required")
        if c.forecast.donor_integration_state_hash != baseline.donor_integration_state_hash:
            raise ValueError("same current donor for every exit candidate required")
        if c.score["feasible_predicted_exit"] and (
            type(c.score["proxy_cost"]) is not float
            or not np.isfinite(c.score["proxy_cost"])
            or c.score["proxy_cost"] < 0
        ):
            raise ValueError("finite nonnegative eligible exit proxy required")
    if (
        baseline.end_integration_state.tobytes() != repeat.end_integration_state.tobytes()
        or not same_arrays(baseline.arrays, repeat.arrays)
        or candidates[0].score != candidates[-1].score
        or any(
            len(x) != 3
            for x in (
                baseline.next_target_records,
                repeat.next_target_records,
                baseline.next_neural_records,
                repeat.next_neural_records,
            )
        )
        or any(
            not same_arrays(a, b)
            for aa, bb in (
                (baseline.next_target_records, repeat.next_target_records),
                (baseline.next_neural_records, repeat.next_neural_records),
            )
            for a, b in zip(aa, bb, strict=True)
        )
    ):
        raise ValueError("independent repeated-zero exit body/targets/NN/score differ")
    feasible = [(i, c) for i, c in enumerate(candidates[:-1]) if c.score["feasible_predicted_exit"]]
    if not feasible:
        return 0  # Explicit zero fallback, not a claim of predicted safety.
    index, best = min(feasible, key=lambda pair: pair[1].score["proxy_cost"])
    if (
        candidates[0].score["feasible_predicted_exit"]
        and candidates[0].score["proxy_cost"] - best.score["proxy_cost"] <= 1e-6
    ):
        return 0
    return index


class PostContactExitController:
    def __init__(
        self,
        model: Any,
        *,
        world_hash: str,
        kp: np.ndarray,
        kd: np.ndarray,
        reuse_owned_updates: bool = False,
    ) -> None:
        if type(reuse_owned_updates) is not bool:
            raise ValueError("explicit branch continuation option required")
        self.model = model
        self.forecaster = G1EarlyFeedbackForecaster(
            model,
            world_hash=world_hash,
            kp=kp,
            kd=kd,
            max_forecast_steps=MAX_FORECAST_STEPS,
            reuse_owned_updates=reuse_owned_updates,
        )
        self.sampler = G1JointTaskKinematicSampler(model, expected_world_hash=world_hash)
        self.next_frame = 0
        self.trigger_frame: int | None = None
        self.side: int | None = None
        self.selected_name: str | None = None
        self.screened = self.cancelled = self.failed = False
        self.held = np.zeros(29)
        self.end_frame = 0
        self.last_candidates: list[PostContactExitCandidate] = []

    def plan(
        self,
        data: Any,
        current_target_rad: np.ndarray,
        *,
        frame: int,
        previous_contact_forces: np.ndarray,
        tracker: Any,
        decoder: Any,
        ik: Any,
        policy: dict[str, Any],
        body_prefix: dict[str, np.ndarray],
        previous_motor_delta_rad: np.ndarray,
    ) -> PostContactExitCommand:
        if self.failed:
            raise ValueError("failed exit controller cannot resume/retry")
        self.failed = True
        if (
            type(frame) is not int
            or frame != self.next_frame
            or not 0 <= frame < 300
            or type(data) is not mujoco.MjData
            or data.model is not self.model
            or data is self.forecaster._scratch
            or not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or not np.isclose(data.time, frame * 0.02, rtol=0, atol=1e-8)
            or type(current_target_rad) is not np.ndarray
            or current_target_rad.shape != (29,)
            or current_target_rad.dtype != np.float64
            or not np.isfinite(current_target_rad).all()
            or np.max(abs(current_target_rad)) > 10
            or type(previous_contact_forces) is not np.ndarray
            or previous_contact_forces.shape != (6,)
            or previous_contact_forces.dtype != np.float64
            or not np.isfinite(previous_contact_forces).all()
            or np.any(previous_contact_forces < 0)
            or (frame == 0 and np.any(previous_contact_forces))
        ):
            raise ValueError("next own current state/target and completed contact forces required")
        self.last_candidates = []
        before = self.forecaster.attempted_steps

        def finish(status: str, selected: int | None = None) -> PostContactExitCommand:
            offset = self.held.copy() if frame < self.end_frame else np.zeros(29)
            report = dict(
                schema="soccer.rsi.postcontact80ms_exit_plan.v1",
                frame=frame,
                status=status,
                trigger_frame=self.trigger_frame,
                measured_contact_side=self.side,
                selected_index=selected,
                selected_name=self.selected_name,
                cancelled=self.cancelled,
                requested_offset_nm=offset.tolist(),
                action_frames_remaining_including_current=max(0, self.end_frame - frame),
                forecast_substeps=self.forecaster.attempted_steps - before,
                accumulated_forecast_substeps=self.forecaster.attempted_steps,
                future_source_frame_or_target_used=False,
                full_h300_gain_proven=False,
                neural_policy_trained=False,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            self.next_frame += 1
            self.failed = False
            return PostContactExitCommand(offset, report, tuple(self.last_candidates))

        if data.qpos[2] < 0.65 or np.any(previous_contact_forces[2:] > 1):
            self.cancelled = self.screened = True
            self.held.fill(0)
            self.end_frame = frame
            return finish("CANCELLED_OBSERVED_UNSAFE_NO_RETRY")
        if frame < self.end_frame:
            return finish("HELD_BOUNDED_POSTCONTACT60MS_ACTION")
        if self.screened:
            return finish("ALREADY_SCREENED_NO_SECOND_BANK")
        if frame < 30 or not np.any(previous_contact_forces[:2] > 1):
            return finish("WAITING_FOR_COMPLETED_FOOT_CONTACT")
        self.trigger_frame, self.side = frame, int(np.argmax(previous_contact_forces[:2]))
        self.screened = True
        if frame > 294 or frame // 20 != (frame + 3) // 20:
            return finish("CONTACT_OUTSIDE_NO_REPLAN80MS_WINDOW_ZERO_ONLY")
        sample = self.sampler.sample(data)
        bank = impact_torque_candidates(sample, self.side)
        for name, offset in (*bank, ("repeat_zero", np.zeros(29))):
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
            result = PostContactExitCandidate(
                name,
                offset.copy(),
                forecast,
                postcontact_exit_score(
                    forecast, offset, previous_contact_forces=previous_contact_forces
                ),
            )
            self.last_candidates.append(result)
        selected = select_exit_candidate(self.last_candidates)
        best = self.last_candidates[selected]
        self.selected_name = best.name
        if selected:
            self.held = best.offset_nm.copy()
            self.end_frame = frame + 3
        return finish("POSTCONTACT_EXIT_PROXY_SCREENED_NOT_TEACHER_QUALIFIED", selected)

    def review(self) -> dict[str, Any]:
        if self.failed or self.next_frame != 300:
            raise ValueError("whole once-only exit episode required")
        return dict(
            trigger_frame=self.trigger_frame,
            measured_contact_side=self.side,
            selected_name=self.selected_name,
            cancelled=self.cancelled,
            forecast_integrated_steps=self.forecaster.integrated_steps,
            maximum_forecast_steps=MAX_FORECAST_STEPS,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
