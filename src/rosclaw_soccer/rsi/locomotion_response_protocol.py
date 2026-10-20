"""Predeclared open-loop SIM locomotion identification, not a football exam."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class LocomotionResponseCourse:
    index: int
    split: str
    command_xy_mps: tuple[float, float]
    onset_frame: int
    pulse_frames: int = 100
    ball_course: tuple[float, float, float] = (3.0, 3.0, 0.0)

    def command(self, frame: int) -> tuple[float, float, float]:
        if type(frame) is not int or not 0 <= frame < 300:
            raise ValueError("sequential H300 control frame required")
        return (
            (*self.command_xy_mps, 0.0)
            if self.onset_frame <= frame < self.onset_frame + self.pulse_frames
            else (0.0, 0.0, 0.0)
        )


def response_courses() -> tuple[LocomotionResponseCourse, ...]:
    rows = [LocomotionResponseCourse(0, "train", (0.0, 0.0), 50)]
    for speed in (0.1, 0.3):
        for axis in (0, 1):
            for sign in (-1, 1):
                xy = [0.0, 0.0]
                xy[axis] = float(sign * speed)
                rows.append(LocomotionResponseCourse(len(rows), "train", (xy[0], xy[1]), 50))
    for axis in (0, 1):
        for sign in (-1, 1):
            xy = [0.0, 0.0]
            xy[axis] = float(sign * 0.2)
            rows.append(LocomotionResponseCourse(len(rows), "validation", (xy[0], xy[1]), 75))
    for diagonal in ((0.14, 0.14), (-0.14, 0.14)):
        rows.append(LocomotionResponseCourse(len(rows), "validation", diagonal, 75))
    return tuple(rows)


def preflight_response_courses(courses: tuple[LocomotionResponseCourse, ...]) -> str:
    expected = response_courses()
    if type(courses) is not tuple or len(courses) != 15:
        raise ValueError("entire fifteen-command intervention bank required")
    for row, gold in zip(courses, expected, strict=True):
        if (
            type(row) is not LocomotionResponseCourse
            or row != gold
            or type(row.index) is not int
            or type(row.onset_frame) is not int
            or type(row.pulse_frames) is not int
            or type(row.command_xy_mps) is not tuple
            or type(row.ball_course) is not tuple
            or any(
                type(x) is not float or not np.isfinite(x)
                for x in (*row.command_xy_mps, *row.ball_course)
            )
            or np.linalg.norm(row.command_xy_mps) > 0.3000000001
        ):
            raise ValueError(
                "exact finite immutable locomotion interventions required; no replacements"
            )
    return hash_json(
        dict(
            schema="soccer.rsi.locomotion_response_bank.v1",
            courses=[asdict(c) for c in courses],
            independent_initial_worlds=1,
            football_exam=False,
            source_motion_or_gate_changed=False,
            hardware_authorized=False,
        )
    )


def summarize_response(
    raw: dict[str, np.ndarray], course: LocomotionResponseCourse
) -> dict[str, Any]:
    from rosclaw_soccer.rsi.recorded_entry_review import recorded_target, validate_recording

    # Borrow shape/dtype validation only, with explicitly synthetic unused entry
    # metadata. These four fields are not stored as observations or exam proof.
    extra = dict(
        entry_stage=np.zeros(300, dtype=np.int64),
        entry_command=raw["navigation_command"],
        entry_target_root_xy_m=np.zeros((300, 2)),
        entry_error_m=np.zeros(300),
    )
    physical = {k: v for k, v in raw.items() if k != "navigation_command"}
    contacts = physical["ball_contact_force_by_body_n"]
    validate_recording({**physical, **extra}, frames=300, nbody=contacts.shape[2])
    if not np.array_equal(
        raw["navigation_command"], np.array([course.command(f) for f in range(300)])
    ):
        raise ValueError("actual navigation commands differ from declared exogenous pulse")
    q, v = raw["canonical_qpos"], raw["canonical_qvel"]
    for key, after in (
        ("canonical_qpos", "target_body_canonical_qpos_after_substep"),
        ("canonical_qvel", "target_body_canonical_qvel_after_substep"),
    ):
        if not np.array_equal(raw[key][1:], raw[after][:-1, -1]):
            raise ValueError("continuous actual native body required")
    for frame in range(300):
        if not np.array_equal(recorded_target(raw, frame), raw["joint_target_rad"][frame]):
            raise ValueError("frozen neural action to PD target mapping differs")
    start, end = course.onset_frame, course.onset_frame + course.pulse_frames
    upright = 1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)
    return dict(
        course=asdict(course),
        pulse_displacement_xy_m=(q[end, :2] - q[start, :2]).tolist(),
        pulse_last_second_mean_velocity_mps=v[end - 50 : end, :2].mean(axis=0).tolist(),
        terminal_second_mean_velocity_mps=v[250:300, :2].mean(axis=0).tolist(),
        after_pulse_root_drift_m=float(np.linalg.norm(q[-1, :2] - q[end, :2])),
        minimum_pelvis_z_m=float(raw["pelvis_z_per_substep_m"].min()),
        minimum_upright=float(upright.min()),
        maximum_robot_ball_contact_n=float(contacts[:, :, 1:].max()),
        no_robot_ball_contact=bool(contacts[:, :, 1:].max() <= 1),
        stable_height_and_upright=bool(
            raw["pelvis_z_per_substep_m"].min() >= 0.65 and upright.min() >= 0.9
        ),
        football_skill_proven=False,
        learned_policy_improvement_proven=False,
    )
