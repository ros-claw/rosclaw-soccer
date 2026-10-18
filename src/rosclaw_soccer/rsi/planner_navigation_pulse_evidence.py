"""Zero-reference pre-planner pulse request and independent scalar capture audit.

No simulator allocation or execution. This validates a fixed research probe,
not a trained policy, physical gain, task success, or permission to deploy.
"""

from collections.abc import Callable
from dataclasses import asdict
from typing import Any, cast

import numpy as np
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.planner_navigation_pulse import PlannerNavigationPulse
from rosclaw_soccer.rsi.reference_student_execution import own_model as _own_model
from rosclaw_soccer.rsi.reference_student_probe import original_arrays as _original_arrays
from rosclaw_soccer.sim.contracts import hash_json

own_model = cast(Callable[[dict[str, Any]], dict[str, Any]], _own_model)
original_arrays = cast(Callable[[dict[str, Any]], dict[str, Any]], _original_arrays)
TAG = "pre_planner_navigation_pulse"
PREFIX = "planner_pulse_"
FIELDS = ("baseline_command", "requested_residual", "applied_command")


def require_pulse_physical_parity(
    trace: dict[str, Any], baseline: dict[str, Any], *, frames: int
) -> dict[str, Any]:
    """Compare fixed original arrays, without teaching legacy readers to ignore extras.

    Callers separately authenticate and validate pulse capture. A pulse OFF
    compares all300 frames; its active arm must preserve the first100 frames.
    """
    if type(frames) is not int or frames not in (100, 300):
        raise ValueError("exact OFF300 or before-pulse100 comparison required")

    def physical(value: dict[str, Any]) -> dict[str, Any]:
        found = {k for k in value if k.startswith(PREFIX)}
        if found and found != {PREFIX + k for k in FIELDS}:
            raise ValueError("no unknown or partial pulse arrays may be hidden")
        return original_arrays({k: v for k, v in value.items() if k not in found})

    left, right = physical(trace), physical(baseline)
    if left.keys() != right.keys() or any(
        left[k].shape != right[k].shape
        or left[k].dtype != right[k].dtype
        or left[k][:frames].tobytes() != right[k][:frames].tobytes()
        for k in left
    ):
        raise ValueError("all original33 physical arrays must match the declared pulse prefix")
    return dict(
        compared_frames=frames,
        physical_array_count=33,
        original_array_bits_equal=True,
        runtime_execution_authorized=False,
    )


def require_navigation_pulse_request(
    pulse: Any, model: Any, *, profile: Any, record_planner_events: Any
) -> None:
    if (
        type(pulse) is not PlannerNavigationPulse
        or type(profile) is not BoundedFirstTouchControl
        or asdict(profile) != asdict(BoundedFirstTouchControl())
        or record_planner_events is not True
        or type(model) is not dict
    ):
        raise ValueError(
            "fixed B2 pre-planner SIM probe with recorded planner and zero model required"
        )
    pulse.__post_init__()
    if own_model(model)["parameters"] != initial_parameters(
        135, 12, hidden_dimension=64, seed=2026101902
    ):
        raise ValueError(
            "exact original seed1902 zero reference head required; no learned/mixed model"
        )


def require_navigation_pulse_capture(
    report: dict[str, Any], commitment: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any]:
    present = {k for k in trace if k.startswith(PREFIX)}
    if TAG not in commitment or present != {PREFIX + k for k in FIELDS}:
        raise ValueError("explicit pulse commitment and all three owned capture channels required")
    contract = commitment[TAG]
    pulse = PlannerNavigationPulse(**contract["parameters"])
    if contract != pulse.contract() or report.get(TAG) != contract:
        raise ValueError("same typed source-bound pulse report and commitment required")
    profile = BoundedFirstTouchControl(**commitment["bounded_control_profile"]["parameters"])
    if commitment["bounded_control_profile"] != profile.contract():
        raise ValueError("original bounded profile source required")
    require_navigation_pulse_request(
        pulse,
        commitment.get("reference_student_model"),
        profile=profile,
        record_planner_events=True,
    )

    def array(key: str, tail: tuple[int, ...]) -> np.ndarray:
        a = np.asarray(trace[key])
        if (
            a.shape != (300, 1, *tail)
            or a.dtype != np.float64
            or not np.isfinite(a).all()
            or np.max(np.abs(a)) > 1e6
        ):
            raise ValueError("whole finite float64 H300 pulse audit channels required")
        return a[:, 0]

    baseline, residual, applied = (array(PREFIX + k, (3,)) for k in FIELDS)
    nav = array("navigation_command", (3,))
    raw = array("approach_requested_navigation_command", (3,))
    root, ball = array("root_pose_xyzw_m", (7,)), array("ball_position_before_step_m", (3,))
    velocity, force = array("ball_linear_velocity_before_step_m_s", (3,)), array("force_n", (6,))
    if np.any(force < 0):
        raise ValueError("original nonnegative contact forces required")
    contacts = np.flatnonzero(np.any(force > 1, axis=1))
    first = int(contacts[0]) if len(contacts) else None
    for key in ("requested_reference_increment_rad", "applied_reference_increment_rad"):
        if np.any(array("reference_student_" + key, (12,)) != 0):
            raise ValueError("entire zero-reference baseline required; cannot mix in contact IK")
    for frame in range(300):
        gap = ball[frame] - root[frame, :3]
        if first is None or frame <= first:
            fwd = float(velocity[frame, 0]) + 1.2 * (float(gap[0]) - 0.35)
            fade = min(1.0, max(0.0, (float(gap[0]) - 0.35) / (0.95 - 0.35)))
            lat = 1.2 * (float(gap[1]) - 0.0) * fade
        else:
            fwd, lat = 1.4, 0.0
        expected_raw = np.asarray([fwd, lat, 0.0])
        base = np.asarray([np.clip(fwd, 0.0, 1.4), np.clip(lat, -0.2, 0.2), 0.0])
        # Independently reconstruct, never call the producer's compose().
        delta = pulse.lateral_delta_m_s if 100 <= frame < 160 else 0.0
        expected = base.copy()
        if delta != 0.0:
            expected[1] = min(0.2, max(-0.2, float(base[1]) + delta))
        if (
            not np.array_equal(raw[frame], expected_raw)
            or baseline[frame].tobytes() != base.tobytes()
            or residual[frame].tobytes() != np.asarray([0.0, delta, 0.0]).tobytes()
            or applied[frame].tobytes() != expected.tobytes()
            or nav[frame].tobytes() != expected.tobytes()
        ):
            raise ValueError("independent B2 plus pulse command reconstruction failed")
    result = dict(
        schema="soccer.rsi.pre_planner_navigation_pulse_review.v1",
        contract_hash=contract["contract_hash"],
        frames_reconstructed=300,
        original_bounded_navigation_reconstructed=True,
        pulse_composition_reconstructed=True,
        entire_leg_reference_strictly_zero=True,
        forward_and_yaw_unchanged=True,
        source_trace_authentication_performed_here=False,
        planner_and_encoder_effect_verified_here=False,
        new_physical_steps=0,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
