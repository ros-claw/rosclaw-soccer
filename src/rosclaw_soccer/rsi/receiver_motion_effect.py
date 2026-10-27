"""Compare actually executed receiver histories, never counterfactual gradients."""

import numpy as np


def _trace(data: dict[str, np.ndarray]) -> None:
    required = {
        "control": (None,),
        "context14": (None, 14),
        "action": (None, 29),
        "realized_delta_radians": (None, 29),
        "contact_flags_before": (None, 2),
    }
    if not required.keys() <= data.keys():
        raise ValueError("complete actual receiver control trace required")
    n = len(data["control"])
    if not 1 <= n <= 300:
        raise ValueError("bounded nonempty receiver history required")
    for name, shape in required.items():
        a = data[name]
        if not isinstance(a, np.ndarray) or a.shape != (n, *shape[1:]):
            raise ValueError("aligned actual receiver history required")
        if not np.isfinite(a).all():
            raise ValueError("finite receiver history required")
    if (
        not np.array_equal(data["control"], np.arange(n))
        or data["contact_flags_before"].dtype != np.bool_
        or not np.array_equal(data["context14"][:, 12:14], data["contact_flags_before"])
        or np.any(np.diff(data["contact_flags_before"].astype(np.int8), axis=0) < 0)
    ):
        raise ValueError("sequential own controls and causal persistent contact flags required")


def _measure(data: dict[str, np.ndarray]) -> dict:
    context = data["context14"].astype(np.float64)
    # Distances and translational relative velocities are measured in the
    # rotating body frame. Their 3D norms are rotation invariant, but they are
    # root-to-ball proxies, not foot-to-ball distances or contact impulses.
    distance = np.linalg.norm(context[:, :3], axis=1)
    speed = np.linalg.norm(context[:, 3:6], axis=1)
    index = np.flatnonzero(data["contact_flags_before"].any(axis=1))
    first = int(index[0]) if len(index) else None
    closest = int(np.argmin(distance))
    return dict(
        controls=len(context),
        min_root_ball_distance_3d_m=float(distance[closest]),
        relative_speed_at_closest_3d_mps=float(speed[closest]),
        closest_control=closest,
        first_preaction_past_contact_control=first,
        root_ball_distance_at_first_flag_m=float(distance[first]) if first is not None else None,
        relative_speed_at_first_flag_mps=float(speed[first]) if first is not None else None,
        max_realized_joint_delta_rad=float(np.abs(data["realized_delta_radians"]).max()),
    )


def paired_receiver_motion_effect(
    baseline: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict:
    """Time-align two own histories; differing states remain coupled outcomes."""
    _trace(baseline)
    _trace(candidate)
    common = min(len(baseline["control"]), len(candidate["control"]))
    return dict(
        baseline=_measure(baseline),
        candidate=_measure(candidate),
        common_control_prefix=common,
        max_executed_action_difference=float(
            np.abs(baseline["action"][:common] - candidate["action"][:common]).max()
        ),
        max_observed_relative_position_difference_m=float(
            np.linalg.norm(
                baseline["context14"][:common, :3].astype(np.float64)
                - candidate["context14"][:common, :3].astype(np.float64),
                axis=1,
            ).max()
        ),
        causal_attribution_proven=False,
        foot_contact_quality_proven=False,
    )
