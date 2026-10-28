"""Verify when a SIM passer-follow counterfactual actually begins."""

import numpy as np


def verify_follow_prefix(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    initial_control: int,
    receiver_velocity_controls: int,
) -> dict[str, object]:
    """Require exact physical prefix and causal next-boundary foot trigger.

    This checks archived arrays, not independent physics replay or possession.
    """
    if type(initial_control) is not int or initial_control < 0:
        raise ValueError("valid initial control required")
    if type(receiver_velocity_controls) is not int or receiver_velocity_controls < 1:
        raise ValueError("valid pre-return boundary required")
    for data in (baseline, candidate):
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            if key not in data or data[key].ndim not in (3, 4) or not np.isfinite(data[key]).all():
                raise ValueError("finite native physics arrays required")
        if data["ball_forces"].shape[1:] != (10, 2, 3):
            raise ValueError("native foot-contact witness shape required")
        if any(len(data[k]) != len(data["qpos"]) for k in ("qvel", "torque", "ball_forces")):
            raise ValueError("physical array length mismatch")
    controls = candidate.get("passer_follow_control")
    if controls is None or controls.ndim != 1 or controls.dtype.kind not in "iu":
        raise ValueError("explicit consecutive actual follow controls required")
    if controls.size and not np.array_equal(
        controls, np.arange(controls[0], controls[0] + len(controls))
    ):
        raise ValueError("follow controls must be consecutive")
    for key, width in (
        ("passer_follow_action", 29),
        ("passer_follow_command", 3),
        ("passer_follow_observation", 98),
        ("passer_follow_target", 29),
    ):
        value = candidate.get(key)
        if value is None or value.shape != (len(controls), width) or not np.isfinite(value).all():
            raise ValueError("complete actual follow evidence required")
    witness = np.any(
        candidate["ball_forces"][receiver_velocity_controls:, :, 1, :2] > 1.0, axis=(1, 2)
    )
    events = np.flatnonzero(witness)
    first_foot = None if not events.size else receiver_velocity_controls + int(events[0])
    expected = None if first_foot is None else first_foot + 1
    if expected is not None and expected < len(candidate["qpos"]):
        if not controls.size or int(controls[0]) != initial_control + expected:
            raise ValueError("follow did not begin at next actual foot-contact boundary")
        prefix = expected
    else:
        if controls.size:
            raise ValueError("follow has no causal prior foot contact")
        prefix = len(candidate["qpos"])
    if len(baseline["qpos"]) < prefix:
        raise ValueError("baseline ends before intervention")
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        if not np.array_equal(baseline[key][:prefix], candidate[key][:prefix]):
            raise ValueError(f"counterfactual pre-follow physics differs: {key}")
    if controls.size and len(controls) != len(candidate["qpos"]) - prefix:
        raise ValueError("missing follow controls after switch")
    return {
        "exact_prefix_controls": prefix,
        "first_returner_foot_control": first_foot,
        "follow_controls": len(controls),
        "sim_contact_oracle_used": True,
    }
