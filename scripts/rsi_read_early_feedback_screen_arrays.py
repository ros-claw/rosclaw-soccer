"""All80ms action/target/capture checks, sharing proxy formula but no physics."""

import numpy as np

from rosclaw_soccer.physics.g1_early_feedback_forecast import G1EarlyFeedbackForecast
from rosclaw_soccer.rsi.early_feedback_impact_score import early_feedback_impact_score
from rosclaw_soccer.rsi.impact_torque_screen import CAPS
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json
from scripts.rsi_impact_torque_authority_screen import equal


def verify_early_prediction(a, metadata, side):
    offsets = a["requested_offset_nm"]
    if offsets.shape != (40, 29) or offsets.dtype != np.float64 or not np.isfinite(offsets).all():
        raise ValueError("whole finite80ms action schedule required")
    offset = offsets[0].copy()
    if (
        np.any(abs(offset) > CAPS)
        or offset.tobytes() != np.asarray(metadata["offset_nm"], dtype=np.float64).tobytes()
        or metadata["is_zero"] is not bool(not np.any(offset))
    ):
        raise ValueError("same capped actual60ms correction required")
    allowed = np.zeros(29, dtype=bool)
    if side in (0, 1):
        allowed[6 * side : 6 * side + 6] = True
    if np.any(offset[~allowed] != 0):
        raise ValueError("only original current causal swing leg may receive correction")
    for key, shape in (
        ("canonical_qpos_after_substep", (40, 43)),
        ("canonical_qvel_after_substep", (40, 41)),
        ("requested_pd_torque_nm", (40, 29)),
        ("applied_control_nm", (40, 29)),
        ("actual_actuator_force_nm", (40, 29)),
        ("completed_contact_peak_force_n", (40, 6)),
    ):
        v = a[key]
        if v.shape != shape or v.dtype != np.float64 or not np.isfinite(v).all():
            raise ValueError("whole finite80ms physical arrays required")
    end = a["end_integration_state"]
    if np.any(a["completed_contact_peak_force_n"] < 0):
        raise ValueError("nonnegative causal contact-force magnitudes required")
    if (
        end.ndim != 1
        or end.dtype != np.float64
        or not 41 <= len(end) <= 65536
        or not np.isfinite(end).all()
    ):
        raise ValueError("whole finite80ms integration end required")
    target_records, neural_records = [], []
    for j in range(1, 4):
        record, neural = {}, {}
        for name, size in (
            ("navigation_target_rad", 29),
            ("taskspace_increment_rad", 29),
            ("motor_delta_rad", 12),
            ("joint_target_rad", 29),
            ("canonical_qpos", 43),
            ("canonical_qvel", 41),
        ):
            v = a[f"next_{j}_" + name]
            if v.shape != (size,) or v.dtype != np.float64 or not np.isfinite(v).all():
                raise ValueError("all three finite next-target/body records required")
            record[name] = v
        for name, width in (
            ("encoder_features", 640),
            ("latent_token", 64),
            ("decoder_input", 994),
            ("raw_action_isaac", 29),
            ("joint_target_mujoco_rad", 29),
        ):
            v = a[f"neural_next_{j}_" + name]
            if v.shape != (1, width) or v.dtype != np.float32 or not np.isfinite(v).all():
                raise ValueError("all three actual finite neural captures required")
            neural[name] = v
        equal(
            record["canonical_qpos"],
            a["canonical_qpos_after_substep"][j * 10 - 1],
            f"actual intermediate{j} body q",
        )
        equal(
            record["canonical_qvel"],
            a["canonical_qvel_after_substep"][j * 10 - 1],
            f"actual intermediate{j} body v",
        )
        target = record["navigation_target_rad"] + record["taskspace_increment_rad"]
        target[:12] += record["motor_delta_rad"]
        equal(record["joint_target_rad"], target, f"whole intermediate{j} target decomposition")
        target_records.append(record)
        neural_records.append(neural)
    requested = a["requested_pd_torque_nm"].copy()
    requested[:30] += offset
    expected = np.clip(requested, -np.asarray(G1_HARD_TORQUE_LIMITS), G1_HARD_TORQUE_LIMITS)
    equal(a["applied_control_nm"], expected, "first60ms correction then unmodified20ms feedback")
    equal(a["actual_actuator_force_nm"], a["applied_control_nm"], "actual canonical motor force")
    forecast = G1EarlyFeedbackForecast(
        metadata["donor_hash"], end, a, tuple(target_records), tuple(neural_records)
    )
    score = early_feedback_impact_score(forecast, offset)
    if hash_json(score) != hash_json(metadata["score"]):
        raise ValueError("all40substeps raw impact proxy differs from metadata")
    return score
