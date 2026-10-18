"""Typed captured surface-point input/target evidence, not physical qualification.

The CPU dynamics replay must additionally reconstruct side, current material
point, Jacobian, IK and release from its own causal states and contact history.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


def require_surface_swing_capture(
    report: dict[str, Any], commitment: dict[str, Any], trace: dict[str, Any]
) -> None:
    contract = commitment["mesh_surface_swing"]
    if (
        report.get("mesh_surface_swing") != contract
        or contract.get("schema") != "soccer.rsi.mesh_surface_swing.v1"
        or contract.get("contract_hash")
        != hash_json({k: v for k, v in contract.items() if k != "contract_hash"})
        or contract.get("world_hash") != commitment.get("compiled_model_hash")
        or contract.get("strike_direction_world") != [1.0, 0.0, 0.0]
        or contract.get("maximum_joint_increment_rad") != 0.35
        or contract.get("release_frames") != 20
        or contract.get("parent_motor_recomputed_after_new_pre_motor_target") is not True
        or contract.get("activation_ceiling") != "SIM_ONLY"
        or any(
            contract.get(k) is not False
            for k in (
                "trained_policy",
                "teacher_qualified",
                "independent_ik_implementation",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(
            k in commitment
            for k in (
                "swing_release_ablation",
                "contact_exit_torque_mpc",
                "directional_contact_compliance",
                "cerebellar_torque_contract",
                "foundation_approach_ablation",
                "generated_seed_episode_binding",
            )
        )
        or "bounded_control_profile" not in commitment
        or "foundation_observation_capture" not in commitment
        or "default_locomotion_reference_capture" not in commitment
        or commitment.get("swing_mode") != "bounded_contact_diagnostic"
        or commitment.get("execution_profile") != "taskspace_plus_motor"
        or {k for k in trace if k.startswith("mesh_surface_")} != {"mesh_surface_swing_record"}
    ):
        raise ValueError("unmixed source-bound captured surface swing required")
    record = trace["mesh_surface_swing_record"]
    if (
        not isinstance(record, np.ndarray)
        or record.shape != (300, 1, 39)
        or record.dtype != np.float64
        or not np.isfinite(record).all()
        or np.max(abs(record)) > 1e6
        or not np.array_equal(record[:, 0, 0], np.arange(300))
        or not np.isin(record[:, 0, 2], [0.0, 1.0]).all()
    ):
        raise ValueError("complete finite typed current surface record and clock required")
    active = record[:, 0, 2] == 1
    if (
        np.any(active[:30])
        or not np.isin(record[active, 0, 1], [0.0, 1.0]).all()
        or np.any(record[~active, 0, 1] != -1)
        or np.any(record[~active, 0, 3:] != 0)
        or np.any(abs(record[active, 0, 33:39]) > 0.35 + 1e-12)
    ):
        raise ValueError("no fabricated inactive geometry or out-of-domain surface increments")
    forces = np.asarray(trace.get("force_n"))
    if forces.shape != (300, 1, 6) or not np.isfinite(forces).all():
        raise ValueError("complete causal six-body monitored force history required")
    first_monitored = np.flatnonzero(np.any(forces[:, 0] > 1, axis=1))
    if len(first_monitored) and np.any(active[first_monitored[0] + 1 :]):
        raise ValueError("surface solve must end after original first-any-body contact latch")
