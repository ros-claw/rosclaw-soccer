"""Finite SIM interface-authority probe, not training or a good-football claim."""

import copy

import numpy as np
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.reference_student_execution import PREFIX, make_model
from rosclaw_soccer.sim.contracts import hash_bytes

SEED = 2026101901
JOBS = (
    dict(job="reference-off-c6", model_index=None, group=76, option_index=2),
    dict(job="reference-zero-c6", model_index=0, group=76, option_index=2),
    dict(job="reference-left-ankle-c6", model_index=1, group=76, option_index=2),
)


def probe_models(world_hash, parent_model_hash):
    parameters = initial_parameters(135, 12, hidden_dimension=64, seed=SEED)
    nonzero = copy.deepcopy(parameters)
    nonzero["head_bias"][4] = 0.2
    return [
        make_model(
            p,
            mean=np.zeros(135),
            scale=np.ones(135),
            world_hash=world_hash,
            parent_model_hash=parent_model_hash,
        )
        for p in (parameters, nonzero)
    ]


def original_arrays(trace):
    arrays = {k: v for k, v in trace.items() if not k.startswith(PREFIX)}
    if len(arrays) != 33 or any(
        not isinstance(v, np.ndarray) or len(v) != 300 for v in arrays.values()
    ):
        raise ValueError("all original thirty-three complete arrays required")
    return arrays


def require_parity(trace, baseline, frames):
    if type(frames) is not int or frames not in (30, 300):
        raise ValueError("explicit full episode or pre-action prefix required")
    actual, original = original_arrays(trace), original_arrays(baseline)
    if actual.keys() != original.keys() or any(
        a.dtype != original[k].dtype
        or a.shape != original[k].shape
        or a[:frames].tobytes() != original[k][:frames].tobytes()
        for k, a in actual.items()
    ):
        raise ValueError("complete original physical arrays must remain byte-exact")
    return {k: hash_bytes(a[:frames].tobytes()) for k, a in sorted(actual.items())}


def measured_authority(zero, nonzero):
    prefix = require_parity(nonzero, zero, 30)
    applied = nonzero[PREFIX + "applied_reference_increment_rad"]
    zero_applied = zero[PREFIX + "applied_reference_increment_rad"]
    if np.any(zero_applied != 0) or np.any(applied[:30] != 0):
        raise ValueError("numeric zero and original pre-action interval must remain zero")
    values = dict(
        maximum_applied_reference_difference_rad=float(np.max(np.abs(applied - zero_applied))),
        maximum_pre_parent_target_difference_rad=float(
            np.max(
                np.abs(nonzero["pre_motor_joint_target_rad"] - zero["pre_motor_joint_target_rad"])
            )
        ),
        maximum_final_parent_target_difference_rad=float(
            np.max(np.abs(nonzero["joint_target_rad"] - zero["joint_target_rad"]))
        ),
        maximum_joint_position_difference_rad=float(
            np.max(np.abs(nonzero["joint_position_rad"] - zero["joint_position_rad"]))
        ),
    )
    if any(not np.isfinite(v) for v in values.values()):
        raise ValueError("finite measured body response required")
    return dict(
        **values,
        all_original_pre_action_prefix_hashes=prefix,
        reference_control_authority_observed=all(v > 1e-8 for v in values.values()),
        probe_is_trained_policy=False,
        success_improvement_claimed=False,
        dataset_admission_authorized=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
