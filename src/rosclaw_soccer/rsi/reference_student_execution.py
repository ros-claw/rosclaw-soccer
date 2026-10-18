"""Owned numeric reference model and causal SIM replay, never policy admission.

The parent motor consumes the proposed pre-parent target on the current body.
Raw numeric parameters are not proof of training. No fitting, Runtime or I/O.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import rosclaw.growth.causal_residual_memory as memory_module

from rosclaw_soccer.rsi import pre_parent_reference_features as features
from rosclaw_soccer.rsi import reference_student_decoder as decoder_module
from rosclaw_soccer.rsi.contact_motor_primitive import JOINT_NAMES
from rosclaw_soccer.rsi.contact_reference_labels import reference_labels
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

FIELDS = {
    "input_features": 135,
    "normalized_input_features": 135,
    "requested_reference_increment_rad": 12,
    "applied_reference_increment_rad": 12,
    "pre_parent_joint_target_rad": 29,
    "recurrent_hidden_state": 64,
}
PREFIX = "reference_student_"


def require_native_request(
    value,
    *,
    args,
    partition,
    control_profile,
    approach_mode,
    swing_mode,
    factory,
    world_hash,
    world_review,
    mixed_probe,
):
    from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
    from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
        RecurrentSamplingEpisodeFactory,
    )

    if (
        partition != "DECLARED_DEVELOPMENT_DIAGNOSTIC"
        or type(control_profile) is not BoundedFirstTouchControl
        or approach_mode != "bounded_approach_diagnostic"
        or swing_mode != "bounded_contact_diagnostic"
        or getattr(args, "record_foundation_observation", False) is not True
        or getattr(args, "foundation_only", False) is not False
        or getattr(args, "proposal_decoder", None) != "reference"
        or getattr(args, "step_model", None) is None
        or getattr(args, "body_response_bundle", None) is not None
        or getattr(args, "neural_model", None) is not None
        or getattr(args, "motor_policy", None) is not None
        or getattr(args, "cached_proposal_envelope", False) is not False
        or getattr(args, "observation_snapshot", None) != "cached"
        or getattr(args, "root_velocity_reference", None) != "body-com"
        or type(factory) is not RecurrentSamplingEpisodeFactory
        or world_hash is None
        or world_review is None
        or mixed_probe
    ):
        raise ValueError(
            "captured unmixed reviewed-world pre-parent reference SIM request required"
        )
    model = own_model(value)
    if model["world_hash"] != world_hash:
        raise ValueError("reference student must bind the declared world before allocation")
    factory._stable()


def _sources():
    return {
        name: hash_bytes(Path(module.__file__).read_bytes())
        for name, module in (
            ("decoder", decoder_module),
            ("features", features),
            ("memory", memory_module),
        )
    }


def make_model(parameters, *, mean, scale, world_hash, parent_model_hash):
    """Seal validated numeric weights, not a training/physics certificate."""
    decoder = decoder_module.ReferenceStudentDecoder(
        parameters,
        mean=np.asarray(mean, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        world_hash=world_hash,
        parent_model_hash=parent_model_hash,
    )
    value = dict(
        schema="soccer.rsi.pre_parent_reference_numeric_model.v1",
        parameters=copy.deepcopy(parameters),
        mean=np.asarray(mean).tolist(),
        scale=np.asarray(scale).tolist(),
        world_hash=world_hash,
        parent_model_hash=parent_model_hash,
        feature_names=list(features.FEATURE_NAMES),
        joint_names=list(JOINT_NAMES),
        action_kind="pre-parent leg reference increment, not torque",
        observation_snapshot="cached",
        root_velocity_reference="body-com",
        decoder_contract=decoder.contract(),
        source_hashes=_sources(),
        execution_source_hash=hash_bytes(Path(__file__).read_bytes()),
        activation_ceiling="SIM_ONLY",
        training_provenance_authenticated=False,
        physics_qualified=False,
        teacher_qualified=False,
        runtime_execution_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
        fresh_holdout_open_authorized=False,
    )
    return dict(value, model_hash=hash_json(value))


def own_model(value):
    """Reject altered schemas/parameters before allocating a physics body."""
    if type(value) is not dict:
        raise ValueError("explicit sealed numeric reference model required")
    owned = copy.deepcopy(value)
    expected = make_model(
        owned["parameters"],
        mean=owned["mean"],
        scale=owned["scale"],
        world_hash=owned["world_hash"],
        parent_model_hash=owned["parent_model_hash"],
    )
    if owned != expected:
        raise ValueError("complete unchanged numeric reference model and source contract required")
    return owned


def decoder_from_model(value, *, world_hash, parent_model_hash):
    owned = own_model(value)
    if owned["world_hash"] != world_hash or owned["parent_model_hash"] != parent_model_hash:
        raise ValueError("exact current physical world and unchanged parent behavior required")
    return decoder_module.ReferenceStudentDecoder(
        owned["parameters"],
        mean=np.asarray(owned["mean"], dtype=np.float64),
        scale=np.asarray(owned["scale"], dtype=np.float64),
        world_hash=world_hash,
        parent_model_hash=parent_model_hash,
    )


def require_capture(report, commitment, trace):
    present = {k for k in trace if k.startswith(PREFIX)}
    if (
        report.get("reference_student_model") != commitment.get("reference_student_model")
        or report.get("reference_student_contract") != commitment.get("reference_student_contract")
        or "reference_student_model" not in commitment
        or present != {PREFIX + k for k in FIELDS}
        or "bounded_control_profile" not in commitment
        or "foundation_observation_capture" not in commitment
        or any(
            k in commitment
            for k in (
                "mesh_surface_swing",
                "contact_exit_torque_mpc",
                "directional_contact_compliance",
                "swing_release_ablation",
                "cerebellar_torque_contract",
                "body_response_guidance",
            )
        )
    ):
        raise ValueError("complete unmixed captured pre-parent reference student required")
    own_model(commitment["reference_student_model"])
    for name, width in FIELDS.items():
        a = trace[PREFIX + name]
        if (
            type(a) is not np.ndarray
            or a.shape != (300, 1, width)
            or a.dtype != np.float64
            or not np.isfinite(a).all()
        ):
            raise ValueError("whole finite float64 reference student clock required")


def reconstruct(commitment, trace, *, joint_limits, parent_model_hash):
    """Called after independent dynamics/state verification; no new physics here.

    Reuses the pinned decoder implementation, not a second neural algorithm.
    Navigation targets reconstruct from actual raw Foundation actions; previous
    reference and hidden state come from this new decoder's own causal history.
    """
    require_capture(commitment, commitment, trace)
    decoder = decoder_from_model(
        commitment["reference_student_model"],
        world_hash=commitment["compiled_model_hash"],
        parent_model_hash=parent_model_hash,
    )
    navigation = reference_labels(trace)["navigation_joint_target_rad"]
    for frame in range(300):
        record = decoder.advance(
            trace,
            frame=frame,
            navigation_target=navigation[frame, 0],
            joint_limits=joint_limits,
        )
        if any(record[k].tobytes() != trace[PREFIX + k][frame, 0].tobytes() for k in FIELDS):
            raise ValueError(
                "causal reconstructed student inputs, hidden state or own action differ"
            )
        if (
            record["pre_parent_joint_target_rad"].tobytes()
            != trace["pre_motor_joint_target_rad"][frame, 0].tobytes()
        ):
            raise ValueError("student proposal must replace teacher before original parent motor")
    if decoder.contract() != commitment["reference_student_contract"]:
        raise ValueError("actual reference decoder identity differs")
    return dict(
        schema="soccer.rsi.pre_parent_reference_student_reconstruction.v1",
        model_hash=commitment["reference_student_model"]["model_hash"],
        frames_reconstructed=300,
        own_hidden_and_applied_reference_reconstructed=True,
        raw_foundation_navigation_targets_bit_exact=True,
        reconstruction_uses_same_pinned_decoder=True,
        independent_neural_implementation=False,
        parent_target_composition_reconstructed_separately=True,
        training_provenance_authenticated=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
