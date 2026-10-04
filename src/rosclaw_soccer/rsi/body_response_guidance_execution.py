"""Explicit CPU SIM adapter, independently reconstructible from body evidence."""

import copy
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.body_response_guidance import BodyResponseRecoveryProposal
from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD, SLEW_RAD
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def make_bundle(
    parent_model_hash: str,
    model_pairs: Any,
    *,
    active_contact_phases: tuple[int, ...] = (2,),
    action_dimensions: int = 12,
    maximum_accumulated_increment_rad: float = 0.02,
) -> dict[str, Any]:
    """Bind complete prediction models and source; grants no policy qualification."""
    if (
        type(parent_model_hash) is not str
        or len(parent_model_hash) != 71
        or not parent_model_hash.startswith("sha256:")
        or any(c not in "0123456789abcdef" for c in parent_model_hash[7:])
    ):
        raise ValueError("complete explicit parent model identity required")
    pairs = copy.deepcopy(model_pairs)
    proposal = BodyResponseRecoveryProposal(
        pairs,
        active_contact_phases=active_contact_phases,
        action_dimensions=action_dimensions,
        maximum_accumulated_increment_rad=maximum_accumulated_increment_rad,
    )
    bundle = dict(
        schema="soccer.rsi.body_response_guidance_bundle.v3",
        parent_model_hash=parent_model_hash,
        model_pairs=pairs,
        active_contact_phases=list(active_contact_phases),
        action_dimensions=action_dimensions,
        maximum_accumulated_increment_rad=maximum_accumulated_increment_rad,
        proposal_contract=proposal.contract(),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        activation_ceiling="SIM_ONLY",
        qualification="UNQUALIFIED_SIM_RECOVERY_EXPERIMENT",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    bundle["bundle_hash"] = hash_json(bundle)
    return bundle


class BodyResponseGuidanceExecution:
    """Stateful residual composition, not a hardware executor.

    Contact phase comes only from previous completed forces. The original
    parent protection guard is checked on the actual current observation.
    Total target limits/cap/slew apply after composing the recovery residual.
    """

    def __init__(self, bundle: Any, policy: Any) -> None:
        proof = policy.get("step_motor_proof") if type(policy) is dict else None
        parent = proof.get("model") if type(proof) is dict else None
        if (
            type(bundle) is not dict
            or bundle.get("schema") != "soccer.rsi.body_response_guidance_bundle.v3"
            or bundle.get("activation_ceiling") != "SIM_ONLY"
            or bundle.get("qualification") != "UNQUALIFIED_SIM_RECOVERY_EXPERIMENT"
            or bundle.get("promotion_authorized") is not False
            or bundle.get("hardware_authorized") is not False
            or bundle.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
            or bundle.get("bundle_hash")
            != hash_json({k: v for k, v in bundle.items() if k != "bundle_hash"})
            or type(policy) is not dict
            or "proposal_memory_motor_proof" not in policy
            or type(parent) is not dict
            or parent.get("schema") != "soccer.rsi.proposal_memory_motor.v1"
            or bundle.get("parent_model_hash") != parent.get("model_hash")
        ):
            raise ValueError(
                "complete source-bound SIM guidance and explicit proposal parent required"
            )
        phases = bundle.get("active_contact_phases")
        if type(phases) is not list:
            raise ValueError("explicit complete contact phase configuration required")
        dimensions = bundle.get("action_dimensions")
        maximum = bundle.get("maximum_accumulated_increment_rad")
        if type(dimensions) is not int or type(maximum) is not float:
            raise ValueError("explicit integer dimensions and float recovery bound required")
        self._proposal = BodyResponseRecoveryProposal(
            copy.deepcopy(bundle["model_pairs"]),
            active_contact_phases=tuple(phases),
            action_dimensions=dimensions,
            maximum_accumulated_increment_rad=maximum,
        )
        if hash_json(self._proposal.contract()) != hash_json(bundle["proposal_contract"]):
            raise ValueError("complete guidance prediction sources changed")
        self._memory = ContactPhaseMemory()
        self._dimensions = dimensions
        self._previous = np.zeros((1, self._dimensions))
        self._nonleg_final = np.zeros(self._dimensions - 12)

    @property
    def action_dimensions(self) -> int:
        return int(self._dimensions)

    def advance(
        self,
        decoder: Any,
        body: Any,
        *,
        frame: int,
        nominal_target: Any,
        parent_delta: Any,
        previous_final: Any,
        limits: Any,
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        previous_forces = np.asarray(body["force_n"][frame - 1])[0] if frame else np.zeros(6)
        phase = self._memory.advance(frame, previous_forces)
        nominal, parent, prior, bounds = [
            np.asarray(v, dtype=np.float64)
            for v in (nominal_target, parent_delta, previous_final, limits)
        ]
        if (
            nominal.shape != (29,)
            or parent.shape != (12,)
            or prior.shape != (12,)
            or bounds.shape != (self._dimensions, 2)
            or not all(np.isfinite(v).all() for v in (nominal, parent, prior, bounds))
        ):
            raise ValueError("complete bounded parent composition inputs required")
        state = features_at_frame(
            body,
            frame=frame,
            nominal_target=nominal,
            previous=prior,
            previous_contact_forces=previous_forces,
        )
        context = np.concatenate((decoder.features(state)[:134], [phase]))
        gate = float(decoder._guard.gate(context))
        if not np.isfinite(gate) or not 0 <= gate <= 1:
            raise ValueError("finite original parent protection gate required")
        applied = nominal.copy()
        applied[:12] += parent
        ball = np.concatenate(
            (
                np.asarray(body["ball_position_before_step_m"][frame])[0]
                - np.asarray(body["root_pose_xyzw_m"][frame])[0, :3],
                np.asarray(body["ball_linear_velocity_before_step_m_s"][frame])[0]
                - np.asarray(body["root_velocity_world"][frame])[0, :3],
            )
        )
        proposal = self._proposal.propose(
            qpos=np.asarray(body["canonical_qpos"][frame]),
            qvel=np.asarray(body["canonical_qvel"][frame]),
            nominal_target=applied[None],
            relative_ball=ball[None],
            foundation_input=np.asarray(body["foundation_neural_decoder_input"][frame]),
            previous_increment=self._previous,
            contact_phase=phase,
            protected=gate == 0 or frame < 30,
        )
        parent_all = np.concatenate((parent, np.zeros(self._dimensions - 12)))
        prior_all = np.concatenate((prior, self._nonleg_final))
        proposed = parent_all + np.asarray(proposal["target_increment"])
        lower = np.maximum(np.minimum(0, bounds[:, 0] - nominal[: self._dimensions]), -CAP_RAD)
        upper = np.minimum(np.maximum(0, bounds[:, 1] - nominal[: self._dimensions]), CAP_RAD)
        final: np.ndarray[Any, Any] = np.clip(
            np.clip(proposed, prior_all - SLEW_RAD, prior_all + SLEW_RAD), lower, upper
        )
        # Preserve the exact original decoder result in protected/inactive cases.
        if not proposal["active"]:
            final = parent_all.copy()
        self._previous = (final - parent_all)[None].copy()
        self._nonleg_final = final[12:].copy()
        return final[:12].copy(), dict(
            proposal,
            applied_increment=(final - parent_all).tolist(),
            final_target_increment=final.tolist(),
        )
