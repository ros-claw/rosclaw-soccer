"""Experimental learned post-contact proposals; no actuator or execution entry."""

from pathlib import Path
from typing import Any

import numpy as np
from rosclaw.growth import bounded_response_proposal as proposal_module
from rosclaw.growth.compiled_context_prediction import CompiledContextPrediction

from rosclaw_soccer.rsi import body_response_field
from rosclaw_soccer.rsi.body_response_features import _matrix, measured_response_features
from rosclaw_soccer.rsi.body_response_field import BodyResponseField
from rosclaw_soccer.sim.contracts import hash_bytes


class BodyResponseRecoveryProposal:
    """Fixed four-pair ensemble, active only in externally verified phase 2.

    This is a learned dynamics-guided 12-joint POSITION residual proposal,
    not a torque policy, physical safety proof, or permission to execute.
    Contact memory, parent-policy protection, final limits, and independent
    physics replay must be supplied by the downstream simulation executor.
    """

    def __init__(self, model_pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
        if not isinstance(model_pairs, list) or len(model_pairs) != 4:
            raise ValueError("exactly four fixed body response model pairs required")
        self._fields = [BodyResponseField(a, b, implementation="compiled") for a, b in model_pairs]
        self._baselines = [CompiledContextPrediction(a) for a, _ in model_pairs]
        self._pins = {
            str(p): hash_bytes(p.read_bytes())
            for p in (
                Path(__file__),
                Path(body_response_field.__file__),
                Path(proposal_module.__file__),
            )
        }

    def contract(self) -> dict[str, Any]:
        return dict(
            schema="soccer.rsi.body_response_recovery_proposal.v1",
            fields=[f.contract() for f in self._fields],
            source_pins=dict(self._pins),
            fixed_pair_weights=[0.25] * 4,
            maximum_increment_rad=0.02,
            maximum_increment_change_rad=0.002,
            regularization=0.05,
            active_contact_phase=2,
            outcome_conditioned_selection=False,
            physical_validity_verified=False,
            runtime_execution_authorized=False,
            activation_ceiling="SIM_ONLY",
            promotion_authorized=False,
            hardware_authorized=False,
        )

    def propose(
        self,
        *,
        qpos: Any,
        qvel: Any,
        nominal_target: Any,
        relative_ball: Any,
        foundation_input: Any,
        previous_increment: Any,
        contact_phase: int,
        protected: bool,
    ) -> dict[str, Any]:
        if (
            type(contact_phase) is not int
            or contact_phase not in (0, 1, 2)
            or type(protected) is not bool
        ):
            raise ValueError("explicit causal phase and parent protection required")
        result: dict[str, Any] = dict(
            target_increment=[0.0] * 12,
            active=False,
            fallback=False,
            contact_phase=contact_phase,
            protected=protected,
            physical_validity_verified=False,
            runtime_execution_authorized=False,
            hardware_authorized=False,
        )
        if protected or contact_phase != 2:
            return result
        try:
            if any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items()):
                raise ValueError("body guidance dependency source changed")
            q, v, target, ball, foundation, previous = (
                _matrix(value, width)
                for value, width in (
                    (qpos, 43),
                    (qvel, 41),
                    (nominal_target, 29),
                    (relative_ball, 6),
                    (foundation_input, 994),
                    (previous_increment, 12),
                )
            )
            if (
                any(len(a) != 1 for a in (q, v, target, ball, foundation, previous))
                or np.max(np.abs(previous)) > 0.02
            ):
                raise ValueError("one bounded current body state required")
            features = measured_response_features(q, v, target, ball)
            base = np.concatenate((foundation, ball, target, v[:, :35]), axis=1)
            next_velocity = v[0, :35] + np.mean(
                [m.predict(base)[0] for m in self._baselines], axis=0
            )
            # Root angular velocity is in the native body's local axes.
            desired = np.zeros(35)
            desired[3:5] = [features[0, 2], -features[0, 1]]
            error = next_velocity - desired
            weights = np.zeros(35)
            weights[2] = 0.2
            weights[3:5] = 1.0
            state = dict(
                qpos=np.repeat(q, 24, axis=0),
                qvel=np.repeat(v, 24, axis=0),
                nominal_target=np.repeat(target, 24, axis=0),
                relative_ball=np.repeat(ball, 24, axis=0),
                foundation_input=np.repeat(foundation, 24, axis=0),
            )
            probes = np.zeros((24, 12))
            for j in range(12):
                probes[2 * j, j], probes[2 * j + 1, j] = -0.01, 0.01
            effects = np.mean(
                [f.predict_effect(**state, target_increment=probes) for f in self._fields], axis=0
            )
            matrix = ((effects[1::2] - effects[::2]) / 0.02).T
            proposal = proposal_module.bounded_response_proposal(
                matrix,
                error,
                weights,
                maximum_increment=0.02,
                regularization=0.05,
            )
            delta = np.clip(
                np.asarray(proposal["target_increment"]), previous[0] - 0.002, previous[0] + 0.002
            )
            single = dict(
                qpos=q,
                qvel=v,
                nominal_target=target,
                relative_ball=ball,
                foundation_input=foundation,
            )
            predicted_effect = np.mean(
                [f.predict_effect(**single, target_increment=delta[None]) for f in self._fields],
                axis=0,
            )[0]
            before = float(np.sum(weights * error**2))
            after = float(
                np.sum(weights * (error + predicted_effect) ** 2) + 0.05 * np.sum(delta**2)
            )
            if not np.isfinite(after) or after > before:
                raise ValueError("bounded nonlinear prediction rejected proposal")
            result.update(
                target_increment=delta.tolist(),
                active=True,
                predicted_baseline_cost=before,
                predicted_candidate_cost=after,
            )
        except (ValueError, OSError, FloatingPointError):
            # Exact parent fallback has precedence over residual slew.
            # The executor must still apply its own final total-policy limits.
            result["fallback"] = True
        return result
