"""Same-player neural history handoff; no physical state or reference writes.

This adapter transfers measured proprioceptive history into an explicitly reset
new option, not a new robot. The shared simulation owner supplies both immutable
observations and owns their identity/clock provenance. This is not an authority
token, a simulator-equivalence proof or a policy promotion mechanism.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation


@dataclass(frozen=True)
class SonicHistoryHandoffReceipt:
    agent_id: str
    frame: int
    time_sec: float
    history_hash: str
    binding_hash: str
    activation_ceiling: str = "SIM_ONLY"


def _parameters(controller: G1SonicRunupController) -> dict[str, object]:
    result: dict[str, object] = {"foundation": controller.qualification.qualification_hash}
    for name in ("default_angles", "_kp", "_kd", "_action_scale"):
        value = np.asarray(getattr(controller, name))
        if value.shape != (29,) or value.dtype.kind not in "fi" or not np.isfinite(value).all():
            raise ValueError("finite qualified SONIC normalization parameters required")
        result[name] = value.tolist()
    return result


def handoff_sonic_history(
    *,
    source: G1SonicRunupController,
    destination: G1SonicRunupController,
    source_observation: TeamMotorObservation,
    destination_observation: TeamMotorObservation,
) -> SonicHistoryHandoffReceipt:
    """Validate everything before replacing the new option's private history.

    Destination must have just been reset from the same measured body state.
    Only inference-copy XY translation of body AND ball is permitted; physical
    course transforms need the separate contact-course-frame validation. Source
    must have committed its latest post-control observation. Reference schedules
    stay separate and neither controller's physics state is touched.
    """
    if (
        not isinstance(source, G1SonicRunupController)
        or not isinstance(destination, G1SonicRunupController)
        or source is destination
        or getattr(destination, "_history_handoff_binding", None) is not None
        or not isinstance(source_observation, TeamMotorObservation)
        or not isinstance(destination_observation, TeamMotorObservation)
        or source_observation.agent_id != destination_observation.agent_id
        or source_observation.frame != destination_observation.frame
        or abs(source_observation.time_sec - destination_observation.time_sec) > 1e-8
        or abs(source_observation.time_sec - source_observation.frame * 0.02) > 1e-6
    ):
        raise ValueError("distinct same-player options at one measured control boundary required")
    parameters = _parameters(source)
    if parameters != _parameters(destination):
        raise ValueError("SONIC foundation or action normalization differs")
    sq, dq = (np.asarray(o.qpos) for o in (source_observation, destination_observation))
    sv, dv = (np.asarray(o.qvel) for o in (source_observation, destination_observation))
    if any(
        abs(float(np.linalg.norm(q[s])) - 1) > 1e-4
        for q in (sq, dq)
        for s in (slice(3, 7), slice(39, 43))
    ):
        raise ValueError("unit measured body and ball orientations required")
    unchanged = np.ones(43, dtype=bool)
    unchanged[[0, 1, 36, 37]] = False
    if (
        not np.allclose(sq[unchanged], dq[unchanged], rtol=0, atol=1e-8)
        or not np.allclose(sv, dv, rtol=0, atol=1e-8)
        or not np.allclose(dq[:2] - sq[:2], dq[36:38] - sq[36:38], rtol=0, atol=1e-8)
    ):
        raise ValueError("handoff observations do not describe the same measured state")
    histories = []
    for controller in (source, destination):
        if len(controller._history) != 10:
            raise ValueError("complete ten-frame SONIC history required")
        history = []
        for entry in controller._history:
            if len(entry) != 5:
                raise ValueError("five proprioceptive history fields required")
            copied = []
            for value, size in zip(entry, (3, 29, 29, 29, 3), strict=True):
                value = np.asarray(value)
                if (
                    value.shape != (size,)
                    or value.dtype.kind not in "fi"
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("finite complete proprioceptive history required")
                copied.append(value.copy())
            history.append(tuple(copied))
        histories.append(history)
    source_action = np.asarray(source.action)
    destination_action = np.asarray(destination.action)
    if (
        source_action.shape != (29,)
        or source_action.dtype.kind not in "fi"
        or not np.isfinite(source_action).all()
        or destination_action.shape != (29,)
        or np.any(destination_action != 0)
    ):
        raise ValueError("source action and a freshly reset destination required")
    source_entry = source._history_entry(SimpleNamespace(qpos=sq, qvel=sv), source_action)
    destination_entry = destination._history_entry(SimpleNamespace(qpos=dq, qvel=dv), np.zeros(29))
    if any(
        not np.allclose(a, b, rtol=0, atol=1e-8)
        for a, b in zip(histories[0][-1], source_entry, strict=True)
    ):
        raise ValueError("source has not observed the current post-control state")
    if any(
        not np.allclose(a, b, rtol=0, atol=1e-8)
        for entry in histories[1]
        for a, b in zip(entry, destination_entry, strict=True)
    ):
        raise ValueError("destination already advanced beyond its declared reset")
    reference_hashes = []
    for controller in (source, destination):
        reference = np.asarray(controller.reference)
        if (
            reference.ndim != 2
            or reference.shape[1] != 36
            or not len(reference)
            or not np.isfinite(reference).all()
        ):
            raise ValueError("initialized finite reference schedules required")
        reference_hashes.append(hash_bytes(np.ascontiguousarray(reference).tobytes()))
    history_hash = str(hash_json([[v.tolist() for v in entry] for entry in histories[0]]))
    binding = str(
        hash_json(
            {
                "schema": "rosclaw_soccer.sonic_history_handoff.v1",
                "agent_id": source_observation.agent_id,
                "frame": source_observation.frame,
                "time_sec": source_observation.time_sec,
                "parameters": parameters,
                "history_hash": history_hash,
                "source_reference_hash": reference_hashes[0],
                "destination_reference_hash": reference_hashes[1],
                "source_qpos": source_observation.qpos,
                "destination_qpos": destination_observation.qpos,
                "qvel": source_observation.qvel,
                "activation_ceiling": "SIM_ONLY",
            }
        )
    )
    # New arrays prevent a later source observation from changing the recipient.
    destination._history = collections.deque(histories[0], maxlen=10)
    destination.action = source_action.copy()
    destination._history_handoff_binding = binding
    return SonicHistoryHandoffReceipt(
        source_observation.agent_id,
        source_observation.frame,
        source_observation.time_sec,
        history_hash,
        binding,
    )
