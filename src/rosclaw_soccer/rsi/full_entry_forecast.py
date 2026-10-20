"""Privileged own-neural forecasts through entry, contact and task end.

Research only: same frozen reference and actual H300, no future targets or
states adopted by the actual actor. Only an explicitly bounded command LAW
may be selected. No training, promotion, hardware or official exam authority.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
from rosclaw_soccer.physics.sonic_navigation_forecast import (
    _clone_body,
    _encode,
    clone_navigation,
    navigation_state_hash,
)
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import quarantine_wrist_reference
from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision
from rosclaw_soccer.rsi.full_task_branch import integration_hash
from rosclaw_soccer.rsi.predictive_ball_entry import entry_candidates
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamMotorTarget


@dataclass(frozen=True)
class EntryCommandLaw:
    """Nominal causal feedback, or20 held ticks then nominal feedback."""

    name: str
    held_command: tuple[float, float, float] | None
    feedback_index: int | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("named entry command law required")
        if self.feedback_index is not None and (
            type(self.feedback_index) is not int
            or not 0 <= self.feedback_index < 6
            or self.held_command is not None
        ):
            raise ValueError("one declared feedback law without held command required")
        c = self.held_command
        if c is not None and (
            type(c) is not tuple
            or len(c) != 3
            or not np.isfinite(c).all()
            or np.linalg.norm(c[:2]) > 0.6000000001
            or abs(c[2]) > 1
        ):
            raise ValueError("bounded finite immutable entry command required")

    def command(
        self, q: np.ndarray, v: np.ndarray, *, frame: int, affordance: SkillEntryAffordance
    ) -> tuple[float, float, float]:
        nominal = ball_entry_decision(q, v, frame=frame, affordance=affordance)
        if nominal.timed_out:
            return 0.0, 0.0, 0.0
        if self.feedback_index is not None and frame >= 20:
            return entry_candidates(q, v, frame=frame, affordance=affordance)[self.feedback_index][
                1
            ]
        if 20 <= frame < 40 and self.held_command is not None:
            return self.held_command
        return nominal.command


def full_entry_laws(
    q: np.ndarray, v: np.ndarray, *, affordance: SkillEntryAffordance
) -> list[EntryCommandLaw]:
    return [EntryCommandLaw("nominal_feedback", None)] + [
        EntryCommandLaw(name + "_held20", cmd)
        for name, cmd in entry_candidates(q, v, frame=20, affordance=affordance)
    ]


def feedback_entry_laws() -> list[EntryCommandLaw]:
    """Fixed six laws, evaluated on each OWN current state until handoff.

    Same bounded gains as the earlier bank, not a hyperparameter sweep. Unlike
    a held20 command, damping reacts to the continuing measured root velocity.
    """
    return [
        EntryCommandLaw(name, None, index)
        for index, name in enumerate(
            (
                "nominal_feedback",
                "half_feedback",
                "zero_feedback",
                "damped_feedback",
                "strong_damped_feedback",
                "brake_feedback",
            )
        )
    ]


@dataclass(frozen=True)
class EntryContactOption:
    law: EntryCommandLaw
    offset_root_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if (
            self.law.feedback_index not in (0, 3, 4)
            or self.law.held_command is not None
            or type(self.offset_root_m) is not tuple
            or self.offset_root_m not in ((0.0, 0.0, 0.0), (0.0, -0.025, -0.025))
        ):
            raise ValueError("fixed coupled command/reference option required")

    @property
    def name(self) -> str:
        return self.law.name + ("__contact_lower" if any(self.offset_root_m) else "__contact_off")


def coupled_entry_options() -> list[EntryContactOption]:
    """Fixed3 existing entry laws x2 already-tested reference requests."""
    laws = feedback_entry_laws()
    return [
        EntryContactOption(laws[i], offset)
        for i in (0, 3, 4)
        for offset in ((0.0, 0.0, 0.0), (0.0, -0.025, -0.025))
    ]


def _observation(
    nav: G1SonicNavigation, body: G1TargetEpisode, frame: int, command: tuple[float, float, float]
) -> TeamMotorObservation:
    q, v = body.canonical()
    return TeamMotorObservation(
        agent_id=nav.agent_id,
        frame=frame,
        time_sec=float(body.data.time),
        intent="shoot",
        prospective_owner=True,
        qpos=tuple(float(x) for x in q),
        qvel=tuple(float(x) for x in v),
        target_position_m=(float(q[36]), float(q[37]), float(q[38])),
        navigation_command=command,
    )


def _tick(
    nav: G1SonicNavigation,
    body: G1TargetEpisode,
    observation: TeamMotorObservation,
    motor_provider: Callable[[G1SonicNavigation, TeamMotorObservation], TeamMotorTarget]
    | None = None,
) -> dict[str, np.ndarray]:
    proposal = (
        nav.propose(observation) if motor_provider is None else motor_provider(nav, observation)
    )
    if nav._next_frame != observation.frame + 1 or nav._batch_pending:
        raise ValueError("motor provider must commit exactly one actual action before integration")
    if not (np.array_equal(proposal.kp, body.kp) and np.array_equal(proposal.kd, body.kd)):
        raise ValueError("entry/contact policy changed native PD gains")
    local = observation.frame - nav._origin_frame
    reference = nav.backend.reference[local : local + 10].copy()
    action = nav.backend.action.copy()
    result = body.step(observation.frame, np.asarray(proposal.target_rad, dtype=np.float64))
    result.update(neural_action_isaac=action[None], encoder_reference_poses=reference[None])
    return result


def _record_step(
    observer: Callable[[dict[str, np.ndarray]], None] | None, row: dict[str, np.ndarray]
) -> None:
    """Observer gets independent read-only copies, never execution buffers."""
    if observer is not None:
        owned = {k: a.copy() for k, a in row.items()}
        for a in owned.values():
            a.setflags(write=False)
        observer(owned)


def nominal_prefix20(
    nav: G1SonicNavigation,
    body: G1TargetEpisode,
    affordance: SkillEntryAffordance,
    *,
    step_observer: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> dict[str, np.ndarray]:
    if step_observer is not None and not callable(step_observer):
        raise ValueError("optional recording observer must be callable")
    if body.next_frame != 0 or nav._next_frame != 0:
        raise ValueError("new continuous actor/body required for prefix")
    rows = []
    for frame in range(20):
        q, v = body.canonical()
        d = ball_entry_decision(q, v, frame=frame, affordance=affordance)
        row = _tick(nav, body, _observation(nav, body, frame, d.command))
        row.update(
            entry_stage=np.array([0], dtype=np.int64),
            entry_command=np.asarray([d.command]),
            entry_target_root_xy_m=np.asarray([d.target_root_xy_m]),
            entry_error_m=np.asarray([d.horizontal_error_m]),
        )
        _record_step(step_observer, row)
        rows.append(row)
    return {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}


def run_entry_suffix(
    nav: G1SonicNavigation,
    body: G1TargetEpisode,
    *,
    model_root: Path,
    source_poses: np.ndarray,
    source_evidence_hash: str,
    affordance: SkillEntryAffordance,
    law: EntryCommandLaw,
    contact_reference_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    step_observer: Callable[[dict[str, np.ndarray]], None] | None = None,
    entry_reference_mode: str = "legacy",
    motor_provider: Callable[[G1SonicNavigation, TeamMotorObservation], TeamMotorTarget]
    | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Execute frame20..299 with its OWN NN and original native integration."""
    if step_observer is not None and not callable(step_observer):
        raise ValueError("optional recording observer must be callable")
    if motor_provider is not None and not callable(motor_provider):
        raise ValueError("optional SIM research motor provider must be callable")
    _validate_entry_mode(entry_reference_mode, contact_reference_offset_m)
    if (
        body.next_frame != 20
        or nav._next_frame != 20
        or nav._origin_frame != 0
        or nav.config.pose_reference is not None
        or nav.config.latent_schedule is not None
        or body.record_all_ball_contacts is not True
        or source_poses.ndim != 2
        or source_poses.shape[1] != 36
        or len(source_poses) < 767
        or not np.isfinite(source_poses).all()
    ):
        raise ValueError("aligned measured locomotion prefix20 and original finite source required")
    rows = []
    stage, handoff_frame, handoff_receipt = 0, None, None
    handoff_hash, reference_hash, wrist_hash = None, None, None
    contact_adaptation = None
    entry_adaptation = None
    target, error = np.zeros(2), 0.0
    for frame in range(20, 300):
        q, v = body.canonical()
        command = (0.0, 0.0, 0.0)
        decision = None
        if stage in (0, 2):
            decision = ball_entry_decision(q, v, frame=frame, affordance=affordance)
            target, error = np.asarray(decision.target_root_xy_m), decision.horizontal_error_m
            command = law.command(q, v, frame=frame, affordance=affordance)
            if decision.timed_out:
                stage = 2
        obs = _observation(nav, body, frame, command)
        if decision is not None and decision.ready:
            held, quarantine = quarantine_wrist_reference(source_poses, q[:36])
            if entry_reference_mode == "legacy":
                prepared = prepare_sonic_reference(
                    held[286:767],
                    q[:36],
                    source_frequency_hz=120,
                    entry_intervals=50,
                    source_evidence_hash=source_evidence_hash,
                )
                reference = prepared.reference
                preparation_hash = prepared.contract_hash
            else:
                from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
                from rosclaw_soccer.providers.g1.sonic_matched_entry import (
                    prepare_matched_entry_pair,
                )

                limits = body.model.jnt_range[
                    [body.model.joint(n).id for n in G1_DDS_JOINT_NAMES]
                ].copy()
                pair = prepare_matched_entry_pair(
                    held[286:767], q[:36], v, limits, source_evidence_hash=source_evidence_hash
                )
                reference = (
                    pair.position if entry_reference_mode == "matched_position" else pair.velocity
                )
                entry_adaptation = pair.report(entry_reference_mode)
                preparation_hash = hash_json(entry_adaptation)
            from rosclaw_soccer.rsi.contact_reference_offset import offset_contact_reference

            adapted_reference, contact_adaptation = offset_contact_reference(
                body.model, reference, offset_root_m=contact_reference_offset_m
            )
            destination = G1SonicNavigation(
                model_root,
                nav.agent_id,
                SonicNavigationConfig(
                    maximum_frames=300,
                    model_variant="low_latency",
                    inference_threads=1,
                    pose_reference=adapted_reference,
                ),
            )
            if destination.backend.qualification != nav.backend.qualification:
                raise ValueError("entry/contact foundation identity changed")
            handoff_hash = integration_hash(body.model, body.data)
            handoff = nav.handoff_to_navigation(destination, observation=obs)
            if integration_hash(body.model, body.data) != handoff_hash:
                raise ValueError("contact handoff changed native physical state")
            handoff_frame, handoff_receipt = frame, asdict(handoff)
            reference_hash, wrist_hash = preparation_hash, quarantine.contract_hash
            nav, stage = destination, 1
        row = (
            _tick(nav, body, obs)
            if motor_provider is None
            else _tick(nav, body, obs, motor_provider)
        )
        row.update(
            entry_stage=np.asarray([stage], dtype=np.int64),
            entry_command=np.asarray([command]),
            entry_target_root_xy_m=target[None].copy(),
            entry_error_m=np.asarray([error]),
        )
        _record_step(step_observer, row)
        rows.append(row)
    arrays = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
    report = dict(
        body=body.review(),
        handoff_frame=handoff_frame,
        handoff_receipt=handoff_receipt,
        handoff_native_state_unchanged_hash=handoff_hash,
        reference_preparation_hash=reference_hash,
        wrist_reference_quarantine_hash=wrist_hash,
        entry_timed_out=stage == 2,
        law=asdict(law),
        neural_policy_recomputed_on_own_state=True,
        contact_controller_retargets_to_ball_after_handoff=False,
        contact_reference_adaptation=contact_adaptation,
        optimizer_steps=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if entry_reference_mode != "legacy":
        report["entry_reference_adaptation"] = entry_adaptation
    if motor_provider is not None:
        report["motor_provider_override"] = True
        report["neural_policy_recomputed_on_own_state"] = False
    return arrays, report


def _validate_entry_mode(mode: str, offset: tuple[float, float, float]) -> None:
    if type(mode) is not str or mode not in ("legacy", "matched_position", "matched_velocity"):
        raise ValueError("explicit declared entry reference mode required")
    if mode != "legacy" and offset != (0.0, 0.0, 0.0):
        raise ValueError("matched entry study permits zero C only; source phase unchanged")


@dataclass(frozen=True)
class FullEntryForecast:
    arrays: dict[str, np.ndarray]
    report: dict[str, Any]
    donor_body_hash: str
    donor_navigation_hash: str
    new_attempted_native_steps: int
    new_integrated_native_steps: int


def body_fields_hash(body: G1TargetEpisode) -> str:
    return hash_json(
        _encode({k: v for k, v in body.__dict__.items() if k not in ("model", "data")})
    )


class FullEntryForecaster:
    """Declared finite280-frame previews; default seven, explicit maximum32."""

    def __init__(self, *, world_hash: str, max_forecasts: int = 7) -> None:
        if type(max_forecasts) is not int or not 1 <= max_forecasts <= 32:
            raise ValueError("explicit integer forecast budget1..32 required")
        self.world_hash = world_hash
        self._max_forecasts = max_forecasts
        self.attempted_native_steps = self.integrated_native_steps = 0
        self.failed = False

    @property
    def max_forecasts(self) -> int:
        return self._max_forecasts

    def forecast(
        self,
        nav: G1SonicNavigation,
        body: G1TargetEpisode,
        *,
        model_root: Path,
        source_poses: np.ndarray,
        source_evidence_hash: str,
        affordance: SkillEntryAffordance,
        law: EntryCommandLaw,
        contact_reference_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        entry_reference_mode: str = "legacy",
    ) -> FullEntryForecast:
        _validate_entry_mode(entry_reference_mode, contact_reference_offset_m)
        if (
            self.failed
            or body.failed
            or body.next_frame != 20
            or nav._next_frame != 20
            or body.record_all_ball_contacts is not True
            or self.attempted_native_steps + 2800 > self.max_forecasts * 2800
        ):
            raise ValueError("healthy aligned prefix20 and remaining full-task budget required")
        G1PDTorqueForecaster(
            body.model,
            expected_world_hash=self.world_hash,
            kp=body.kp,
            kd=body.kd,
            max_forecast_steps=2800,
        )
        donor_body, donor_nav = integration_hash(body.model, body.data), navigation_state_hash(nav)
        donor_fields = body_fields_hash(body)
        branch = None
        origin_attempted, origin_integrated = body.attempted_steps, body.integrated_steps
        try:
            controller, branch = clone_navigation(nav), _clone_body(body)
            arrays, report = run_entry_suffix(
                controller,
                branch,
                model_root=model_root,
                source_poses=source_poses.copy(),
                source_evidence_hash=source_evidence_hash,
                affordance=affordance,
                law=law,
                contact_reference_offset_m=contact_reference_offset_m,
                entry_reference_mode=entry_reference_mode,
            )
            result = FullEntryForecast(
                arrays,
                report,
                donor_body,
                donor_nav,
                branch.attempted_steps - origin_attempted,
                branch.integrated_steps - origin_integrated,
            )
        except Exception:
            self.failed = True
            raise
        finally:
            if branch is not None:
                self.attempted_native_steps += branch.attempted_steps - origin_attempted
                self.integrated_native_steps += branch.integrated_steps - origin_integrated
            if (
                integration_hash(body.model, body.data) != donor_body
                or navigation_state_hash(nav) != donor_nav
                or body_fields_hash(body) != donor_fields
            ):
                self.failed = True
                raise ValueError("full-task prediction modified the donor actor/body")
        return result


def score_full_entry(model: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Explicit research selection criterion, NOT the original RSI-M0 exam."""
    q = arrays["target_body_canonical_qpos_after_substep"]
    v = arrays["target_body_canonical_qvel_after_substep"]
    forces = arrays["ball_contact_force_by_body_n"]
    canonical = arrays["canonical_qpos"]
    if (
        q.shape != (280, 10, 43)
        or v.shape != (280, 10, 41)
        or forces.shape != (280, 10, model.nbody)
        or canonical.shape != (280, 43)
        or any(not np.isfinite(a).all() for a in (q, v, forces, canonical))
        or np.any(forces < 0)
    ):
        raise ValueError("complete finite frame20..299 own-native task outcome required")
    root = model.body_rootid[model.body("pelvis").id]
    feet = [
        model.body(side + suffix).id
        for side in ("left", "right")
        for suffix in ("_ankle_roll_link", "_ankle_pitch_link")
    ]
    nonfeet = np.flatnonzero((model.body_rootid == root) & ~np.isin(np.arange(model.nbody), feet))
    if root == 0 or not len(nonfeet):
        raise ValueError("complete robot body-tree classification required")
    foot_indices = np.flatnonzero(np.any(forces.reshape(2800, model.nbody)[:, feet] > 1, axis=1))
    touch = int(foot_indices[0]) if len(foot_indices) else None
    forward, lateral = (q[-1, -1, 36:38] - canonical[0, 36:38]).tolist()
    peak_nonfoot = float(forces[:, :, nonfeet].max())
    height = float(q[:, :, 2].min())
    upright = float((1 - 2 * (q[:, :, 4] ** 2 + q[:, :, 5] ** 2)).min())
    max_vx = float(v[:, :, 35].max())
    physical = peak_nonfoot <= 1 and height >= 0.65 and upright >= 0.9
    # Require a complete60-frame post-touch observation, without extending H300.
    touch_frame = 20 + touch // 10 if touch is not None else None
    observable = touch_frame is not None and touch_frame <= 239
    qualified = bool(
        physical and observable and forward >= 0.5 and abs(lateral) <= 0.3 and max_vx >= 0.5
    )
    late_speed2 = float(np.mean(np.sum(v[-60:, :, :2] ** 2, axis=-1)))
    if touch is None:
        drift = None
    else:
        root_xy = q[:, :, :2].reshape(2800, 2)
        drift = float(np.linalg.norm(root_xy[touch:] - root_xy[touch], axis=1).max())
    cost = None
    if qualified:
        assert drift is not None
        cost = lateral**2 + 0.02 * late_speed2 + 0.02 * drift**2 - 0.01 * min(forward, 3)
    return dict(
        schema="soccer.rsi.full_entry_research_selection.v1",
        qualified=qualified,
        cost=cost,
        physical_prefix_and_suffix_safe=bool(physical),
        first_foot_contact_global_frame=touch_frame,
        complete_post60_observable=bool(observable),
        forward_to_task_end_m=forward,
        lateral_to_task_end_m=lateral,
        maximum_ball_vx_mps=max_vx,
        maximum_nonfoot_force_n=peak_nonfoot,
        minimum_pelvis_z_m=height,
        minimum_upright=upright,
        late60_mean_root_horizontal_speed_squared=late_speed2,
        post_touch_maximum_root_drift_m=drift,
        official_football_exam_pass=False,
        promotion_authorized=False,
    )
