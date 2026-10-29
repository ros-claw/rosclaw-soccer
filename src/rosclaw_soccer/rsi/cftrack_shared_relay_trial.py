"""One continuous two-G1 SIM pass and own-history CFtrack continuation."""

from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory
from rosclaw_soccer.rsi.cftrack_shared_handoff import handoff_shared_cftrack
from rosclaw_soccer.rsi.live_neural_transition import blend_live_targets, quintic_weight
from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
from rosclaw_soccer.rsi.online_receiving_boundary import OnlineReceivingBoundary
from rosclaw_soccer.rsi.paid_velocity_shadow import policy_action_from_hg_target
from rosclaw_soccer.rsi.passer_velocity_recovery import PasserVelocityRecovery
from rosclaw_soccer.rsi.receiving_witness import body_contact_witness, readiness_witness
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters
from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
from rosclaw_soccer.rsi.shared_relay_setup import motion_segment, receiving_target_for_player
from rosclaw_soccer.rsi.shared_shadow_motor import SharedShadowMotor


def new_passer_recovery(setup: Any, world: Any) -> PasserVelocityRecovery:
    """Use the existing continuous own-history stream for this H600 world."""
    stream = ContinuousVelocityStream(
        setup.velocity_path,
        setup.velocity_config,
        expected_model_hash=setup.pins[str(setup.velocity_path)],
        expected_config_hash=setup.pins[str(setup.velocity_config)],
        horizon_controls=600,
    )
    return PasserVelocityRecovery(world, "a/", stream)


def early_incoming_boundary(world: Any, prefix: str, forward: np.ndarray) -> tuple[bool, float]:
    """Causal SIM trigger, not an indication of contact or a qualified pass."""
    world._require_current_boundary()
    if (
        prefix not in world.layouts
        or not isinstance(forward, np.ndarray)
        or forward.shape != (2,)
        or not np.isfinite(forward).all()
        or abs(float(np.linalg.norm(forward)) - 1.0) > 1e-10
    ):
        raise ValueError("actual body and finite unit pass direction required")
    _, velocity = world.layouts[prefix].canonical(world.data)
    if velocity.shape != (41,) or not np.isfinite(velocity).all():
        raise ValueError("finite actual canonical velocity required")
    toward = float(velocity[35:37] @ forward)
    return bool(world.control_steps >= 5 and not world.failed and toward > 0.4), toward


def run_shared_cftrack_trial(
    setup: Any,
    actor: Any,
    *,
    context: int,
    lateral_m: float,
    use_cftrack: bool,
    policy_factory: Callable[[], Any],
    expected_policy_type: type[Any],
    early_cftrack: bool = False,
    receiver_actor: Any | None = None,
    record_witness: bool = False,
    passer_velocity_recovery: bool = False,
    ball_planar_offset_world: np.ndarray | None = None,
    passer_reference_rate: float = 1.0,
) -> dict[str, Any]:
    """Use real measured readiness; no body/ball resets after initialization."""
    if (
        type(use_cftrack) is not bool
        or type(early_cftrack) is not bool
        or type(record_witness) is not bool
        or type(passer_velocity_recovery) is not bool
        or type(context) is not int
        or lateral_m not in (-0.35, 0.35)
        or (receiver_actor is not None and use_cftrack)
        or type(passer_reference_rate) is not float
        or not np.isfinite(passer_reference_rate)
        or not 0.5 <= passer_reference_rate <= 1.0
    ):
        raise ValueError("explicit predeclared context, side and study arm required")
    from rosclaw_soccer.rsi.pass_transfer_curriculum import initial_ball_offset

    offset = initial_ball_offset(ball_planar_offset_world)
    ep, w, forward, home, anchor = (
        setup.prepare(context, lateral_m)
        if offset is None
        else setup.prepare(context, lateral_m, ball_planar_offset_world=offset)
    )
    initial_q, initial_v = w.data.qpos.copy(), w.data.qvel.copy()
    segment = motion_segment(ep.clip, 80)
    passer = SharedNativeNeuralPlayer(
        w,
        "a/",
        actor,
        clip=ep.clip,
        torso_index=ep.torso_index,
        build_observation=ep.build_observation,
        goal=ep.goal,
        offset=ep.offset,
        scale=ep.scale,
        order=ep.order,
        reference_rate=passer_reference_rate,
    )
    shadow = SharedShadowMotor(
        w,
        "b/",
        actor if receiver_actor is None else receiver_actor,
        clip=segment,
        torso_index=ep.torso_index,
        build_observation=ep.build_observation,
        goal=receiving_target_for_player(w, "a/"),
        offset=ep.offset,
        scale=ep.scale,
        order=ep.order,
    )
    history = CFtrackLiveHistory(w, "b/")
    velocity, readiness = setup.velocity(), OnlineReceivingBoundary(w, "b/")
    recovery = new_passer_recovery(setup, w) if passer_velocity_recovery else None
    passer_foot_seen = False
    recovery_boundary = None
    recovery_old_pd = recovery_new_pd = None
    passer_velocity_trace = []
    receiver = None
    ready = launch = False
    handoff = None
    physical, cf_trace, dual_trace, shadow_trace, velocity_trace, passer_trace = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    termination = "WORLD_HORIZON_COMPLETE"
    contacts, readiness_records = [], []
    for step in range(600):
        if receiver is None and not ready and step >= 300:
            termination = "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300"
            break
        if receiver is None and ready:
            goal = receiving_target_for_player(w, "a/")
            if use_cftrack:
                policy = policy_factory()
                receiver, transition = handoff_shared_cftrack(
                    w, "b/", history, policy, expected_policy_type=expected_policy_type, goal=goal
                )
                handoff = dict(control=step, return_goal=goal.tolist(), transition=transition)
                if early_cftrack:
                    _, speed = early_incoming_boundary(w, "b/", forward)
                    handoff["incoming_ball_speed_along_pass_mps"] = speed
            else:
                profile = w.switch_pd_profile("b/", NativePDParameters(ep.kp, ep.kd, ep.effort))
                receiver = shadow.handoff(goal=goal, carry_memory=True)
                handoff = dict(
                    control=step,
                    return_goal=goal.tolist(),
                    profile={k: v for k, v in profile.items() if k not in ("before", "after")},
                )
        mujoco.mj_forward(w.model, w.data)
        if passer.phase == "MOTION_COMPLETE" and (recovery is None or not recovery.active):
            if recovery is not None and passer_foot_seen:
                params = recovery.stream.params
                recovery_old_pd = w.parameters["a/"].owned_copy()
                recovery_new_pd = NativePDParameters(
                    np.asarray(params.stiffness, np.float64),
                    np.asarray(params.damping, np.float64),
                    recovery_old_pd.effort.copy(),
                )
                recovery.activate(motion_complete=True, passer_foot_seen=True)
                recovery_boundary = step
            passer.begin_standing()
        recovery_target = None
        if recovery is not None:
            recovery_target = recovery.observe(np.zeros(3, np.float32))
            recovery_trace = dict(
                control=np.array(step),
                obs=recovery.stream.last_observation.copy(),
                proposed_action=recovery.stream.last_action.copy(),
                phase=np.array(recovery.stream.phase),
            )
        blend_index = step - recovery_boundary if recovery_boundary is not None else -1
        blend_active = 0 <= blend_index < 20
        executed_a = None
        if recovery is not None and recovery.active and not blend_active:
            if blend_index == 20:
                w.switch_pd_profile("a/", recovery_new_pd)
            target_a, a_trace = recovery_target, None
        else:
            target_a, a_trace = passer.propose()
        if blend_active:
            weight = quintic_weight(blend_index, start=0, duration=20)
            target_a, profile = blend_live_targets(
                target_a, recovery_target, recovery_old_pd, recovery_new_pd, weight=weight
            )
            w.switch_pd_profile("a/", profile)
            executed_a = policy_action_from_hg_target(
                target_a, passer.offset, passer.scale, passer.order
            )
            target_a = passer.offset + executed_a[passer.order] * passer.scale
        if receiver is None:
            s_trace = shadow.observe()
            q, v = w.layouts["b/"].canonical(w.data)
            qw, x, y, z = q[3:7]
            yaw = float(np.arctan2(2 * (qw * z + x * y), 1 - 2 * (y * y + z * z)))
            intent = receiving_intent(
                root_xy=q[:2],
                yaw=yaw,
                ball_xy=q[36:38],
                ball_velocity_xy=v[35:37],
                home_xy=home,
                pass_direction=forward,
                launch_seen=launch,
                ready_anchor_xy=anchor,
                mode="pursuit-v1",
            )
            launch = intent.launch_seen
            command = np.asarray(intent.command_body, np.float32)
            target_b = velocity.propose(
                q, v, w.layouts["b/"].angular_velocity_body(w.data), command, frame=step
            )
            velocity_trace.append(
                dict(
                    control=np.array(step),
                    obs=velocity.last_observation.copy(),
                    action=velocity.last_action.copy(),
                    command=command.copy(),
                    phase=np.array(velocity.phase),
                )
            )
        elif use_cftrack:
            target_b, _ = receiver.propose()
            cf_trace.append(
                dict(
                    control=np.array(step),
                    obs=receiver.policy.obs.copy(),
                    action=receiver.policy.action.copy(),
                    target=target_b.copy(),
                )
            )
        else:
            if receiver.phase == "MOTION_COMPLETE":
                receiver.begin_standing()
            target_b, b_trace = receiver.propose()
            dual_trace.append(b_trace)
        row = w.advance({"a/": target_a, "b/": target_b})
        if record_witness:
            contacts.append(body_contact_witness(w))
        if a_trace is not None:
            if blend_active:
                passer.commit_sim_transition(executed_action=executed_a, target=target_a)
            else:
                passer.commit()
        if recovery is not None:
            recovery_row = recovery.commit(
                transition_target=np.asarray(target_a, np.float64) if blend_active else None
            )
            recovery_trace.update({k: np.asarray(v) for k, v in recovery_row.items()})
            recovery_trace["blend_index"] = np.array(blend_index)
            passer_velocity_trace.append(recovery_trace)
        passer_foot_seen = passer_foot_seen or bool((row["ball_forces"][:, 0, :2] > 10).any())
        if receiver is None:
            shadow.commit_actual()
            if not w.failed:
                history.capture()
                stable_ready = readiness.update()
                if record_witness:
                    readiness_records.append(readiness_witness(readiness, stable_ready))
                ready = (
                    early_incoming_boundary(w, "b/", forward)[0]
                    if use_cftrack and early_cftrack
                    else stable_ready
                )
            shadow_trace.append(s_trace)
        elif use_cftrack:
            receiver.commit()
        else:
            receiver.commit()
        if a_trace is not None:
            passer_trace.append(a_trace)
        physical.append(
            {k: row[k] for k in ("qpos", "qvel", "torque", "ball_forces")}
            | dict(
                body_failed=np.array([row["body_failed"][p] for p in ("a/", "b/")]),
                time_before=np.array(row["time_before"]),
                target_a=np.asarray(target_a, np.float64),
                target_b=np.asarray(target_b, np.float64),
            )
        )
        if w.failed:
            termination = "BODY_FAILURE"
            break

    def stack(rows):
        return {key: np.stack([row[key] for row in rows]) for key in rows[0]} if rows else {}

    result = dict(
        termination=termination,
        controls=w.control_steps,
        native_steps=w.native_steps,
        handoff=handoff,
        initial=dict(qpos=initial_q, qvel=initial_v),
        physics=stack(physical),
        passer_nn=stack(passer_trace),
        receiver_return_nn=stack(dual_trace),
        receiver_shadow=stack(shadow_trace),
        receiver_velocity=stack(velocity_trace),
        cftrack=stack(cf_trace),
    )
    if record_witness:
        result["witness"] = dict(contacts=contacts, readiness=readiness_records)
    if passer_velocity_recovery:
        result["passer_velocity"] = stack(passer_velocity_trace)
        result["passer_recovery_boundary"] = recovery_boundary
    return result
