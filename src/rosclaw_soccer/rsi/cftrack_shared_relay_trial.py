"""One continuous two-G1 SIM pass and own-history CFtrack continuation."""

from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory
from rosclaw_soccer.rsi.cftrack_shared_handoff import handoff_shared_cftrack
from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
from rosclaw_soccer.rsi.online_receiving_boundary import OnlineReceivingBoundary
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters
from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
from rosclaw_soccer.rsi.shared_relay_setup import motion_segment, receiving_target_for_player
from rosclaw_soccer.rsi.shared_shadow_motor import SharedShadowMotor


def run_shared_cftrack_trial(
    setup: Any,
    actor: Any,
    *,
    context: int,
    lateral_m: float,
    use_cftrack: bool,
    policy_factory: Callable[[], Any],
    expected_policy_type: type[Any],
) -> dict[str, Any]:
    """Use real measured readiness; no body/ball resets after initialization."""
    if type(use_cftrack) is not bool or type(context) is not int or lateral_m not in (-0.35, 0.35):
        raise ValueError("explicit predeclared context, side and study arm required")
    ep, w, forward, home, anchor = setup.prepare(context, lateral_m)
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
    )
    shadow = SharedShadowMotor(
        w,
        "b/",
        actor,
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
            else:
                profile = w.switch_pd_profile("b/", NativePDParameters(ep.kp, ep.kd, ep.effort))
                receiver = shadow.handoff(goal=goal, carry_memory=True)
                handoff = dict(
                    control=step,
                    return_goal=goal.tolist(),
                    profile={k: v for k, v in profile.items() if k not in ("before", "after")},
                )
        mujoco.mj_forward(w.model, w.data)
        if passer.phase == "MOTION_COMPLETE":
            passer.begin_standing()
        target_a, a_trace = passer.propose()
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
        passer.commit()
        if receiver is None:
            shadow.commit_actual()
            if not w.failed:
                history.capture()
                ready = readiness.update()
            shadow_trace.append(s_trace)
        elif use_cftrack:
            receiver.commit()
        else:
            receiver.commit()
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

    return dict(
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
