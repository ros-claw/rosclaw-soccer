"""One bounded cold SIM torque trial; immutable raw evidence returned to caller."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_torque_intent import (
    CFtrackTorqueIntentPlan,
    CFtrackTorqueIntentPolicy,
)
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
from rosclaw_soccer.rsi.shared_native_torque_layer import SharedNativeTorqueLayer
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer


def complete_torque_trial(run: dict[str, Any], *, controls: int = 160) -> bool:
    """Require the entire planned controller/physics/feedback chain, not a prefix."""
    if type(controls) is not int or not 1 <= controls <= 300:
        raise ValueError("bounded declared control horizon required")
    return (
        run.get("controls") == controls
        and run.get("native_steps") == controls * 10
        and run.get("neural_calls") == controls
        and run.get("layer_commands") == controls * 10
        and run.get("layer_feedback") == controls * 10
        and run.get("incomplete_control_native_steps") == 0
        and run.get("runtime_failure") is None
        and run.get("body_failed") is False
    )


class TorqueTrialCapture:
    def __init__(self, session: Any, *, maximum_controls: int = 160):
        if type(maximum_controls) is not int or not 1 <= maximum_controls <= 300:
            raise ValueError("bounded declared NN horizon required")
        self.session = session
        self.maximum_controls = maximum_controls
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []

    def run(self, outputs: Any, inputs: Any) -> Any:
        if (
            len(self.observations) >= self.maximum_controls
            or inputs["obs"].shape != (1, 1244)
            or not np.isfinite(inputs["obs"]).all()
            or not np.array_equal(inputs["time_step"], np.zeros((1, 1), np.float32))
        ):
            raise ValueError("bounded finite original NN1244/zero timestep required")
        result = self.session.run(outputs, inputs)
        self.observations.append(inputs["obs"].copy())
        self.actions.append(result[0].copy())
        return result


def run_cold_torque_trial(
    player: SharedOmniContactPlayer, plan: CFtrackTorqueIntentPlan, *, controls: int = 160
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    w = player.world
    w._require_current_boundary()
    if (
        type(player) is not SharedOmniContactPlayer
        or type(plan) is not CFtrackTorqueIntentPlan
        or type(controls) is not int
        or not 1 <= controls <= 300
        or w.native_steps != 0
        or w.control_steps != 0
        or player.policy.counter_step != 0
        or player.pending is not None
        or player.failed
        or player.motor_plan is not None
        or w._torque_layer_owners
        or len(w.layouts) != 1
        or plan.parent_policy_hash != cftrack_state_hash(player.policy)
    ):
        raise ValueError("one untouched private cold player and matching torque intention required")
    policy = CFtrackTorqueIntentPolicy(plan, w.parameters[player.prefix].effort)
    layer = SharedNativeTorqueLayer(
        w,
        player.prefix,
        policy,
        target_y_m=0.0,
        target_z_m=0.11,
        horizon_controls=controls,
        expected_parent_policy_hash=plan.parent_policy_hash,
    )
    original = player.policy.ort_session
    capture = TorqueTrialCapture(original, maximum_controls=controls)
    player.policy.ort_session = capture
    initial_qpos, initial_qvel = w.data.qpos.copy(), w.data.qvel.copy()
    rows = []
    targets = []
    decoded = []
    failure = None
    try:
        for _ in range(controls):
            target, measured = player.propose()
            row = w.advance({player.prefix: target}, torque_layers={player.prefix: layer})
            rows.append(row)
            targets.append(target)
            decoded.append(measured["nn_decoded_target"])
            player.commit()
            if w.failed:
                break
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    finally:
        player.policy.ort_session = original
    raw = {
        k: np.concatenate([r[k] for r in rows]) if rows else np.empty((0, width))
        for k, width in [
            ("qpos", w.model.nq),
            ("qvel", w.model.nv),
            ("torque", w.model.nu),
            ("parent_torque", w.model.nu),
        ]
    }
    raw.update(
        ball_forces=np.concatenate([r["ball_forces"] for r in rows])
        if rows
        else np.empty((0, 1, 3)),
        joint_targets=np.array(targets).reshape(-1, 29),
        nn_decoded_targets=np.array(decoded).reshape(-1, 29),
        neural_obs=np.array(capture.observations).reshape(-1, 1, 1244),
        neural_actions=np.array(capture.actions).reshape(-1, 1, 29),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        terminal_qpos=w.data.qpos.copy(),
        terminal_qvel=w.data.qvel.copy(),
    )
    return raw, dict(
        controls=len(rows),
        native_steps=w.native_steps,
        neural_calls=len(capture.observations),
        runtime_failure=failure,
        body_failed=w.failed,
        layer_commands=layer.commands,
        layer_feedback=layer.applied,
        saturation_steps=policy.saturation_steps,
        measured_stop_guard_steps=policy.stop_guard_steps,
        incomplete_control_native_steps=w.native_steps - len(rows) * 10,
        actual_changed_torque_entries=int((raw["torque"] != raw["parent_torque"]).sum()),
    )
