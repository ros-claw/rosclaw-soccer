"""Owned SIM forecasts with copied neural history and the SAME native PD law.

Unlike held-target prediction, every later control calls the original NN on
the predicted body's measured state. Only a stateless CPU ORT session is shared.
No donor history/world mutation, target playback, root forcing or model changes.
"""

import copy
import hashlib
from collections import deque
from enum import Enum
from importlib import import_module
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.motion_reference_timing import MotionTimingPlan, condition_motion_reference
from rosclaw_soccer.rsi.motion_synergy_reference import (
    MotionSynergyPlan,
    condition_synergy_reference,
)
from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from rosclaw_soccer.rsi.whole_joint_reference import (
    WholeJointReferencePlan,
    condition_whole_joint_reference,
)


def policy_state_hash(policy: Any) -> str:
    """Validate a finite data-only policy graph before copying or fingerprinting.

    Unknown resources/callables/transport objects are rejected, not deep-copied.
    The qualified caller owns the pure policy/state/output classes' provenance.
    """
    allowed = (type(policy), type(policy.state_cmd), type(policy.policy_output))
    h = hashlib.sha256()
    active: set[int] = set()

    def visit(value: Any) -> None:
        kind = type(value)
        h.update((kind.__module__ + "." + kind.__qualname__).encode())
        if value is None or kind in (str, bool, int, float) or isinstance(value, Enum):
            if kind is float and not np.isfinite(value):
                raise ValueError("nonfinite policy scalar")
            h.update(repr(value).encode())
        elif isinstance(value, np.generic):
            visit(value.item())
        elif kind is np.ndarray:
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError("finite numeric policy arrays required")
            h.update(str(value.dtype).encode())
            h.update(repr(value.shape).encode())
            h.update(value.tobytes())
        else:
            if id(value) in active:
                raise ValueError("cyclic policy software state is unsupported")
            active.add(id(value))
            try:
                if kind is dict:
                    if any(type(key) is not str for key in value):
                        raise ValueError("data-only string-key policy dictionaries required")
                    for key in sorted(value):
                        visit(key)
                        visit(value[key])
                elif kind in (list, tuple, deque):
                    h.update(repr((len(value), value.maxlen if kind is deque else None)).encode())
                    for item in value:
                        visit(item)
                elif kind in allowed:
                    visit(vars(value))
                else:
                    raise ValueError("unsupported resource in SIM policy state: " + repr(kind))
            finally:
                active.remove(id(value))

    visit({k: v for k, v in vars(policy).items() if k != "ort_session"})
    return "sha256:" + h.hexdigest()


def copy_sim_policy(policy: Any) -> Any:
    import onnxruntime as ort

    if (
        policy.runtime_mode != "sim"
        or policy.use_body_frame_ball
        or type(policy.ort_session) is not ort.InferenceSession
        or policy.ort_session.get_providers() != ["CPUExecutionProvider"]
    ):
        raise ValueError("pure SIM policy with original stateless CPU ORT session required")
    before = policy_state_hash(policy)
    state = copy.deepcopy({k: v for k, v in vars(policy).items() if k != "ort_session"})
    cloned = object.__new__(type(policy))
    vars(cloned).update(state)
    cloned.ort_session = policy.ort_session
    if policy_state_hash(cloned) != before:
        raise ValueError("copied software history changed")
    return cloned


def _fork_pending_player(
    player: SharedRoboNaldoPlayer,
    world: SharedNativeMotorWorld,
    policy: Any,
    *,
    before_proposal: bool = False,
) -> SharedRoboNaldoPlayer:
    """Fork already-admitted software into a separate owned SIM world, NO enter()."""
    if (
        world.model is not player.world.model
        or world.data is player.world.data
        or world._private_policy_owners
        or ((player.pending is None) != before_proposal)
        or type(policy) is not type(player.policy)
        or policy is player.policy
        or policy.state_cmd is player.policy.state_cmd
        or policy.policy_output is player.policy.policy_output
    ):
        raise ValueError("private copied policy and separate owned scratch world required")
    ghost = object.__new__(SharedRoboNaldoPlayer)
    ghost.world, ghost.prefix, ghost.policy = world, player.prefix, policy
    ghost.layout, ghost.torso = world.layouts[player.prefix], player.torso
    ghost.goal = player.goal.copy()
    ghost.boundary = world.control_steps
    ghost.pending = None if player.pending is None else player.pending.copy()
    ghost.failed = False
    ghost.proposals, ghost.commits = player.proposals, player.commits
    ghost.measured_entry_receipt = player.measured_entry_receipt
    world._private_policy_owners[player.prefix] = (
        id(policy),
        id(policy.state_cmd),
        id(policy.policy_output),
    )
    return ghost


class SharedNNForecaster:
    def __init__(
        self,
        player: SharedRoboNaldoPlayer,
        *,
        expected_policy_type: type,
        expected_model_hash: str,
        native_step_budget: int,
    ) -> None:
        import mujoco

        world = player.world
        world._require_current_boundary()
        if (
            type(player.policy) is not expected_policy_type
            or player.failed
            or len(world.layouts) != 1
            or world._torque_layer_owners
            or world.model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER
            or compiled_model_hash(world.model) != expected_model_hash
        ):
            raise ValueError(
                "qualified single-player policy and unmodified shared Euler PD required"
            )
        if (
            type(native_step_budget) is not int
            or not 10 <= native_step_budget <= 600000
            or native_step_budget % 10
        ):
            raise ValueError("explicit bounded whole-control forecast budget required")
        copy_sim_policy(
            player.policy
        )  # Unsupported state is rejected before any forecast allocation.
        self.player = player
        self.model_hash, self.policy_type = expected_model_hash, expected_policy_type
        self.budget = native_step_budget
        self.calls = self.attempted_steps = self.integrated_steps = self.future_nn_calls = 0

    def counters(self) -> dict[str, int]:
        return dict(
            forecast_calls=self.calls,
            attempted_native_steps=self.attempted_steps,
            integrated_native_steps=self.integrated_steps,
            future_nn_calls=self.future_nn_calls,
            maximum_native_steps=self.budget,
        )

    def forecast(
        self,
        *,
        controls: int,
        first_target: np.ndarray | None = None,
        first_action_history: np.ndarray | None = None,
        before_proposal: bool = False,
        reference_plan: MotionTimingPlan
        | MotionSynergyPlan
        | WholeJointReferencePlan
        | None = None,
    ) -> dict[str, Any]:
        import mujoco

        source, donor = self.player, self.player.world
        donor._require_current_boundary()
        if (
            source.failed
            or type(before_proposal) is not bool
            or ((source.pending is None) != before_proposal)
            or source.boundary != donor.control_steps
            or type(source.policy) is not self.policy_type
            or donor._torque_layer_owners
            or compiled_model_hash(donor.model) != self.model_hash
        ):
            raise ValueError("one admitted pending proposal at unchanged actual boundary required")
        if type(controls) is not int or not 1 <= controls <= (128 if before_proposal else 64):
            raise ValueError("bounded64 pending/128 pre-proposal closed-loop horizon required")
        if (reference_plan is not None and not before_proposal) or (
            before_proposal and (first_target is not None or first_action_history is not None)
        ):
            raise ValueError(
                "reference conditioning is pre-proposal only, no first target override"
            )
        charge = controls * 10
        if self.attempted_steps + charge > self.budget:
            raise RuntimeError("closed-loop forecast budget exhausted; no automatic extension")
        self.calls += 1
        self.attempted_steps += (
            charge  # Full requested horizon charged, including rejected/early-failed attempts.
        )
        before = policy_state_hash(source.policy)
        policy = copy_sim_policy(source.policy)
        reference_applied = False
        if reference_plan is not None:
            if type(reference_plan) is MotionSynergyPlan:
                reference_applied = condition_synergy_reference(policy, reference_plan)["applied"]
            elif type(reference_plan) is WholeJointReferencePlan:
                reference_applied = condition_whole_joint_reference(policy, reference_plan)[
                    "applied"
                ]
            elif isinstance(reference_plan, MotionTimingPlan):
                reference_applied = condition_motion_reference(policy, reference_plan)["applied"]
            else:
                raise ValueError("explicit supported data-only reference plan required")
        # A pre-proposal fork has no target/history to rewind. The first NN call
        # will be made below on the owned world's measured body, exactly once.
        parent_target = np.zeros(29, np.float64)
        if not before_proposal:
            assert source.pending is not None
            parent_target = source.pending.copy()
        target = parent_target.copy() if first_target is None else np.asarray(first_target)
        if target.shape != (29,) or target.dtype != np.float64 or not np.isfinite(target).all():
            raise ValueError("finite actual-encoding float64 first target required")
        changed = (
            np.array([], dtype=int) if before_proposal else np.flatnonzero(target != parent_target)
        )
        if len(changed):
            if (
                changed.max() >= 12
                or np.max(abs(target - parent_target)) > 0.120001
                or first_action_history is None
                or first_action_history.shape != (29,)
                or first_action_history.dtype != np.float32
                or not np.isfinite(first_action_history).all()
            ):
                raise ValueError("bounded leg target with exact composite action history required")
            mapping = np.array(import_module(type(policy).__module__).MUJOCO_TO_ISAAC)
            if mapping.shape != (29,) or sorted(mapping.tolist()) != list(range(29)):
                raise ValueError("original canonical action bijection required")
            untouched = np.ones(29, dtype=bool)
            untouched[mapping[changed]] = False
            if not np.array_equal(
                first_action_history[untouched], policy.last_action_il[untouched]
            ):
                raise ValueError("unmodified joint action history must remain original")
            decoded = (
                policy.default_q_mj + policy.action_scale_mj * first_action_history[mapping]
            ).astype(np.float64)
            limits = donor.model.jnt_range[source.layout.joints]
            if (
                not np.array_equal(decoded, target)
                or (target[changed] < limits[changed, 0]).any()
                or (target[changed] > limits[changed, 1]).any()
                or (first_action_history < policy.action_clip_lo_il).any()
                or (first_action_history > policy.action_clip_hi_il).any()
            ):
                raise ValueError("composite history/physical target/action limit mismatch")
        elif first_action_history is not None:
            raise ValueError("zero branch must preserve original history without re-encoding")
        scratch = mujoco.MjData(donor.model)
        mujoco.mj_copyData(scratch, donor.model, donor.data)
        owned = SharedNativeMotorWorld(
            donor.model,
            scratch,
            players=donor.parameters,
            ball_body_name=donor.model.body(donor.ball).name,
        )
        ghost = _fork_pending_player(source, owned, policy, before_proposal=before_proposal)
        if not before_proposal:
            ghost.pending = target.copy()
        records, targets, ticks = [], [], []
        try:
            for step in range(controls):
                if step or before_proposal:
                    target, _ = ghost.propose()
                    self.future_nn_calls += 1
                result = owned.advance({source.prefix: target})
                ghost.commit()
                if step == 0 and len(changed):
                    assert first_action_history is not None
                    policy.last_action_il = first_action_history.copy()
                records.append(result)
                targets.append(target.copy())
                ticks.append(int(policy.time_step))
                if owned.failed:
                    break  # Real physical failure retained, never padded to a fictitious horizon.
        finally:
            self.integrated_steps += owned.native_steps
            donor._require_current_boundary()
            if before != policy_state_hash(source.policy):
                source.failed = True
                raise ValueError("forecast mutated the actual donor policy history")
        arrays = {
            key: np.concatenate([r[key] for r in records])
            for key in ("qpos", "qvel", "torque", "ball_forces")
        }
        return dict(
            **arrays,
            joint_targets=np.stack(targets),
            policy_ticks=np.array(ticks, np.int64),
            donor_control_frame=donor.control_steps,
            controls=len(records),
            requested_controls=controls,
            native_steps=owned.native_steps,
            body_failed=owned.failed,
            donor_software_unchanged=True,
            before_proposal=before_proposal,
            reference_conditioned=reference_applied,
            prediction_is_not_actual_execution=True,
            promotion_authorized=False,
            hardware_authorized=False,
        )
