"""Owned remainder-of-H300 SIM control/body, not an enabled search teacher.

Copies only a COMPLETED current episode boundary, with no future input. Future
steps execute the original Foundation, IK, residual and streaming planner on
their own predicted physics. Original counters are not charged as new steps.
No existing controller calls this experimental helper. Nonzero navigation
uses a distinct born-at-current lease; no old controller clock is fabricated.
Physical nonzero parity must precede any predictive teacher admission.
"""

import copy
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.providers.g1.sonic_branch_state import fork_sonic_tracker
from rosclaw_soccer.providers.g1.sonic_navigation_branch import fork_plain_navigation
from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
from rosclaw_soccer.rsi.born_navigation_recovery import FORWARD_CAPS, BornNavigationRecovery
from rosclaw_soccer.rsi.born_postcontact_release import BORN_RELEASE_ACTIONS
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.causal_impact_replay import CausalImpactFrame, CausalImpactReplay
from rosclaw_soccer.rsi.feedback_branch_control import BODY_SHAPES, fork_motor
from rosclaw_soccer.rsi.foot_heading_evidence import FootHeadingCpuReconstruction
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes

CAUSAL_FIELDS = frozenset(
    {
        "ai",
        "attempted_steps",
        "ball",
        "bg",
        "body",
        "born_release_capture",
        "bq",
        "bv",
        "constant_forward_approach",
        "contacts",
        "data",
        "decoder",
        "early_evidence",
        "early_impact_enabled",
        "exit_evidence",
        "failed",
        "foundation",
        "geometry",
        "ik",
        "initial_reference",
        "integrated_steps",
        "kd",
        "kp",
        "limits",
        "model",
        "navigation",
        "navigation_recovery",
        "next_frame",
        "pelvis",
        "policy",
        "postcontact_exit_enabled",
        "previous",
        "previous_forces",
        "profile",
        "pulse",
        "qi",
        "release_control",
        "reuse_owned_postcontact_updates",
        "teacher",
        "torque_limits",
        "tracker",
        "vi",
        "warnings",
    }
)
IK_FIELDS = frozenset(
    {
        "model",
        "feet",
        "ball",
        "vi",
        "limits",
        "side",
        "contact",
        "next_frame",
        "contact_delta",
        "pending",
        "probe",
        "last_record",
    }
)


def fork_completed_ik(
    ik: FootHeadingCpuReconstruction, *, next_frame: int
) -> FootHeadingCpuReconstruction:
    if (
        type(ik) is not FootHeadingCpuReconstruction
        or set(vars(ik)) != IK_FIELDS
        or ik.pending is not None
        or type(next_frame) is not int
        or not 31 <= next_frame < 300
        or ik.next_frame != next_frame
    ):
        raise ValueError("completed actual IK boundary required, never fabricated pending state")
    result = object.__new__(FootHeadingCpuReconstruction)
    for k, v in vars(ik).items():
        setattr(result, k, v if k == "model" else copy.deepcopy(v))
    return result


def integration_hash(model: Any, data: Any) -> str:
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    if not np.isfinite(state).all():
        raise ValueError("finite complete donor integration state required")
    return hash_bytes(state.tobytes())


class FullTaskBranch:
    """One private predictive remainder, new native-step budget <=2690."""

    def __init__(self, parent: CausalImpactReplay, *, world_hash: str, action_name: str) -> None:
        if (
            type(parent) is not CausalImpactReplay
            or set(vars(parent)) != CAUSAL_FIELDS
            or parent.failed is not False
            or type(parent.next_frame) is not int
            or not 31 <= parent.next_frame < 300
            or type(parent.attempted_steps) is not int
            or type(parent.integrated_steps) is not int
            or parent.attempted_steps != parent.next_frame * 10
            or parent.integrated_steps != parent.attempted_steps
            or parent.teacher is not None
            or parent.release_control is not None
            or parent.early_impact_enabled is not False
            or parent.postcontact_exit_enabled is not False
            or parent.reuse_owned_postcontact_updates is not False
            or type(parent.constant_forward_approach) is not bool
            or type(parent.profile) is not BoundedFirstTouchControl
            or parent.profile != BoundedFirstTouchControl()
            or parent.early_evidence is not None
            or parent.exit_evidence is not None
            or type(action_name) is not str
            or action_name not in ("baseline", *FORWARD_CAPS, *BORN_RELEASE_ACTIONS)
            or type(parent.born_release_capture) is not bool
            or (action_name in BORN_RELEASE_ACTIONS and not parent.born_release_capture)
            or (action_name in FORWARD_CAPS and parent.born_release_capture)
            or parent.navigation_recovery is not None
            or parent.tracker is None
            or parent.tracker._next_frame != parent.next_frame
            or parent.tracker.model is not parent.foundation
            or type(parent.foundation) is not FrozenSonicG1Torch
            or str(parent.foundation.device) != "cpu"
            or not np.array_equal(parent.torque_limits, np.asarray(G1_HARD_TORQUE_LIMITS))
            or parent.ik.model is not parent.model
            or parent.navigation._next_frame != parent.next_frame
            or not np.isclose(parent.data.time, parent.next_frame * 0.02, atol=1e-8, rtol=0)
            or mujoco.get_mjcb_control() is not None
        ):
            raise ValueError("whole completed plain current SIM episode boundary required")
        # Reauthenticate the compiled world/gains without any physics step.
        G1PDTorqueForecaster(
            parent.model,
            expected_world_hash=world_hash,
            kp=parent.kp,
            kd=parent.kd,
            max_forecast_steps=20,
        )
        start = parent.next_frame
        for key, shape in (
            ("previous", (12,)),
            ("previous_forces", (6,)),
            ("torque_limits", (29,)),
            ("kp", (29,)),
            ("kd", (29,)),
            ("limits", (29, 2)),
        ):
            value = getattr(parent, key)
            if (
                type(value) is not np.ndarray
                or value.shape != shape
                or value.dtype != np.float64
                or not np.isfinite(value).all()
            ):
                raise ValueError("complete finite typed current control arrays required")
        if set(parent.body) != set(BODY_SHAPES) or any(
            type(a) is not list
            or len(a) != start
            or any(
                type(v) is not np.ndarray
                or v.shape != (1, *BODY_SHAPES[k])
                or v.dtype != np.float64
                or not np.isfinite(v).all()
                for v in a
            )
            for k, a in parent.body.items()
        ):
            raise ValueError("only whole measured current/past body history required")
        if (
            parent.decoder._active_frame is not None
            or parent.decoder._memory.last_frame != start - 1
            or parent.decoder._policy_hash != parent.policy.get("policy_hash")
        ):
            raise ValueError("same completed original motor and current policy identity required")
        self._parent, self._donor_hash = parent, integration_hash(parent.model, parent.data)
        self.start, self.action_name, self.world_hash = start, action_name, world_hash
        owned = object.__new__(CausalImpactReplay)
        shared = {"model", "foundation", "policy", "profile", "pulse", "initial_reference"}
        special = {"data", "tracker", "decoder", "ik", "navigation", "release_control"}
        for k, v in vars(parent).items():
            if k not in special:
                setattr(owned, k, v if k in shared else copy.deepcopy(v))
        owned.data = mujoco.MjData(parent.model)
        # Preserve cached solver/body epochs exactly. Do NOT mj_forward here.
        mujoco.mj_copyData(owned.data, parent.model, parent.data)
        owned.navigation = fork_plain_navigation(parent.navigation, next_frame=start)
        owned.tracker = fork_sonic_tracker(parent.tracker)
        owned.decoder = fork_motor(parent.decoder, next_frame=start)
        owned.ik = fork_completed_ik(parent.ik, next_frame=start)
        # Never pretend an old full-episode release controller consumed prefix.
        owned.release_control = None
        if action_name in BORN_RELEASE_ACTIONS:
            owned.install_born_release(
                world_hash=world_hash, action_name=BORN_RELEASE_ACTIONS[action_name]
            )
        elif action_name in FORWARD_CAPS:
            owned.install_navigation_recovery(BornNavigationRecovery(start, action_name))
        # The prefix was already integrated by the donor, not by this branch.
        # Inherited whole-episode review intentionally cannot pass these counts.
        owned.attempted_steps = owned.integrated_steps = 0
        self._owned = owned
        self._initial_planner_calls = owned.navigation.backend.planner_calls
        self._finished = False
        if integration_hash(parent.model, owned.data) != self._donor_hash:
            raise ValueError("whole copied donor integration state differs")

    def step(self) -> CausalImpactFrame:
        if (
            self._finished
            or self._owned.failed
            or self._owned.next_frame >= 300
            or self._owned.attempted_steps + 10 > (300 - self.start) * 10
        ):
            raise ValueError("once-only next full-task prediction step within fixed H300 required")
        return self._owned.step(self._owned.next_frame)

    def review(self) -> dict[str, Any]:
        actor = self._owned
        steps = (300 - self.start) * 10
        if (
            self._finished
            or actor.failed
            or actor.next_frame != 300
            or actor.attempted_steps != steps
            or actor.integrated_steps != steps
            or integration_hash(self._parent.model, self._parent.data) != self._donor_hash
            or self._parent.next_frame != self.start
        ):
            raise ValueError("whole remaining prediction and unchanged live donor required")
        self._finished = True
        return dict(
            schema="soccer.rsi.full_task_branch_review.v1",
            start_frame=self.start,
            end_exclusive=300,
            borrowed_prefix_frames=self.start,
            donor_integration_hash=self._donor_hash,
            action_name=self.action_name,
            born_navigation_recovery=None
            if actor.navigation_recovery is None
            else actor.navigation_recovery.contract(),
            born_postcontact_release=None
            if actor.release_control is None
            else actor.release_control.review(),
            new_native_attempted_steps=actor.attempted_steps,
            new_native_integrated_steps=actor.integrated_steps,
            recomputed_control_frames=300 - self.start,
            recomputed_foundation_encoder_decoder_pairs=300 - self.start,
            new_planner_onnx_calls=actor.navigation.backend.planner_calls
            - self._initial_planner_calls,
            future_observations_from_own_simulation=True,
            recorded_future_targets_or_states_consumed=False,
            prefix_steps_counted_as_new=False,
            new_optimizer_updates=0,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
