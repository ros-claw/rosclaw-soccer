"""Bounded SIM privileged contact teacher on the unchanged shared PD body.

The finite candidate set is a local action-authority study, not a learned skill.
Predictions never count as actual execution. Off/abstain preserves parent path.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.bounded_joint_tasks import LinearJointTask, solve_bounded_joint_tasks
from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
from rosclaw_soccer.rsi.shared_target_forecast import SharedTargetForecaster, compiled_model_hash

LEG_JOINTS = tuple(range(12))


def encode_modified_target(
    policy: Any,
    target: np.ndarray,
    mapping: np.ndarray,
    *,
    modified_indices: tuple[int, ...] = LEG_JOINTS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Quantize changed legs to the original float32 PD/action encoding BEFORE prediction."""
    encoded = policy.last_action_il.copy()
    if (
        not modified_indices
        or len(set(modified_indices)) != len(modified_indices)
        or any(type(i) is not int or not 0 <= i < 12 for i in modified_indices)
    ):
        raise ValueError("explicit changed leg indices required")
    legs = np.asarray(modified_indices)
    il = mapping[legs]
    raw = ((target[legs] - policy.default_q_mj[legs]) / policy.action_scale_mj[legs]).astype(
        np.float32
    )
    if (raw < policy.action_clip_lo_il[il]).any() or (raw > policy.action_clip_hi_il[il]).any():
        return None
    encoded[il] = raw
    quantized = target.copy()
    quantized[legs] = (policy.default_q_mj[legs] + policy.action_scale_mj[legs] * raw).astype(
        np.float64
    )
    return quantized, encoded


def task_candidate(
    layout: Any,
    donor: Any,
    nominal: np.ndarray,
    previous: np.ndarray,
    surfaces: dict[str, Any],
    *,
    contact: bool,
) -> tuple[np.ndarray | None, str]:
    current = donor.qpos[layout.qi].copy()
    delta = nominal - current
    tasks = []
    available_feet = [
        (name, s) for name, s in surfaces.items() if name.endswith("foot") and s["available"]
    ]
    if len(available_feet) != 2:
        return None, "ABSTAIN_NO_TWO_FOOT_SURFACES"
    active = min(available_feet, key=lambda item: item[1]["gap_m"])[0]
    for name, foot in available_feet:
        displacement = foot["jacobian_hg"] @ delta
        if contact and name == active:
            # Contact-conditioned lateral/height alignment, not ankle-center pursuit.
            gap_vector = foot["ball_point"] - foot["world_point"]
            correction = np.array(
                [0.02, np.clip(gap_vector[1], -0.03, 0.03), np.clip(gap_vector[2], -0.015, 0.015)]
            )
            displacement = displacement + correction
        if np.linalg.norm(displacement) > 0.25:
            return None, "ABSTAIN_LOCAL_LINEARIZATION_TOO_LARGE"
        tasks.append(LinearJointTask(name, foot["jacobian_hg"], displacement, 100.0))
    for name, shin in surfaces.items():
        if not name.endswith("shin") or not shin["available"] or not shin["smooth_separated"]:
            continue
        displacement = shin["jacobian_hg"] @ delta
        predicted_linear_gap = shin["gap_m"] + shin["gap_jacobian_hg"] @ delta
        retreat = float(np.clip(0.015 - predicted_linear_gap, 0, 0.04))
        displacement = displacement - shin["normal_toward_ball"] * retreat
        if np.linalg.norm(displacement) > 0.25:
            return None, "ABSTAIN_LOCAL_LINEARIZATION_TOO_LARGE"
        tasks.append(LinearJointTask(name, shin["jacobian_hg"], displacement, 100.0))
    proposal = solve_bounded_joint_tasks(
        current,
        nominal,
        previous,
        layout.model.jnt_range[layout.joints].astype(np.float64),
        tuple(tasks),
        observation_step_cap_rad=0.2,
        command_step_cap_rad=0.1,
        nominal_trust_rad=0.12,
        posture_cost=0.1,
        active_joint_indices=LEG_JOINTS,
    )
    return (None if proposal.target_rad is None else np.array(proposal.target_rad)), proposal.status


class TaskSpaceKickPlayer(SharedRoboNaldoPlayer):
    """Own history commits the actual composite action, never an unexecuted prior."""

    foundation_relative = False
    precontact_sparse = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from importlib import import_module

        self.mapping = np.array(import_module(type(self.policy).__module__).MUJOCO_TO_ISAAC)
        if sorted(self.mapping.tolist()) != list(range(29)):
            raise ValueError("actual original HG-to-Isaac bijection required")
        self.forecaster = SharedTargetForecaster(
            self.world,
            expected_model_hash=compiled_model_hash(self.world.model),
            native_step_budget=10800,
        )
        self.previous_target = self.world.last_applied_target(self.prefix)
        self.executed_history: np.ndarray | None = None
        self.selected_prediction: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.changed_controls = 0
        self.previous_delta = np.zeros(29, np.float64)
        self.nominal_for_commit: np.ndarray | None = None
        self.first_foot_control: int | None = None
        self.selected_records: list[tuple[int, dict[str, Any]]] = []

    def propose(self) -> tuple[np.ndarray, dict[str, Any]]:
        nominal, receipt = super().propose()
        self.nominal_for_commit = nominal.copy()
        self.executed_history = None
        self.selected_prediction = None
        surfaces = measure_ball_surfaces(self.world.model, self.world.data, self.layout)["surfaces"]
        foot_gaps = [
            s["gap_m"] for n, s in surfaces.items() if n.endswith("foot") and s["available"]
        ]
        if (
            self.world.control_steps >= 300
            or not foot_gaps
            or min(foot_gaps) > 0.18
            or (self.precontact_sparse and self.first_foot_control is not None)
        ):
            self.events.append(
                dict(
                    frame=self.world.control_steps,
                    selected="parent_far_or_diagnostic",
                    changed=False,
                )
            )
            return nominal, receipt
        prediction = self.forecaster.forecast({self.prefix: nominal})
        candidates: list[tuple[str, np.ndarray, np.ndarray | None, dict[str, Any]]] = [
            ("parent", nominal, None, prediction)
        ]
        statuses = []
        candidate_function = task_candidate
        if self.foundation_relative:
            from rosclaw_soccer.rsi.foundation_delta_tasks import incremental_task_candidate

            candidate_function = incremental_task_candidate
        for contact in (False, True):
            extra = dict(sparse_admissible=True) if self.precontact_sparse else {}
            candidate, status = candidate_function(
                self.layout,
                self.world.data,
                nominal,
                self.previous_delta if self.foundation_relative else self.previous_target,
                surfaces,
                contact=contact,
                **extra,
            )
            statuses.append(status)
            if candidate is None:
                continue
            modified = (
                tuple(int(i) for i in np.flatnonzero(candidate != nominal))
                if self.precontact_sparse
                else LEG_JOINTS
            )
            if not modified:
                continue
            result = encode_modified_target(
                self.policy, candidate, self.mapping, modified_indices=modified
            )
            if result is None:
                statuses.append("ABSTAIN_ORIGINAL_ACTION_LIMIT")
                continue
            candidate, history = result
            ids = list(modified)
            limits = self.world.model.jnt_range[self.layout.joints]
            if (
                (candidate[ids] < limits[ids, 0]) | (candidate[ids] > limits[ids, 1])
            ).any() or np.max(abs(candidate[ids] - nominal[ids])) > 0.120001:
                statuses.append("ABSTAIN_QUANTIZED_PHYSICAL_LIMIT")
                continue
            predicted = self.forecaster.forecast({self.prefix: candidate})
            candidates.append(
                ("contact" if contact else "clearance", candidate, history, predicted)
            )
        scores = []
        for _name, target, _history, predicted in candidates:
            import mujoco

            scratch = mujoco.MjData(self.world.model)
            scratch.qpos[:] = predicted["qpos"][-1]
            scratch.qvel[:] = predicted["qvel"][-1]
            future = measure_ball_surfaces(self.world.model, scratch, self.layout)["surfaces"]
            shins = [s["gap_m"] for n, s in future.items() if n.endswith("shin") and s["available"]]
            feet = [s["gap_m"] for n, s in future.items() if n.endswith("foot") and s["available"]]
            forces = predicted["ball_forces"][:, 0]
            nonfoot = float(forces[:, 2].max())
            progress = float(
                predicted["qpos"][-1, self.layout.bq] - self.world.data.qpos[self.layout.bq]
            )
            # Fixed local proxy: physical vetoes first; then clearance/contact/progress.
            score = (
                int(not predicted["body_failed"][self.prefix]),
                int(nonfoot <= 1.0),
                -max(nonfoot - 1.0, 0.0),
                2.0 * progress
                - min(feet, default=2.0)
                - 4.0 * max(0.015 - min(shins, default=2.0), 0.0)
                - 0.02 * float(np.linalg.norm(target - nominal)),
            )
            scores.append(score)
        chosen = max(range(len(candidates)), key=lambda i: scores[i])
        name, target, chosen_history, predicted = candidates[chosen]
        changed = not np.array_equal(target, nominal)
        if changed:
            if chosen_history is None:
                raise ValueError("modified target has no composite history")
            self.pending = target.copy()
            self.executed_history = chosen_history.copy()
            self.changed_controls += 1
        self.selected_prediction = predicted
        self.events.append(
            dict(
                frame=self.world.control_steps,
                selected=name,
                changed=changed,
                solver_statuses=statuses,
                scores=scores,
                requested_delta_rad=(target - nominal).tolist(),
                measured_gaps={n: s.get("gap_m") for n, s in surfaces.items()},
                nominal_tracking_error_rad=float(
                    np.max(abs(nominal[:12] - self.world.data.qpos[self.layout.qi[:12]]))
                ),
            )
        )
        return target.copy(), receipt

    def commit(self) -> None:
        if self.pending is None:
            raise ValueError("actual proposed target required before commit")
        target = self.pending.copy()
        super().commit()
        if self.selected_prediction is not None:
            p = self.selected_prediction
            if any(
                not np.array_equal(p[k][-1], actual)
                for k, actual in (
                    ("qpos", self.world.data.qpos),
                    ("qvel", self.world.data.qvel),
                    ("torque", self.world.data.ctrl),
                )
            ):
                self.failed = True
                raise ValueError("chosen forecast and actual committed native state differ")
            self.selected_records.append((self.world.control_steps - 1, p))
        if self.executed_history is not None:
            self.policy.last_action_il = self.executed_history.copy()
            if not np.array_equal(
                (
                    self.policy.default_q_mj
                    + self.policy.action_scale_mj * self.executed_history[self.mapping]
                )[:12].astype(np.float64),
                target[:12],
            ):
                raise ValueError("composite action history does not encode actual executed legs")
        if self.events[-1]["changed"]:
            self.events[-1]["actual_gaps_after"] = {
                n: s.get("gap_m")
                for n, s in measure_ball_surfaces(self.world.model, self.world.data, self.layout)[
                    "surfaces"
                ].items()
            }
        self.previous_target = target
        if self.nominal_for_commit is None:
            raise ValueError("committed target lacks its corresponding foundation target")
        self.previous_delta = target - self.nominal_for_commit
        feet = {self.prefix + side + "_ankle_roll_link" for side in ("left", "right")}
        if self.first_foot_control is None and feet & set(
            self.world.last_ball_contact_bodies[self.prefix]
        ):
            self.first_foot_control = self.world.control_steps - 1
