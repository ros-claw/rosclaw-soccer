"""Native MuJoCo predictive-sampling teacher; no learned or hardware policy.

Design reference: google-deepmind/mujoco_mpc predictive sampling. This adapter
uses MuJoCo's native batched rollout thread pool, unchanged 2ms dynamics, and
explicit HG-to-compiled-actuator mapping. It never steps or resets the supplied
evaluation body. Only independent planning copies are integrated here.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PredictiveTrackingBudget:
    samples: int = 64
    horizon_controls: int = 12
    improvements: int = 2
    threads: int = 8
    seed: int = 1701
    sigma_rad: float = 0.1
    max_delta_rad: float = 0.3

    def __post_init__(self) -> None:
        for value, low, high in (
            (self.samples, 8, 128),
            (self.horizon_controls, 3, 25),
            (self.improvements, 1, 3),
            (self.threads, 1, 16),
            (self.seed, 0, 2**31 - 1),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("bounded integer predictive-control allocation required")
        if not (
            np.isfinite(self.sigma_rad)
            and np.isfinite(self.max_delta_rad)
            and 0 < self.sigma_rad <= self.max_delta_rad <= 0.5
        ):
            raise ValueError("bounded finite joint-target proposals required")


def interpolate_residuals(knots: npt.NDArray[np.float64], horizon: int) -> npt.NDArray[np.float64]:
    """Three linear knots, batch x knot x HG joint; no physical state edits."""
    if (
        knots.ndim != 3
        or knots.shape[1:] != (3, 29)
        or not np.isfinite(knots).all()
        or type(horizon) is not int
        or not 3 <= horizon <= 25
    ):
        raise ValueError("finite three-knot HG29 proposals and bounded horizon required")
    times = np.linspace(0.0, 2.0, horizon)
    lo = np.minimum(times.astype(np.int64), 1)
    fraction = times - lo
    return (
        knots[:, lo, :] * (1.0 - fraction[None, :, None])
        + knots[:, lo + 1, :] * fraction[None, :, None]
    )


def actuator_controls(
    targets: npt.NDArray[np.float64], actuator_ids_hg: tuple[int, ...], decimation: int = 10
) -> npt.NDArray[np.float64]:
    """Encode HG position targets into compiled actuator order, not joint order."""
    if (
        targets.ndim != 3
        or targets.shape[-1] != 29
        or not np.isfinite(targets).all()
        or len(actuator_ids_hg) != 29
        or sorted(actuator_ids_hg) != list(range(29))
        or type(decimation) is not int
        or decimation != 10
    ):
        raise ValueError("finite HG29 targets and an exact actuator permutation required")
    controls = np.empty_like(targets)
    controls[:, :, actuator_ids_hg] = targets
    return np.repeat(controls, decimation, axis=1)


class NativePredictiveTracker:
    """Privileged offline teacher with an auditable, bounded planning budget.

    References and costs are training-only ground truth. A model checkpoint or
    a football qualification does not follow from a successful planning call.
    Every call preserves the external main MjData byte-for-byte.
    """

    def __init__(
        self,
        model: Any,
        *,
        joint_names_hg: tuple[str, ...],
        actuator_ids_hg: tuple[int, ...],
        root_joint_name: str,
        budget: PredictiveTrackingBudget,
    ) -> None:
        import mujoco
        from mujoco.rollout import Rollout

        if model.nu != 29 or model.opt.timestep != 0.002 or len(joint_names_hg) != 29:
            raise ValueError("qualified compiled 29-actuator 2ms model required")
        # Check the map against actual actuator transmissions before allocation.
        ids = []
        qids = []
        for joint, actuator in zip(joint_names_hg, actuator_ids_hg, strict=True):
            jid = model.joint(joint).id
            if (
                model.jnt_type[jid] != int(mujoco.mjtJoint.mjJNT_HINGE)
                or model.actuator_trntype[actuator] != int(mujoco.mjtTrn.mjTRN_JOINT)
                or model.actuator_trnid[actuator, 0] != jid
            ):
                raise ValueError("actuator map disagrees with actual compiled hinge transmission")
            ids.append(jid)
            qids.append(int(model.jnt_qposadr[jid]))
        actuator_controls(np.zeros((1, 3, 29)), actuator_ids_hg)
        root = model.joint(root_joint_name).id
        if model.jnt_type[root] != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("actual robot free joint required")
        if not np.all(model.jnt_limited[ids]):
            raise ValueError("all proposed joint targets must have actual finite joint bounds")
        self.model = model
        self.budget = budget
        self.actuator_ids = actuator_ids_hg
        self.qids = np.asarray(qids)
        self.root_qadr = int(model.jnt_qposadr[root])
        self.bounds = np.asarray(model.jnt_range[ids]).copy()
        if not np.isfinite(self.bounds).all() or np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("finite ordered physical joint limits required")
        self.rng = np.random.default_rng(budget.seed)
        self.rollout = Rollout(nthread=budget.threads)
        self.datas = [mujoco.MjData(model) for _ in range(budget.threads)]
        self.warm = np.zeros((3, 29))
        self.requested_planning_substeps = 0
        self.observed_advancing_substeps = 0
        self.planning_updates = 0

    def close(self) -> None:
        self.rollout.close()

    def plan(
        self,
        data: Any,
        reference_joint: npt.NDArray[np.float64],
        reference_root_position: npt.NDArray[np.float64],
        reference_root_quaternion: npt.NDArray[np.float64],
    ) -> dict[str, Any]:
        import mujoco

        b = self.budget
        h = b.horizon_controls
        if (
            reference_joint.shape != (h, 29)
            or reference_root_position.shape != (h, 3)
            or reference_root_quaternion.shape != (h, 4)
            or not all(
                np.isfinite(v).all()
                for v in (reference_joint, reference_root_position, reference_root_quaternion)
            )
            or not np.allclose(np.linalg.norm(reference_root_quaternion, axis=1), 1.0, atol=1e-5)
        ):
            raise ValueError("finite, explicitly aligned native-reference horizon required")
        initial = np.empty(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(self.model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        # Rollout's initial vector excludes USER state. Freeze all external
        # forces, mocap transforms, equality flags and user data explicitly in
        # its control stream; replacing only CTRL must not reset those fields.
        user_spec = mujoco.mjtState.mjSTATE_USER
        user_state = np.empty(mujoco.mj_stateSize(self.model, user_spec))
        mujoco.mj_getState(self.model, data, user_state, user_spec)
        if not np.array_equal(user_state[: self.model.nu], data.ctrl):
            raise ValueError("qualified MuJoCo USER layout must begin with actuator controls")
        warmstart = data.qacc_warmstart.copy()
        initial_copy = initial.copy()
        best_cost = float("inf")
        best_knots = self.warm.copy()
        nominal_cost = None
        iteration_rows = []
        mean = self.warm.copy()
        sigma = np.full((3, 29), b.sigma_rad)
        for iteration in range(b.improvements):
            knots = mean[None, :, :] + self.rng.normal(size=(b.samples, 3, 29)) * sigma[None, :, :]
            knots = np.clip(knots, -b.max_delta_rad, b.max_delta_rad)
            knots[0] = 0.0  # Always retain the unmodified reference, not only noise.
            knots[1] = best_knots
            residual = interpolate_residuals(knots, h)
            targets = np.clip(
                reference_joint[None, :, :] + residual, self.bounds[:, 0], self.bounds[:, 1]
            )
            control = actuator_controls(targets, self.actuator_ids)
            user_control = np.broadcast_to(
                user_state, (b.samples, control.shape[1], len(user_state))
            ).copy()
            user_control[:, :, : self.model.nu] = control
            states, _ = self.rollout.rollout(
                self.model,
                self.datas,
                initial[None, :],
                user_control,
                control_spec=int(user_spec),
                initial_warmstart=warmstart[None, :],
            )
            requested = b.samples * h * 10
            self.requested_planning_substeps += requested
            times = np.concatenate((np.full((b.samples, 1), data.time), states[:, :, 0]), axis=1)
            dt = np.diff(times, axis=1)
            valid = np.isfinite(states).all(axis=(1, 2)) & np.all(np.abs(dt - 0.002) < 1e-9, axis=1)
            self.observed_advancing_substeps += int(np.count_nonzero(np.abs(dt - 0.002) < 1e-9))
            samples = states[:, 9::10, 1 : 1 + self.model.nq]
            root = samples[:, :, self.root_qadr : self.root_qadr + 7]
            joint_error = np.mean(
                (samples[:, :, self.qids] - reference_joint[None, :, :]) ** 2, axis=(1, 2)
            )
            position_error = np.mean(
                np.sum((root[:, :, :3] - reference_root_position[None, :, :]) ** 2, axis=2), axis=1
            )
            quats = root[:, :, 3:7]
            dot = np.sum(quats * reference_root_quaternion[None, :, :], axis=2)
            orientation_error = np.mean(1.0 - np.clip(dot * dot, 0.0, 1.0), axis=1)
            costs = (
                joint_error
                + 10.0 * position_error
                + 20.0 * orientation_error
                + 0.01 * np.mean(residual * residual, axis=(1, 2))
            )
            costs[~valid] = np.inf
            if iteration == 0:
                nominal_cost = float(costs[0]) if np.isfinite(costs[0]) else None
            if not np.isfinite(costs).any():
                raise ValueError(
                    "all physical planning trajectories diverged; no valid teacher action"
                )
            winner = int(np.argmin(costs))
            if costs[winner] < best_cost:
                best_cost = float(costs[winner])
                best_knots = knots[winner].copy()
            finite = np.flatnonzero(np.isfinite(costs))
            elite = finite[np.argsort(costs[finite])[: max(2, min(8, len(finite)))]]
            mean = knots[elite].mean(axis=0)
            sigma = np.maximum(0.02, knots[elite].std(axis=0))
            self.planning_updates += 1
            iteration_rows.append(
                dict(best_cost=float(costs[winner]), valid_samples=int(valid.sum()))
            )
        new = np.empty_like(initial)
        mujoco.mj_getState(self.model, data, new, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        user_after = np.empty_like(user_state)
        mujoco.mj_getState(self.model, data, user_after, user_spec)
        if (
            not np.array_equal(initial_copy, new)
            or not np.array_equal(warmstart, data.qacc_warmstart)
            or not np.array_equal(user_state, user_after)
        ):
            raise ValueError("planning modified the external evaluation body")
        planned = interpolate_residuals(best_knots[None, :, :], h)[0]
        shifted = np.concatenate((planned[1:], planned[-1:]), axis=0)
        self.warm = np.stack([shifted[i] for i in (0, (h - 1) // 2, h - 1)])
        target = np.clip(reference_joint[0] + planned[0], self.bounds[:, 0], self.bounds[:, 1])
        return dict(
            target_hg=target,
            knots=best_knots,
            nominal_cost=nominal_cost,
            best_cost=best_cost,
            iterations=iteration_rows,
            requested_planning_substeps=b.samples * h * 10 * b.improvements,
            new_neural_calls=0,
            new_optimizer_steps=0,
            hardware_authorized=False,
        )
