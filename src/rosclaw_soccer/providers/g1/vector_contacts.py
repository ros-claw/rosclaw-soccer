"""Optional SIM_ONLY positive-force ball contact sensing for vector G1 courses.

MuJoCo/Warp/Torch remain lazy dependencies. Sensor outputs never count as a
completed skill or authorize policy promotion. Geometry is not modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.vector_motor import G1VectorMotorBatch, G1VectorMotorConfig

if TYPE_CHECKING:
    from rosclaw_soccer.world.field import G1TrainingGoalSpec

CONTACT_CHANNELS = ("left_foot", "right_foot", "other_robot")


def robot_contact_geometry_labels(model: Any) -> NDArray[np.int32]:
    """Return immutable labels 0=environment/ball, 1/2=feet, 3=other robot.

    Descendant membership matches the shared-world foot contract. No distance
    heuristic, mesh expansion, or ankle/shin reclassification is performed.
    """
    parents = np.asarray(model.body_parentid)
    bodies = np.asarray(model.geom_bodyid)
    if (
        parents.ndim != 1
        or bodies.ndim != 1
        or parents.dtype.kind not in "iu"
        or bodies.dtype.kind not in "iu"
        or not len(parents)
        or not len(bodies)
        or parents[0] != 0
        or np.any(parents < 0)
        or np.any(parents >= len(parents))
        or np.any(bodies < 0)
        or np.any(bodies >= len(parents))
    ):
        raise ValueError("finite compiled G1 body/geometry tree required")
    try:
        pelvis = int(model.body("pelvis").id)
        left = int(model.body("left_ankle_roll_link").id)
        right = int(model.body("right_ankle_roll_link").id)
        ball = int(model.geom("ball_geom").id)
    except (KeyError, AttributeError, TypeError, ValueError) as error:
        raise ValueError("named G1 feet, pelvis and ball geometry required") from error
    if (
        len({pelvis, left, right}) != 3
        or any(not 0 < body < len(parents) for body in (pelvis, left, right))
        or not 0 <= ball < len(bodies)
    ):
        raise ValueError("distinct compiled G1 body identities required")

    def ancestors(body: int) -> set[int]:
        result: set[int] = set()
        while body:
            if body in result:
                raise ValueError("cyclic compiled body tree")
            result.add(body)
            body = int(parents[body])
        return result

    lineage = [ancestors(body) for body in range(len(parents))]
    if (
        pelvis not in lineage[left]
        or pelvis not in lineage[right]
        or left in lineage[right]
        or right in lineage[left]
        or pelvis in lineage[int(bodies[ball])]
    ):
        raise ValueError("feet must be separate robot branches and ball must be independent")
    labels = np.zeros(len(bodies), dtype=np.int32)
    for geom, body in enumerate(bodies):
        membership = lineage[int(body)]
        labels[geom] = (
            1
            if left in membership
            else 2
            if right in membership
            else 3
            if pelvis in membership
            else 0
        )
    if not np.any(labels == 1) or not np.any(labels == 2):
        raise ValueError("both feet require compiled geometry")
    return np.frombuffer(labels.tobytes(), dtype=np.int32)


class G1ContactVectorMotorBatch(G1VectorMotorBatch):
    """Read solved normal forces after every existing physics substep.

    Each control call returns cloned ``ball_robot_contact_peak_n`` and
    ``ball_robot_contact_samples`` arrays with shape (worlds, 3), in
    CONTACT_CHANNELS order. Samples count contact constraints across substeps,
    not distinct touches. PD, torque guards and physical steps remain owned
    by G1VectorMotorBatch. A prepared graph is required; no implicit reset.
    """

    def __init__(
        self,
        asset_root: Path,
        config: G1VectorMotorConfig | None = None,
        *,
        goal_spec: G1TrainingGoalSpec | None = None,
    ) -> None:
        super().__init__(asset_root, config, goal_spec=goal_spec)
        import warp as wp

        from rosclaw_soccer.providers.g1._vector_contact_kernel import accumulate_ball_contacts

        self._contact_kernel = accumulate_ball_contacts
        self._contact_labels: Any = wp.array(
            robot_contact_geometry_labels(self.cpu_model), dtype=int, device=self.config.device
        )
        shape = (self.config.environment_count, len(CONTACT_CHANNELS))
        self._contact_peak = wp.zeros(shape, dtype=float, device=self.config.device)
        self._contact_samples = wp.zeros(shape, dtype=int, device=self.config.device)
        self._contact_invalid = wp.zeros(1, dtype=int, device=self.config.device)
        capacity = self._data.contact.geom.shape[0]
        self._contact_ids: Any = wp.array(
            np.arange(capacity, dtype=np.int32), dtype=int, device=self.config.device
        )
        self._contact_force: Any = wp.zeros(
            capacity, dtype=cast(Any, wp.spatial_vector), device=self.config.device
        )
        self._ball_contact_geom = self.cpu_model.geom("ball_geom").id

    def _observe_contacts(self) -> None:
        import warp as wp

        data = self._data
        # Public backend API; never infer force from contact distance.
        self._mjw.contact_force(self._model, data, self._contact_ids, False, self._contact_force)
        wp.launch(
            self._contact_kernel,
            dim=data.contact.geom.shape[0],
            inputs=[
                data.nacon,
                data.contact.worldid,
                data.contact.geom,
                self._contact_force,
                self._contact_labels,
                self._ball_contact_geom,
                self.config.environment_count,
                self.cpu_model.ngeom,
                self._contact_peak,
                self._contact_samples,
                self._contact_invalid,
            ],
            device=self.config.device,
        )

    def prepare_physics_graph(self) -> None:
        super().prepare_physics_graph()
        import warp as wp

        stream = wp.Stream(self.config.device)
        with wp.ScopedStream(stream):
            self._observe_contacts()
            wp.synchronize_stream(stream)
            with wp.ScopedCapture(stream=stream) as capture:
                self._mjw.step(self._model, self._data)
                self._observe_contacts()
        self._physics_graph = capture.graph

    def _step(self, torque: Any, pd: tuple[Any, Any, Any] | None) -> dict[str, Any]:
        if self._physics_graph is None:
            raise RuntimeError("contact sensing requires a prepared physics graph")
        import warp as wp

        peak = wp.to_torch(self._contact_peak)
        samples = wp.to_torch(self._contact_samples)
        invalid = wp.to_torch(self._contact_invalid)
        peak.zero_()
        samples.zero_()
        invalid.zero_()
        state = super()._step(torque, pd)
        if not bool(
            self._torch.isfinite(peak).all()
            and (peak >= 0).all()
            and (samples >= 0).all()
            and (invalid == 0).all()
        ):
            self._ready = False
            self._ctrl.zero_()
            raise FloatingPointError("invalid solved contact observation; failed state retained")
        return {
            **state,
            "ball_robot_contact_peak_n": peak.clone(),
            "ball_robot_contact_samples": samples.clone(),
        }
