"""Read completed MuJoCo ground-contact forces without refreshing the solver.

Call immediately after mj_step. Forces belong to that completed interval, not
to a hypothetical forward solve at the newly integrated position. These sums
are normal-force observations, not balance or task-readiness certificates.
"""

import mujoco
import numpy as np


class CompletedGroundSupport:
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        pelvis_body: int,
        left_foot_geoms: frozenset[int],
        right_foot_geoms: frozenset[int],
        ground_geoms: frozenset[int],
    ) -> None:
        groups = (left_foot_geoms, right_foot_geoms, ground_geoms)
        if (
            not isinstance(model, mujoco.MjModel)
            or type(pelvis_body) is not int
            or not 0 < pelvis_body < model.nbody
            or any(
                type(group) is not frozenset
                or not group
                or any(type(g) is not int or not 0 <= g < model.ngeom for g in group)
                for group in groups
            )
            or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
            or any(model.geom_bodyid[g] != 0 for g in ground_geoms)
        ):
            raise ValueError("explicit disjoint feet and static-ground geometry required")
        for geom in left_foot_geoms | right_foot_geoms:
            body = int(model.geom_bodyid[geom])
            while body not in (0, pelvis_body):
                body = int(model.body_parentid[body])
            if body != pelvis_body:
                raise ValueError("foot outside the bound robot subtree")
        self._model = model
        self._feet = (left_foot_geoms, right_foot_geoms)
        self._ground = ground_geoms

    def read(self, data: mujoco.MjData) -> np.ndarray:
        """Return left/right summed normal forces; never call mj_forward."""
        if not isinstance(data, mujoco.MjData) or data.model is not self._model:
            raise ValueError("completed support must use the bound model")
        force = np.zeros(2, dtype=np.float64)
        wrench = np.zeros(6, dtype=np.float64)
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not pair & self._ground:
                continue
            for side, feet in enumerate(self._feet):
                if pair & feet:
                    mujoco.mj_contactForce(self._model, data, index, wrench)
                    if not np.isfinite(wrench).all():
                        raise ValueError("nonfinite completed contact wrench")
                    force[side] += max(0.0, float(wrench[0]))
        if not np.isfinite(force).all():
            raise ValueError("nonfinite completed support sum")
        return force
