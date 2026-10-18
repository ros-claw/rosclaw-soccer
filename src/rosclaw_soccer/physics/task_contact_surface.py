"""Experimental mesh/object contacts separated from locomotion support contacts.

Copy a caller-owned MJCF spec. Reuse the existing visual mesh at its original
pose/scale, with an explicit task-object pair; exclude automatic body-pair
contacts only for that same task-object/surface-body combination. Ground,
self-contact masks, inertias, joint/controller parameters and visuals remain.
No stepping, rollout qualification, robot transport or policy activation.
"""

from typing import Any

import mujoco
import numpy as np


def with_mesh_task_contacts(
    specification: mujoco.MjSpec, *, task_geom_name: str, surface_body_names: tuple[str, ...]
) -> tuple[mujoco.MjSpec, dict[str, Any]]:
    """Return an owned candidate spec and a construction receipt, NOT admission.

    Deliberately refuses ambiguous surfaces, existing task pairs/exclusions and
    mixed solver settings instead of inventing new contact material parameters.
    MuJoCo mesh collision is convex-hull collision, not exact concave-surface CCD.
    """
    if not isinstance(specification, mujoco.MjSpec):
        raise TypeError("offline MuJoCo specification required")
    if (
        type(task_geom_name) is not str
        or not task_geom_name
        or type(surface_body_names) is not tuple
        or not 1 <= len(surface_body_names) <= 16
        or any(type(name) is not str or not name or len(name) > 128 for name in surface_body_names)
        or len(set(surface_body_names)) != len(surface_body_names)
    ):
        raise ValueError("explicit distinct task geometry and surface body names required")
    candidate = specification.copy()
    before = candidate.compile()
    task = mujoco.mj_name2id(before, mujoco.mjtObj.mjOBJ_GEOM, task_geom_name)
    if task < 0:
        raise ValueError("task geometry missing")
    task_body = int(before.geom_bodyid[task])
    if task_body == 0:
        raise ValueError("task object must belong to a movable non-world body")
    joint = int(before.body_jntadr[task_body])
    if joint < 0 or before.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("explicit free task object required")
    task_body_name = mujoco.mj_id2name(before, mujoco.mjtObj.mjOBJ_BODY, task_body)
    if task_body_name is None:
        raise ValueError("named task body required")
    named_geoms = list(candidate.geoms)
    if len(named_geoms) != before.ngeom:
        raise ValueError("one-to-one original specification geometry order required")
    # Verify the spec/model ordering before assigning names to anonymous meshes.
    for index, geom in enumerate(named_geoms):
        compiled_name = mujoco.mj_id2name(before, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if geom.name != compiled_name:
            raise ValueError("specification geometry order differs from compiled model")
    rows = []
    for ordinal, name in enumerate(surface_body_names):
        body = mujoco.mj_name2id(before, mujoco.mjtObj.mjOBJ_BODY, name)
        if body <= 0 or body == task_body:
            raise ValueError("distinct existing non-world surface bodies required")
        mesh_ids = np.flatnonzero(
            (before.geom_bodyid == body)
            & (before.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
            & (before.geom_contype == 0)
            & (before.geom_conaffinity == 0)
        )
        if len(mesh_ids) != 1:
            raise ValueError("one original visual-only mesh per surface body required")
        mesh = int(mesh_ids[0])
        body_geom_ids = np.flatnonzero(before.geom_bodyid == body)
        for pair in range(before.npair):
            ids = (int(before.pair_geom1[pair]), int(before.pair_geom2[pair]))
            if task in ids and any(i in body_geom_ids for i in ids if i != task):
                raise ValueError("task/surface pair already specified")
        # Existing exclusions are decoded by their source names, not signature math.
        if any(
            {exclude.bodyname1, exclude.bodyname2} == {name, task_body_name}
            for exclude in candidate.excludes
        ):
            raise ValueError("task/surface body exclusion already specified")
        for key in ("geom_solref", "geom_solimp"):
            if not np.array_equal(getattr(before, key)[mesh], getattr(before, key)[task]):
                raise ValueError("matching explicit task/surface solver settings required")
        if (
            before.geom_margin[mesh] != 0
            or before.geom_margin[task] != 0
            or before.geom_gap[mesh] != 0
            or before.geom_gap[task] != 0
            or before.geom_priority[mesh] != before.geom_priority[task]
        ):
            raise ValueError("zero inflation/gap and equal material priority required")
        material = np.maximum(before.geom_friction[mesh], before.geom_friction[task])
        friction = [
            float(material[0]),
            float(material[0]),
            float(material[1]),
            float(material[2]),
            float(material[2]),
        ]
        mesh_name = named_geoms[mesh].name
        if not mesh_name:
            mesh_name = f"task_contact_surface_{ordinal}"
            if any(g.name == mesh_name for g in named_geoms):
                raise ValueError("reserved task surface name already used")
            named_geoms[mesh].name = mesh_name
        pair_name = f"task_contact_pair_{ordinal}"
        exclude_name = f"task_contact_exclude_{ordinal}"
        if any(p.name == pair_name for p in candidate.pairs) or any(
            e.name == exclude_name for e in candidate.excludes
        ):
            raise ValueError("task contact profile already installed")
        candidate.add_exclude(name=exclude_name, bodyname1=name, bodyname2=task_body_name)
        candidate.add_pair(
            name=pair_name,
            geomname1=mesh_name,
            geomname2=task_geom_name,
            condim=int(max(before.geom_condim[mesh], before.geom_condim[task])),
            solref=before.geom_solref[mesh],
            solimp=before.geom_solimp[mesh],
            friction=friction,
            margin=0.0,
            gap=0.0,
        )
        rows.append(
            dict(
                surface_body_name=name,
                source_mesh_geom_id=mesh,
                source_mesh_geom_name=mesh_name,
                pair_name=pair_name,
                exclude_name=exclude_name,
                task_body_name=task_body_name,
                friction_5d=friction,
            )
        )
    after = candidate.compile()
    if (before.ngeom, before.nbody, before.nq, before.nv, before.nu) != (
        after.ngeom,
        after.nbody,
        after.nq,
        after.nv,
        after.nu,
    ):
        raise ValueError("existing geometry/body/actuation dimensions must remain unchanged")
    checked = []
    for key in (
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "jnt_pos",
        "jnt_axis",
        "jnt_range",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "actuator_trnid",
        "actuator_gear",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_contype",
        "geom_conaffinity",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_margin",
        "geom_gap",
        "geom_rgba",
    ):
        if not np.array_equal(getattr(before, key), getattr(after, key)):
            raise ValueError(f"unrelated compiled physical parameter changed: {key}")
        checked.append(key)
    if after.npair - before.npair != len(rows) or after.nexclude - before.nexclude != len(rows):
        raise ValueError("exact declared task pairs/exclusions required")
    return candidate, dict(
        schema="soccer.physics.mesh_task_contact_surface.v1",
        task_geom_name=task_geom_name,
        surfaces=rows,
        geometry_count_unchanged=True,
        compiled_numeric_arrays_unchanged=checked,
        source_specification_modified=False,
        contact_geometry_rescaled=False,
        mesh_collision_uses_convex_hull=True,
        ground_motion_parity_still_required=True,
        new_physics_steps=0,
        physics_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
