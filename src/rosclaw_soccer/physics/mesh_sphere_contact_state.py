"""Current convex mesh/sphere geometry and local differential, never a force.

Caller owns the hull/model binding and fresh kinematics/comPos state. No
integration, observation refresh, model mutation or control is performed.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.physics.convex_surface_distance import ConvexSurfaceDistance


@dataclass(frozen=True)
class MeshSphereContactState:
    gap_m: float
    closest_mesh_point_world_m: tuple[float, float, float]
    outward_normal_world: tuple[float, float, float]
    relative_point_velocity_world_m_s: tuple[float, float, float]
    gap_rate_m_s: float
    gap_jacobian: tuple[float, ...]


def measure_mesh_sphere_contact_state(
    model: Any,
    data: Any,
    mesh_geom: int,
    sphere_geom: int,
    hull: ConvexSurfaceDistance,
) -> MeshSphereContactState:
    """Signed surface gap and its local qvel differential at the closest point.

    The sphere's center velocity, not its surface spin, defines gap rate. Mesh
    rotation contributes through the Jacobian of the actual surface point, not
    the body origin. Interior medial-axis ties are not differentiable; this is
    not a guarantee of future contact, impulse, friction or collision safety.
    """
    import mujoco

    if (
        type(mesh_geom) is not int
        or type(sphere_geom) is not int
        or not 0 <= mesh_geom < model.ngeom
        or not 0 <= sphere_geom < model.ngeom
        or model.geom_type[mesh_geom] != mujoco.mjtGeom.mjGEOM_MESH
        or model.geom_type[sphere_geom] != mujoco.mjtGeom.mjGEOM_SPHERE
        or not isinstance(hull, ConvexSurfaceDistance)
    ):
        raise ValueError("explicit compiled mesh, sphere and matching convex hull required")
    center = np.asarray(data.geom_xpos[sphere_geom], dtype=np.float64)
    rotation = data.geom_xmat[mesh_geom].reshape(3, 3)
    signed, local = hull.nearest(rotation.T @ (center - data.geom_xpos[mesh_geom]))
    point = data.geom_xpos[mesh_geom] + rotation @ local
    displacement = center - point
    distance = float(np.linalg.norm(displacement))
    if distance <= 1e-10:
        raise ValueError("nondegenerate closest-point normal required")
    normal = displacement / distance * (1 if signed >= 0 else -1)
    relative_jacobian = np.zeros((3, model.nv), dtype=np.float64)
    mesh_jacobian = np.zeros_like(relative_jacobian)
    mujoco.mj_jac(model, data, relative_jacobian, None, center, int(model.geom_bodyid[sphere_geom]))
    mujoco.mj_jac(model, data, mesh_jacobian, None, point, int(model.geom_bodyid[mesh_geom]))
    relative_jacobian -= mesh_jacobian
    velocity = relative_jacobian @ data.qvel
    gradient = normal @ relative_jacobian
    gap = signed - float(model.geom_size[sphere_geom, 0])
    if not all(np.isfinite(v).all() for v in (point, normal, velocity, gradient, gap)):
        raise ValueError("finite current geometry, velocity and local differential required")
    return MeshSphereContactState(
        gap_m=gap,
        closest_mesh_point_world_m=(float(point[0]), float(point[1]), float(point[2])),
        outward_normal_world=(float(normal[0]), float(normal[1]), float(normal[2])),
        relative_point_velocity_world_m_s=(
            float(velocity[0]),
            float(velocity[1]),
            float(velocity[2]),
        ),
        gap_rate_m_s=float(gradient @ data.qvel),
        gap_jacobian=tuple(float(v) for v in gradient),
    )
