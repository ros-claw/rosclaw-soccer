"""Opt-in process-local MuJoCo-Warp 3.5 contact-force decoding experiment.

The constraint kernel allocates each pyramid edge independently. Never assume
that efc_address[contact, 0] starts a contiguous block. Installed files stay
unchanged. Install before simulator allocation; this is not qualification.
"""

import warp as wp
from mujoco_warp._src.types import ConeType, vec5


@wp.func
def addressed_contact_force(
    opt_cone: int,
    contact_frame_in: wp.array(dtype=wp.mat33),
    contact_friction_in: wp.array(dtype=vec5),
    contact_dim_in: wp.array(dtype=int),
    contact_efc_address_in: wp.array2d(dtype=int),
    efc_force_in: wp.array2d(dtype=float),
    njmax_in: int,
    nacon_in: wp.array(dtype=int),
    worldid: int,
    contact_id: int,
    to_world_frame: bool,
) -> wp.spatial_vector:
    force = wp.spatial_vector()
    if contact_id < 0 or contact_id >= nacon_in[0]:
        return force
    condim = contact_dim_in[contact_id]
    if opt_cone == ConeType.PYRAMIDAL and condim > 1:
        for i in range(condim - 1):
            a = contact_efc_address_in[contact_id, 2 * i]
            b = contact_efc_address_in[contact_id, 2 * i + 1]
            f1 = 0.0
            f2 = 0.0
            if a >= 0 and a < njmax_in:
                f1 = efc_force_in[worldid, a]
            if b >= 0 and b < njmax_in:
                f2 = efc_force_in[worldid, b]
            force[0] += f1 + f2
            force[i + 1] = (f1 - f2) * contact_friction_in[contact_id][i]
    else:
        for i in range(condim):
            a = contact_efc_address_in[contact_id, i]
            if a >= 0 and a < njmax_in:
                force[i] = efc_force_in[worldid, a]
    if to_world_frame:
        force = wp.spatial_vector(
            wp.spatial_top(force) @ contact_frame_in[contact_id],
            wp.spatial_bottom(force) @ contact_frame_in[contact_id],
        )
    return force


def install():
    """Explicit diagnostic override; no import-time library mutation."""
    import mujoco_warp
    from mujoco_warp._src import support

    if mujoco_warp.__version__ != "3.5.0":
        raise ValueError("addressed decoder experiment requires qualified MuJoCo-Warp 3.5.0")
    support.contact_force_fn = addressed_contact_force
