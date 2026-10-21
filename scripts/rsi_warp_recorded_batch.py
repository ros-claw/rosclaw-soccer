"""Direct SIM recorded-target batch; experimental diagnostics, not qualification.

No MDP resets, learned actor, hardware transport or installed-package edits.
Uses the original compiled model and the already tested addressed force decoder.
"""

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from scripts.rsi_mjwarp_addressed_contact_force import addressed_contact_force, install


@wp.kernel
def _forces(
    model: mjw.Model,
    data: mjw.Data,
    ball: int,
    roles: wp.array(dtype=int),
    out: wp.array2d(dtype=float),
):
    ci = wp.tid()
    if ci >= data.nacon[0]:
        return
    g = data.contact.geom[ci]
    if g[0] < 0 or g[1] < 0:
        return
    b0 = model.geom_bodyid[g[0]]
    b1 = model.geom_bodyid[g[1]]
    other = wp.int32(-1)
    if b0 == ball:
        other = b1
    elif b1 == ball:
        other = b0
    if other < 0:
        return
    role = roles[other]
    if role < 0:
        return
    world = data.contact.worldid[ci]
    force = addressed_contact_force(
        model.opt.cone,
        data.contact.frame,
        data.contact.friction,
        data.contact.dim,
        data.contact.efc_address,
        data.efc.force,
        data.njmax,
        data.nacon,
        world,
        ci,
        False,
    )
    magnitude = wp.sqrt(force[0] * force[0] + force[1] * force[1] + force[2] * force[2])
    wp.atomic_max(out, world, role, magnitude)


class RecordedWarpBatch:
    def __init__(self, model, *, worlds, ball, roles):
        if (
            mujoco.__version__ != "3.5.0"
            or mjw.__version__ != "3.5.0"
            or wp.__version__ != "1.12.0"
        ):
            raise ValueError("pinned native3.5/MJWarp3.5/Warp1.12 diagnostic only")
        if (
            type(worlds) is not int
            or not 1 <= worlds <= 4096
            or roles.shape != (model.nbody,)
            or not np.isin(roles, [-1, 0, 1, 2]).all()
            or not 0 < ball < model.nbody
        ):
            raise ValueError("bounded worlds and explicit measured body roles required")
        if torch.cuda.device_count() != 1 or mujoco.get_mjcb_control() is not None:
            raise ValueError("one isolated GPU and no external control callback required")
        install()
        self.host_model = model
        self.worlds = worlds
        self.ball = int(ball)
        self.ready = False
        self.faulted = False
        self.attempted_steps = 0
        self.completed_steps = 0
        with wp.ScopedDevice("cuda:0"):
            self.model = mjw.put_model(model)
            self.data = mjw.put_data(
                model, mujoco.MjData(model), nworld=worlds, nconmax=128, njmax=1024
            )
            self.roles = wp.array(roles.astype(np.int32), dtype=int)
            self.force = wp.zeros((worlds, 3), dtype=float)

    def restore(self, *, qpos, qvel, warmstart, ctrl, time):
        if self.ready or self.faulted:
            raise ValueError("fresh unused diagnostic batch required")
        arrays = dict(qpos=qpos, qvel=qvel, qacc_warmstart=warmstart, ctrl=ctrl, time=time)
        for key, value in arrays.items():
            tensor = wp.to_torch(getattr(self.data, key))
            if value.shape != tuple(tensor.shape) or not np.isfinite(value).all():
                raise ValueError("finite complete actual initial state required: " + key)
            tensor.copy_(torch.as_tensor(value, dtype=tensor.dtype, device=tensor.device))
        with wp.ScopedDevice("cuda:0"):
            mjw.forward(self.model, self.data)
        wp.to_torch(self.data.qacc_warmstart).copy_(
            torch.as_tensor(warmstart, dtype=torch.float32, device="cuda:0")
        )
        wp.synchronize_device("cuda:0")
        self.ready = True

    def step(self, ctrl):
        if not self.ready or self.faulted:
            raise ValueError("live diagnostic batch required")
        if ctrl.shape != (self.worlds, self.host_model.nu) or not np.isfinite(ctrl).all():
            raise ValueError("complete finite recorded control required")
        try:
            wp.to_torch(self.data.ctrl).copy_(
                torch.as_tensor(ctrl, dtype=torch.float32, device="cuda:0")
            )
            self.attempted_steps += self.worlds
            with wp.ScopedDevice("cuda:0"):
                mjw.step(self.model, self.data)
                wp.synchronize_device("cuda:0")
                self.completed_steps += self.worlds
                nacon = int(wp.to_torch(self.data.nacon)[0])
                if (
                    nacon > self.data.naconmax
                    or int(wp.to_torch(self.data.nefc).max()) > self.data.njmax
                ):
                    raise ValueError("contact/constraint allocation overflow")
                self.force.zero_()
                wp.launch(
                    _forces,
                    dim=self.data.naconmax,
                    inputs=[self.model, self.data, self.ball, self.roles],
                    outputs=[self.force],
                )
                wp.synchronize_device("cuda:0")
            row = {
                k: wp.to_torch(getattr(self.data, k)).cpu().numpy().copy() for k in ("qpos", "qvel")
            }
            row["forces"] = wp.to_torch(self.force).cpu().numpy().copy()
            if any(not np.isfinite(x).all() for x in row.values()):
                raise ValueError("nonfinite actual diagnostic physics")
            return row
        except Exception:
            self.faulted = True
            raise
