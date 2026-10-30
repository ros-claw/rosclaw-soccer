"""Direct SIM recorded-target batch; experimental diagnostics, not qualification.

No MDP resets, learned actor, hardware transport or installed-package edits.
Uses the original compiled model and the already tested addressed force decoder.
"""

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.types import vec5

from scripts.rsi_mjwarp_addressed_contact_force import addressed_contact_force, install


@wp.kernel
def _forces(
    geom_bodyid: wp.array(dtype=int),
    geom: wp.array(dtype=wp.vec2i),
    worldid: wp.array(dtype=int),
    frame: wp.array(dtype=wp.mat33),
    friction: wp.array(dtype=vec5),
    contact_dim: wp.array(dtype=int),
    efc_address: wp.array2d(dtype=int),
    efc_force: wp.array2d(dtype=float),
    nacon: wp.array(dtype=int),
    njmax: int,
    cone: int,
    ball: int,
    roles: wp.array(dtype=int),
    out: wp.array2d(dtype=float),
):
    ci = wp.tid()
    if ci >= nacon[0]:
        return
    g = geom[ci]
    if g[0] < 0 or g[1] < 0:
        return
    b0 = geom_bodyid[g[0]]
    b1 = geom_bodyid[g[1]]
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
    world = worldid[ci]
    force = addressed_contact_force(
        cone,
        frame,
        friction,
        contact_dim,
        efc_address,
        efc_force,
        njmax,
        nacon,
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
        # Torch's current stream is not generally Warp's default stream. The
        # state copies above must precede forward on the same CUDA stream.
        with (
            wp.ScopedDevice("cuda:0"),
            wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())),
        ):
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
            # Use the stream on which Torch wrote ctrl, including when callers
            # intentionally use a non-default stream. A later device-wide sync
            # alone cannot order an earlier cross-stream read after the write.
            with (
                wp.ScopedDevice("cuda:0"),
                wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())),
            ):
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
                    inputs=[
                        self.model.geom_bodyid,
                        self.data.contact.geom,
                        self.data.contact.worldid,
                        self.data.contact.frame,
                        self.data.contact.friction,
                        self.data.contact.dim,
                        self.data.contact.efc_address,
                        self.data.efc.force,
                        self.data.nacon,
                        self.data.njmax,
                        self.model.opt.cone,
                        self.ball,
                        self.roles,
                    ],
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
