"""SIM-only regression: Torch writes and Warp reads share the caller stream."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import warp as wp

from scripts.rsi_warp_recorded_batch import RecordedWarpBatch


@pytest.mark.skipif(torch.cuda.device_count() != 1, reason="requires one isolated GPU")
def test_restore_and_step_use_torch_current_stream(monkeypatch):
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.002"/>
          <worldbody>
            <geom name="ground" type="plane" size="4 4 .1"/>
            <body name="robot" pos="0 0 1">
              <joint name="hinge" type="hinge" axis="0 1 0"/>
              <geom type="capsule" size=".05 .2" mass="1"/>
            </body>
            <body name="ball" pos="1 0 .2">
              <freejoint/><geom type="sphere" size=".1" mass=".43"/>
            </body>
          </worldbody>
          <actuator><motor joint="hinge"/></actuator>
        </mujoco>
        """
    )
    roles = np.full(model.nbody, -1, dtype=np.int32)
    roles[model.body("robot").id] = 2
    batch = RecordedWarpBatch(model, worlds=1, ball=model.body("ball").id, roles=roles)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    observed = []
    forward, step = mjw.forward, mjw.step

    def checked_forward(*args, **kwargs):
        observed.append(("forward", int(wp.get_stream("cuda:0").cuda_stream)))
        return forward(*args, **kwargs)

    def checked_step(*args, **kwargs):
        observed.append(("step", int(wp.get_stream("cuda:0").cuda_stream)))
        return step(*args, **kwargs)

    monkeypatch.setattr(mjw, "forward", checked_forward)
    monkeypatch.setattr(mjw, "step", checked_step)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        batch.restore(
            qpos=data.qpos[None, :].copy(),
            qvel=data.qvel[None, :].copy(),
            warmstart=data.qacc_warmstart[None, :].copy(),
            ctrl=data.ctrl[None, :].copy(),
            time=np.array([data.time]),
        )
        result = batch.step(np.array([[0.25]]))
    assert observed == [("forward", int(stream.cuda_stream)), ("step", int(stream.cuda_stream))]
    assert result["qpos"].shape == (1, model.nq)
    assert np.isfinite(result["qpos"]).all()
    assert batch.completed_steps == 1
