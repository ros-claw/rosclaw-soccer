"""Native threaded vector integration is identical to independent scalar worlds."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.rsi.native_physics_batch import NativePhysicsBatch


def model():
    return mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep="0.002"/>
      <worldbody><body><joint name="j" type="hinge" damping="1"/>
      <geom type="capsule" size="0.03 0.2" pos="0 0 .2"/></body></worldbody>
      <actuator><motor joint="j"/></actuator></mujoco>""")


def test_native_threaded_batch_scalar_and_continuation_exact():
    m = model()
    q = np.array([[0.1], [0.2], [0.3], [0.4]])
    v = q * 0.1
    warm = q * 0.5
    old = q * 2
    t = np.arange(4) * 0.02
    scalar = []
    for i in range(4):
        d = mujoco.MjData(m)
        d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:], d.time = (
            q[i],
            v[i],
            old[i],
            warm[i],
            t[i],
        )
        mujoco.mj_forward(m, d)
        d.qacc_warmstart[:] = warm[i]
        scalar.append(d)
    with NativePhysicsBatch(m, 4, threads=2) as b:
        b.restore(qpos=q, qvel=v, previous_ctrl=old, warmstart=warm, time=t)
        for target in (q * 3, q * -1):
            expected_q, expected_v = [], []
            for i, d in enumerate(scalar):
                d.ctrl[:] = target[i]
                qs, vs = [], []
                for _ in range(10):
                    mujoco.mj_step(m, d)
                    qs.append(d.qpos.copy())
                    vs.append(d.qvel.copy())
                expected_q.append(qs)
                expected_v.append(vs)
            actual = b.advance(target, substeps=10)
            assert np.array_equal(actual["qpos"], expected_q)
            assert np.array_equal(actual["qvel"], expected_v)
        assert b.native_steps == 80
        with pytest.raises(ValueError, match="fresh unused"):
            b.restore(qpos=q, qvel=v, previous_ctrl=old, warmstart=warm, time=t)
    with pytest.raises(ValueError, match="restored live"):
        b.advance(q, substeps=1)


def test_native_batch_validation_before_physics():
    m = model()
    with NativePhysicsBatch(m, 2) as b:
        q = np.zeros((2, 1))
        with pytest.raises(ValueError, match="finite qpos"):
            b.restore(qpos=q * np.nan, qvel=q, previous_ctrl=q, warmstart=q, time=np.zeros(2))
        assert b.native_steps == 0
        b.restore(qpos=q, qvel=q, previous_ctrl=q, warmstart=q, time=np.zeros(2))
        with pytest.raises(ValueError, match="finite ctrl"):
            b.advance(np.zeros((3, 1)), substeps=1)
        with pytest.raises(ValueError, match="physical fields"):
            b.advance(q, substeps=1, fields=("qvel", "qvel"))
        assert b.native_steps == 0


def test_selected_worlds_freeze_terminal_state_and_clock():
    m = model()
    q = np.zeros((3, 1))
    with NativePhysicsBatch(m, 3) as b:
        b.restore(qpos=q, qvel=q, previous_ctrl=q, warmstart=q, time=np.zeros(3))
        inactive = b.data[1].qpos.copy(), b.data[1].qvel.copy(), b.data[1].time
        result = b.advance(q + 0.3, substeps=5, world_ids=np.array([2, 0], dtype=np.int64))
        assert result["qpos"].shape == (2, 5, 1)
        assert b.native_steps == 10
        assert np.array_equal(b.data[1].qpos, inactive[0])
        assert np.array_equal(b.data[1].qvel, inactive[1])
        assert b.data[1].time == inactive[2]
        assert b.data[0].time == b.data[2].time == 0.01
        for ids in (np.array([0, 0]), np.array([-1]), np.array([3]), np.array([], dtype=np.int64)):
            with pytest.raises(ValueError, match="world indices"):
                b.advance(q, substeps=1, world_ids=ids)
        assert b.native_steps == 10
