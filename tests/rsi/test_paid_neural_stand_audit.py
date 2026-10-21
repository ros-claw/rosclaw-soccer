from types import SimpleNamespace

import numpy as np
import pytest
from test_paid_neural_stand import fixture as standing_fixture

from rosclaw_soccer.rsi.paid_neural_stand_audit import (
    StandingObservationAuditor,
    audit_neural_stand,
)


def test_independent_actual_kinematic_observer_reconstructs_all_body_channels():
    import mujoco

    # Geometry-only fixture, no native integration or motion success claim.
    chain = "".join(
        f'<body name="b{i}" pos=".001 0 0">'
        '<joint type="hinge" axis="0 1 0" armature=".01"/>'
        '<geom type="sphere" size=".01" mass=".1" contype="0" conaffinity="0"/>'
        for i in range(29)
    )
    xml = (
        '<mujoco><worldbody><body name="pelvis" pos="0 0 .78"><freejoint/>'
        '<geom type="sphere" size=".05" mass="1" contype="0" conaffinity="0"/>'
        + chain
        + "</body>" * 29
        + '</body><body name="ball" pos="1 0 .11"><freejoint/>'
        '<geom type="sphere" size=".11" mass=".43" contype="0" conaffinity="0"/>'
        "</body></worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    q = model.qpos0.copy()
    q[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    q[7:36] = np.arange(29) * 0.01
    v = np.zeros(41)
    v[3:6] = [0.1, 0.2, 0.3]
    auditor = SimpleNamespace(
        layout=SimpleNamespace(
            model=model,
            pelvis=model.body("pelvis").id,
            ball=model.body("ball").id,
            rv=0,
            qi=np.arange(7, 36),
            vi=np.arange(6, 35),
        ),
        order=np.arange(29),
        offset=np.zeros(29),
    )
    observer = StandingObservationAuditor(auditor)
    pose = np.arange(29, dtype=np.float32) * 0.02
    last = np.ones(29, np.float32) * 0.5
    obs = observer.observe(q, v, pose, last, np.array([6.0, 0, 0.11]))
    assert obs.shape == (160,) and obs.dtype == np.float32 and observer.evaluations == 1
    assert np.array_equal(obs[:29], pose) and not obs[29:58].any()
    assert np.allclose(obs[58:61], [0, 0, -1]) and not obs[61:64].any()
    assert np.allclose(obs[64:67], [0.1, 0.2, 0.3])
    assert np.allclose(obs[67:96], np.arange(29) * 0.01) and not obs[96:125].any()
    assert np.array_equal(obs[125:154], last)
    assert np.allclose(obs[154:157], [0, -1, -0.67])
    assert np.allclose(obs[157:160], [0, -6, -0.67])


def audited_fixture(monkeypatch):
    """Flow-only motor fixture. Independent geometry is tested above."""
    ep, task, _ = standing_fixture(monkeypatch)
    raw, row = task.collect(100)
    a = SimpleNamespace(
        canonical=lambda q, v: (q, v),
        offset=np.zeros(29),
        scale=np.ones(29),
        order=np.arange(29),
        kp=np.ones(29),
        kd=np.ones(29),
        effort=np.ones(29) * 100,
        goal_mocap_id=0,
        layout=SimpleNamespace(ai=np.arange(29)),
    )

    class Observer:
        def observe(self, q, v, pose, last, goal):
            obs = np.zeros(160, np.float32)
            obs[:29] = pose
            obs[125:154] = last
            return obs

    entry = {k.removeprefix("initial_"): v for k, v in raw.items() if k.startswith("initial_")}
    args = dict(
        entry=entry,
        pose=task.pose,
        previous_action=np.zeros(29, np.float32),
        initial_memory=(np.ones((2, 1, 128), np.float32), np.ones((2, 1, 128), np.float32)),
        start_control=0,
        maximum_controls=100,
    )
    return a, Observer(), task.actor, raw, row, args


def test_standing_own_nn_observation_actions_and_every2ms_pd_recompute(monkeypatch):
    a, o, m, z, row, args = audited_fixture(monkeypatch)
    calls = []
    result = audit_neural_stand(a, o, m, z, row, **args, count_inference=lambda: calls.append(1))
    assert result["healthy"] and result["neural_rows"] == len(calls) == 100


@pytest.mark.parametrize("fault", ["obs", "memory", "torque", "clock", "pose"])
def test_standing_reader_rejects_fabricated_body_input_memory_or_motion(fault, monkeypatch):
    a, o, m, z, row, args = audited_fixture(monkeypatch)
    if fault == "obs":
        z["NN_obs"][0, 67] = 0.01
    if fault == "memory":
        z["NN_h_before"][2, 0, 0, 0] += 0.1
    if fault == "torque":
        z["torque"][0, 0, 0] = 1
    if fault == "clock":
        z["time_before"][0] = 0.1
    if fault == "pose":
        z["STAND_reference_pose"][0, 0] = 0.2
    with pytest.raises(ValueError):
        audit_neural_stand(a, o, m, z, row, **args)
