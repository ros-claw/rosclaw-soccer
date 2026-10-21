from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_native_phase_replay import PaidNativePhaseReplay
from rosclaw_soccer.rsi.paid_skill_handoff import snapshot


def actual_physics_fixture():
    """Small artificial body for strict integration tests, NOT a G1 success."""
    import mujoco

    parts = []
    for i in range(29):
        name = (
            "left_ankle_roll_link" if i == 0 else "right_ankle_roll_link" if i == 1 else f"body{i}"
        )
        parts.append(
            f'<body name="{name}" pos="0 .002 0"><joint name="j{i}" axis="0 1 0" '
            'armature=".01"/><geom size=".01" mass=".1" contype="0" conaffinity="0"/>'
        )
    actuators = "".join(f'<motor joint="j{i}"/>' for i in range(29))
    xml = (
        '<mujoco><option timestep=".002" gravity="0 0 0"/><worldbody>'
        '<body name="pelvis"><freejoint/><geom size=".1" mass="10"/>'
        + "".join(parts)
        + "</body>" * 29
        + '</body><body name="ball" pos=".15 0 0"><freejoint/>'
        '<geom size=".1" mass=".43"/></body></worldbody><actuator>'
        + actuators
        + "</actuator></mujoco>"
    )
    m = mujoco.MjModel.from_xml_string(xml)
    layout = SimpleNamespace(
        model=m,
        ai=np.arange(29),
        pelvis=m.body("pelvis").id,
        ball=m.body("ball").id,
        canonical=lambda data: (data.qpos.copy(), data.qvel.copy()),
    )
    reader = PaidNativePhaseReplay(SimpleNamespace(layout=layout))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    initial = snapshot(d)
    rows = []
    for _ in range(2):
        mujoco.mj_forward(m, d)
        row = dict(
            native_qpos_before=d.qpos.copy(),
            native_qvel_before=d.qvel.copy(),
            time_before=d.time,
            target=np.zeros(29),
        )
        q, v, f, t = [], [], [], []
        for _ in range(10):
            d.ctrl[:] = 0
            mujoco.mj_step(m, d)
            q.append(d.qpos.copy())
            v.append(d.qvel.copy())
            # Independent role measurement; fixture's only enabled contact is
            # pelvis-ball, so every recorded constraint belongs to nonfoot.
            force = np.zeros(3)
            for ci in range(d.ncon):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, wrench)
                force[2] = max(force[2], np.linalg.norm(wrench[:3]))
            f.append(force)
            t.append(d.ctrl.copy())
        row.update(qpos=np.array(q), qvel=np.array(v), forces=np.array(f), torque=np.array(t))
        rows.append(row)
    return reader, {k: np.array([r[k] for r in rows]) for k in rows[0]}, initial, snapshot(d)


def test_actual_twenty_native_steps_match_states_forces_and_complete_terminal():
    reader, raw, entry, end = actual_physics_fixture()
    result = reader.replay(raw, entry=entry, terminal=end)
    assert result["native_steps"] == reader.native_steps == 20
    assert result["exact_all_q_v_force_and_terminal"]
    assert result["nonfoot_contacts"][0]["body"] == "pelvis"
    assert result["nonfoot_contacts"][0]["first_before_any_foot_contact"]


@pytest.mark.parametrize("fault", ["q", "v", "force", "torque", "clock", "terminal", "missing"])
def test_native_reader_rejects_fabricated_physics_and_incomplete_entry(fault):
    reader, raw, entry, end = actual_physics_fixture()
    if fault == "terminal":
        end["qacc_warmstart"][0] += 0.01
    elif fault == "missing":
        del entry["xfrc_applied"]
    elif fault == "clock":
        raw["time_before"][0] += 0.001
    else:
        key = dict(q="qpos", v="qvel", force="forces", torque="torque")[fault]
        raw[key][0, 0, 0] += 0.01
    with pytest.raises(ValueError):
        reader.replay(raw, entry=entry, terminal=end)
