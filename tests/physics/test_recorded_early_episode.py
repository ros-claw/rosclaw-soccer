"""Independent toy physical trajectory checks, not G1 policy qualification."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.recorded_early_episode import RecordedEarlyEpisode
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes
from tests.physics.test_g1_feedback_forecast import world


@pytest.fixture
def recorded(request):
    m, _, _ = world()
    m.dof_armature[6:35] = 0.1
    mode = getattr(request, "param", False)
    postcontact = mode in ("postcontact", "postcontact_late")
    course = (2.0, 1.0, 0.2)
    if postcontact:
        # Controlled toy contact, not native G1 skill evidence. Ball approaches
        # the left foot in zero gravity; all modifications precede world hash.
        m.opt.gravity[:] = 0
        m.body_pos[m.body("left_ankle_roll_link").id, 2] = -0.683
        course = (2.072 if mode == "postcontact_late" else 0.999, 0.0, -0.2)
    binary = np.empty(mujoco.mj_sizeModel(m), dtype=np.uint8)
    mujoco.mj_saveModel(m, None, binary)
    kp, kd = np.full(29, 10.0), np.full(29, 0.2)
    audit = RecordedEarlyEpisode(
        m,
        world_hash=hash_bytes(binary.tobytes()),
        kp=kp,
        kd=kd,
        course=course,
        defaults=np.zeros(29),
    )
    d = mujoco.MjData(m)
    d.qpos[:7] = (0, 0, 0.793, 1, 0, 0, 0)
    d.qpos[7:36] = 0
    d.qpos[36:] = (*course[:2], 0.13, 1, 0, 0, 0)
    d.qvel[35:] = (course[2], 0, 0, 0, course[2] / 0.11, 0)
    mujoco.mj_forward(m, d)
    history = {}
    limits = np.asarray(G1_HARD_TORQUE_LIMITS)
    previous_forces = np.zeros(6)
    trigger = None
    for f in range(300):
        offset = np.zeros(29)
        if postcontact:
            if trigger is None and f >= 30 and np.any(previous_forces[:2] > 1):
                trigger = f
            if trigger is not None and f < trigger + 3:
                offset[5] = 0.1
        elif mode is not True and 50 <= f < 53:
            offset[5] = 0.1
        row = dict(
            canonical_qpos=d.qpos.copy()[None],
            canonical_qvel=d.qvel.copy()[None],
            joint_target_rad=np.zeros((1, 29)),
            early_impact_feedback_requested_offset_nm=offset[None],
        )
        pd, tau, actual, applied, heights = [], [], [], [], []
        forces = np.zeros(6)
        for _ in range(10):
            raw = -d.qpos[7:36] * kp - d.qvel[6:35] * kd
            parent = np.clip(raw, -limits, limits)
            torque = np.clip(raw + offset, -limits, limits)
            d.ctrl[:] = torque
            pd.append(raw.copy())
            tau.append(torque.copy())
            applied.append(torque - parent)
            mujoco.mj_step(m, d)
            actual.append(d.actuator_force.copy())
            heights.append(float(d.xpos[m.body("pelvis").id, 2]))
            for n in range(d.ncon):
                c = d.contact[n]
                g1, g2 = int(c.geom1), int(c.geom2)
                if audit.bg in (g1, g2):
                    body = int(m.geom_bodyid[g2 if g1 == audit.bg else g1])
                    if body in audit.contacts:
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(m, d, n, wrench)
                        i = audit.contacts.index(body)
                        forces[i] = max(forces[i], float(np.linalg.norm(wrench[:3])))
        row.update(
            pd_requested_torque_nm=np.asarray(pd)[None],
            torque_nm=np.asarray(tau)[None],
            actual_actuator_force_nm=np.asarray(actual)[None],
            early_impact_feedback_applied_offset_nm=np.asarray(applied)[None],
            pelvis_z_per_substep_m=np.asarray(heights)[None],
            force_n=forces[None],
            ball_position_after_step_m=d.xpos[audit.ball].copy()[None],
        )
        for k, v in row.items():
            history.setdefault(k, []).append(v)
        previous_forces = forces.copy()
    return audit, {k: np.asarray(v) for k, v in history.items()}


def test_full300_frames_reintegrated_from_independent_declared_initial_state(recorded):
    a, t = recorded
    r = a.verify(t)
    assert r["actual_physical_steps"] == a.attempted_steps == 3000
    assert max(r["maximum_absolute_errors"].values()) == 0
    assert r["recorded_targets_consumed"] and not r["neural_targets_recomputed"]
    assert not r["teacher_qualified"] and not r["hardware_authorized"]
    with pytest.raises(ValueError, match="once-only"):
        a.verify(t)
    assert a.attempted_steps == 3000


@pytest.mark.parametrize("recorded", [True], indirect=True)
def test_unforced_no_fabricated_impact_arrays(recorded):
    audit, trace = recorded
    trace = {k: v for k, v in trace.items() if not k.startswith("early_impact_feedback_")}
    original_keys = set(trace)
    result = audit.verify_unforced_control(trace)
    assert result["actual_physical_steps"] == 3000
    assert max(result["maximum_absolute_errors"].values()) == 0
    assert result["plain_pd_only"] and not result["impact_action_arrays_fabricated_or_aliased"]
    assert not result["neural_targets_recomputed"] and set(trace) == original_keys


def test_unforced_rejects_offset_bearing_source_before_integration(recorded):
    audit, trace = recorded
    with pytest.raises(ValueError, match="cannot contain impact fields"):
        audit.verify_unforced_control(trace)
    assert audit.failed and audit.attempted_steps == 0


@pytest.mark.parametrize(
    "field",
    [
        "canonical_qpos",
        "canonical_qvel",
        "pd_requested_torque_nm",
        "torque_nm",
        "actual_actuator_force_nm",
        "pelvis_z_per_substep_m",
        "force_n",
        "ball_position_after_step_m",
        "early_impact_feedback_applied_offset_nm",
    ],
)
def test_every_physical_capture_checked_and_failure_cannot_retry(recorded, field):
    a, t = recorded
    t[field][100].flat[0] += 0.01
    with pytest.raises(ValueError, match="differs"):
        a.verify(t)
    assert a.failed and 0 < a.attempted_steps < 3000
    n = a.attempted_steps
    with pytest.raises(ValueError, match="once-only"):
        a.verify(t)
    assert a.attempted_steps == n


@pytest.mark.parametrize("bad", ["initial", "cap", "lease", "short", "nan"])
def test_invalid_trace_rejected_before_any_physical_step(recorded, bad):
    a, t = recorded
    if bad == "initial":
        t["canonical_qpos"][0, 0, 0] += 0.01
    elif bad == "cap":
        t["early_impact_feedback_requested_offset_nm"][50, 0, 5] = 100
    elif bad == "lease":
        t["early_impact_feedback_requested_offset_nm"][53, 0, 5] = 0.1
    elif bad == "short":
        t["force_n"] = t["force_n"][:-1]
    else:
        t["torque_nm"][100, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        a.verify(t)
    assert a.failed and a.attempted_steps == 0
