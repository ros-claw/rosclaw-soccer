from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_native_episode import PaidNativeEpisode
from rosclaw_soccer.rsi.paid_skill_handoff import practice_ball_drop, snapshot


def boundary():
    ep = PaidNativeEpisode.__new__(PaidNativeEpisode)
    q = np.zeros(43)
    q[2:4] = [0.8, 1]
    d = SimpleNamespace(
        qpos=q,
        qvel=np.arange(41, dtype=float) * 0.001,
        qacc_warmstart=np.ones(41),
        ctrl=np.arange(29, dtype=float),
        mocap_pos=np.array([[6.0, 0, 0.11]]),
        mocap_quat=np.array([[1.0, 0, 0, 0]]),
        time=7.18,
        contact=[],
        warning=[SimpleNamespace(number=0)],
    )
    ep.env = SimpleNamespace(data=d, goal_mocap_id=0)
    ep.layout = SimpleNamespace(canonical=lambda _: (d.qpos.copy(), d.qvel.copy()), ball=99)
    ep.done, ep.termination_reason = True, "REFERENCE_COMPLETE"
    ep.warning = (0,)
    ep.last_action = np.linspace(-1, 1, 29, dtype=np.float32)
    ep.clip = SimpleNamespace(length=300)
    return ep


def test_new_skill_preserves_all_physical_state_and_actual_previous_action():
    ep = boundary()
    before = {k: np.array(v, copy=True) for k, v in vars(ep.env.data).items() if k != "warning"}
    previous = ep.last_action.copy()
    clip = SimpleNamespace(length=200)
    ep.begin_current_state(clip, goal=np.array([6.0, 0, 0.11]))
    assert not ep.done and ep.frame == 0 and ep.clip is clip
    assert ep.termination_reason == "ACTIVE"
    np.testing.assert_array_equal(ep.last_action, previous)
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(ep.env.data, key), value)
    assert ep.initial["time"] == 7.18
    assert not ep.ref_pos.any()


@pytest.mark.parametrize("reason", ["ACTIVE", "BODY_FAILURE", "NOT_STARTED"])
def test_failure_or_active_body_cannot_be_restarted(reason):
    ep = boundary()
    ep.termination_reason = reason
    with pytest.raises(ValueError, match="healthy reference"):
        ep.begin_current_state(ep.clip, goal=np.array([6.0, 0, 0.11]))


def test_changed_goal_unhealthy_body_warning_and_overlap_rejected_without_clock_reset():
    for fault in ("goal", "height", "upright", "warning", "overlap"):
        ep = boundary()
        goal = np.array([6.0, 0, 0.11])
        if fault == "goal":
            goal[0] = 2
        elif fault == "height":
            ep.env.data.qpos[2] = 0.6
        elif fault == "upright":
            ep.env.data.qpos[3:5] = [0, 1]
        elif fault == "warning":
            ep.env.data.warning[0].number = 1
        else:
            ep.env.model = SimpleNamespace(geom_bodyid=[99, 3])
            ep.env.data.contact = [SimpleNamespace(geom=[0, 1])]
            ep.robot = {3}
        with pytest.raises(ValueError):
            ep.begin_current_state(SimpleNamespace(length=180), goal=goal)
        assert ep.done and ep.termination_reason == "REFERENCE_COMPLETE"
        assert ep.env.data.time == 7.18 and ep.clip.length == 300


def drop_fixture(radius=0.2):
    import mujoco

    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="pelvis" pos="0 0 .8"><freejoint/>'
        f'<geom type="sphere" size="{radius}" mass="1"/></body>'
        '<body name="ball" pos="3 0 .11"><freejoint/>'
        '<geom type="sphere" size=".11" mass=".43"/></body></worldbody></mujoco>'
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    def canonical(data):
        q = np.zeros(43)
        q[:7], q[36:] = data.qpos[:7], data.qpos[7:]
        return q, np.zeros(41)

    return SimpleNamespace(
        env=SimpleNamespace(model=m, data=d),
        layout=SimpleNamespace(bq=7, bv=6, ball=m.body("ball").id, canonical=canonical),
        robot={m.body("pelvis").id},
        goal=np.array([6.0, 0, 0.11]),
        termination_reason="REFERENCE_COMPLETE",
    )


def test_practice_drop_changes_only_ball_and_nearby_ball_is_never_teleported():
    ep = drop_fixture()
    before = snapshot(ep.env.data)
    trace = practice_ball_drop(ep, foot="left")
    assert trace["performed"]
    np.testing.assert_array_equal(ep.env.data.qpos[7:], [0.8, 0.15, 0.4, 1, 0, 0, 0])
    np.testing.assert_array_equal(ep.env.data.qpos[:7], before["qpos"][:7])
    for k in before:
        if k != "qpos":
            np.testing.assert_array_equal(getattr(ep.env.data, k), before[k])
    # Now inside the zone: fixture must not move the same ball again.
    near = snapshot(ep.env.data)
    assert not practice_ball_drop(ep, foot="right")["performed"]
    for k in near:
        np.testing.assert_array_equal(getattr(ep.env.data, k), near[k])


def test_invalid_drop_is_rejected_in_scratch_without_any_live_write():
    ep = drop_fixture(radius=1)
    before = snapshot(ep.env.data)
    with pytest.raises(ValueError, match="overlaps"):
        practice_ball_drop(ep, foot="left")
    for k in before:
        np.testing.assert_array_equal(getattr(ep.env.data, k), before[k])


@pytest.mark.parametrize("carry_memory", [False, True])
def test_collector_records_executed_transition_history_with_explicit_memory(carry_memory):
    """Unit test of collector plumbing, not a physical rollout or skill proof."""
    from pathlib import Path
    from threading import Lock

    import torch

    from rosclaw_soccer.rsi.paid_research_bench import PaidResearchBench

    bench = PaidResearchBench.__new__(PaidResearchBench)
    bench.lock, bench.neural_rows = Lock(), 0
    ep = SimpleNamespace(
        done=False,
        frame=0,
        clip=SimpleNamespace(length=14, name="left", path=Path("left.npz")),
        last_action=np.arange(29, dtype=np.float32) * 0.03,
        initial={"time": np.array(9.0)},
        goal=np.array([6.0, 0, 0.11]),
    )

    def observation():
        obs = np.zeros(160, np.float32)
        obs[125:154] = ep.last_action
        return obs

    def step(action):
        q = np.zeros((10, 43))
        q[:, 2:4] = [0.8, 1]
        q[:, 36:39] = [0.8, 0.15, 0.11]
        ep.last_action = action.copy()
        ep.frame += 1
        ep.done = ep.frame == ep.clip.length
        return dict(qpos=q, qvel=np.zeros((10, 41)), forces=np.zeros((10, 3)), body_failed=False)

    class Actor:
        def __call__(self, obs, state):
            return torch.full((1, 1, 29), 0.5), (state[0] + 1, state[1] + 2)

    ep.observation, ep.step = observation, step
    previous = ep.last_action.copy()
    memory = (torch.full((2, 1, 128), 3.0), torch.full((2, 1, 128), 5.0)) if carry_memory else None
    raw, row = bench.collect_current(ep, Actor(), transition_controls=12, initial_state=memory)
    assert bench.neural_rows == 14 and row["composite_teacher"]
    np.testing.assert_array_equal(raw["NN_obs"][0, 125:154], previous)
    np.testing.assert_array_equal(raw["NN_obs"][1:, 125:154], raw["NN_action"][:-1])
    assert np.all(raw["NN_h_before"][0] == (3 if carry_memory else 0))
    assert np.all(raw["NN_c_before"][0] == (5 if carry_memory else 0))
    if carry_memory:
        assert torch.all(memory[0] == 3) and torch.all(memory[1] == 5)
    np.testing.assert_array_equal(raw["NN_h_after"][:-1], raw["NN_h_before"][1:])
    for i in range(12):
        u = (i + 1) / 12
        weight = 10 * u**3 - 15 * u**4 + 6 * u**5
        np.testing.assert_array_equal(
            raw["NN_action"][i], ((1 - weight) * previous + weight * 0.5).astype(np.float32)
        )
    np.testing.assert_array_equal(raw["NN_action"][12:], raw["NN_nominal_action"][12:])


def test_reference_memory_carry_is_explicit_and_never_changes_physical_state():
    ep = boundary()
    ep.ref_pos = np.linspace(-1, 1, 29, dtype=np.float32)
    ep.ref_vel = np.linspace(-2, 2, 29, dtype=np.float32)
    ep.ref_ang = np.array([0.1, 0.2, 0.3], np.float32)
    references = [v.copy() for v in (ep.ref_pos, ep.ref_vel, ep.ref_ang)]
    before = {k: np.array(v, copy=True) for k, v in vars(ep.env.data).items() if k != "warning"}
    ep.begin_current_state(ep.clip, goal=np.array([6.0, 0, 0.11]), preserve_reference=True)
    for actual, expected in zip((ep.ref_pos, ep.ref_vel, ep.ref_ang), references, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(ep.env.data, key), value)


@pytest.mark.parametrize("fault", ["shape", "dtype", "nonfinite", "count", "not_tensor"])
def test_bad_neural_state_cannot_enter_collection(fault):
    import torch

    from rosclaw_soccer.rsi.paid_research_bench import PaidResearchBench

    bench = PaidResearchBench.__new__(PaidResearchBench)
    ep = SimpleNamespace(done=False, frame=0, last_action=np.zeros(29, np.float32))
    h, c = torch.zeros(2, 1, 128), torch.zeros(2, 1, 128)
    if fault == "shape":
        h = torch.zeros(1, 2, 128)
    if fault == "dtype":
        h = h.double()
    if fault == "nonfinite":
        h[0, 0, 0] = float("nan")
    if fault == "not_tensor":
        h = np.zeros((2, 1, 128), np.float32)
    initial = (h,) if fault == "count" else (h, c)
    with pytest.raises(ValueError, match="per-player h/c"):
        bench.collect_current(ep, None, initial_state=initial)
