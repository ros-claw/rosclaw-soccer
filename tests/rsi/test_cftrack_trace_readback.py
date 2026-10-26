import mujoco
import numpy as np
import pytest
from test_shared_cftrack_forecast import motor_plan, setup

from rosclaw_soccer.rsi.cftrack_trace_readback import verify_cftrack_trace


def capture(player, plan):
    obs, actions, rows, targets, decoded = [], [], [], [], []
    if plan:
        player.install_motor_plan(plan)
    session = player.policy.ort_session

    class Capture:
        def run(self, outputs, inputs):
            result = session.run(outputs, inputs)
            obs.append(inputs["obs"].copy())
            actions.append(result[0].copy())
            return result

    player.policy.ort_session = Capture()
    for _ in range(4):
        target, measured = player.propose()
        rows.append(player.world.advance({player.prefix: target}))
        targets.append(target)
        decoded.append(measured["nn_decoded_target"])
        player.commit()
    raw = {k: np.concatenate([r[k] for r in rows]) for k in ["qpos", "qvel"]}
    raw.update(
        neural_obs=np.array(obs),
        neural_actions=np.array(actions),
        joint_targets=np.array(targets),
        nn_decoded_targets=np.array(decoded),
    )
    return raw


@pytest.mark.parametrize("use_motor", [False, True])
def test_independent_scratch_nn_inputs_and_both_target_streams_rebuild(tmp_path, use_motor):
    source, _ = setup(tmp_path, physical_range=1.0, cold=True)
    plan = motor_plan(source) if use_motor else None
    raw = capture(source, plan)
    reader, _ = setup(tmp_path, physical_range=1.0, cold=True)
    data = mujoco.MjData(reader.world.model)
    data.qpos[:] = reader.world.data.qpos
    data.qvel[:] = reader.world.data.qvel
    original = reader.policy.ort_session
    observed = []

    def audit(policy, observation):
        assert policy is reader.policy
        observed.append(observation.copy())

    assert (
        verify_cftrack_trace(
            reader.policy,
            data,
            reader.layout,
            raw,
            controls=4,
            motor_plan=plan,
            proprioception_audit=audit,
        )
        == 4
    )
    assert len(observed) == 4
    np.testing.assert_array_equal(np.asarray(observed), raw["neural_obs"])
    assert (
        data.time == 0 and reader.world.native_steps == 0 and reader.policy.ort_session is original
    )


@pytest.mark.parametrize(
    "key", ["neural_obs", "neural_actions", "nn_decoded_targets", "joint_targets"]
)
def test_tampered_stream_rejects_and_restores_original_session(tmp_path, key):
    source, _ = setup(tmp_path, physical_range=1.0, cold=True)
    raw = capture(source, None)
    raw[key].flat[-1] += 0.001
    reader, _ = setup(tmp_path, physical_range=1.0, cold=True)
    data = mujoco.MjData(reader.world.model)
    data.qpos[:] = reader.world.data.qpos
    original = reader.policy.ort_session
    with pytest.raises(ValueError):
        verify_cftrack_trace(reader.policy, data, reader.layout, raw, controls=4)
    assert reader.policy.ort_session is original and data.time == 0


def test_readback_can_rebuild_explicit_previous_substep_ball_cache(tmp_path):
    source, _ = setup(tmp_path, physical_range=1.0, cold=True)
    current = source.world.data.qpos.copy()
    previous = current.copy()
    previous[source.layout.bq] += 0.03
    source.world.data.qpos[:] = previous
    mujoco.mj_kinematics(source.world.model, source.world.data)
    mujoco.mj_comPos(source.world.model, source.world.data)
    expected_cached_ball = source.world.data.xpos[source.layout.ball].copy()
    source.world.data.qpos[:] = current
    raw = capture(source, None)
    reader, _ = setup(tmp_path, physical_range=1.0, cold=True)
    data = mujoco.MjData(reader.world.model)
    data.qpos[:] = current
    data.qvel[:] = reader.world.data.qvel
    observed = []

    def audit(policy, _):
        observed.append(policy.state_cmd.obj_pos.copy())

    assert (
        verify_cftrack_trace(
            reader.policy,
            data,
            reader.layout,
            raw,
            controls=4,
            proprioception_audit=audit,
            initial_cached_qpos=previous,
        )
        == 4
    )
    np.testing.assert_array_equal(observed[0], expected_cached_ball)
