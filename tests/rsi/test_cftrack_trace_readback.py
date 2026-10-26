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
    assert (
        verify_cftrack_trace(reader.policy, data, reader.layout, raw, controls=4, motor_plan=plan)
        == 4
    )
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
