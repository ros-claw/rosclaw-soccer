import numpy as np
import pytest
from test_shared_native_motor_world import source_xml
from test_shared_nn_forecast import setup

from rosclaw_soccer.rsi.ball_clearance_control import (
    BallClearanceController,
    BallClearancePlan,
    clearance_offset,
)
from rosclaw_soccer.rsi.shared_nn_forecast import policy_state_hash


def test_repulsion_is_measured_directional_smooth_and_leg_only():
    gradient = np.zeros((2, 12))
    gradient[0, :4] = [0.4, -0.2, 0.0, 0.1]
    previous = np.zeros(29)
    for _ in range(20):
        current = clearance_offset(np.array([-0.01, 1.0]), gradient, previous)
        assert abs(current - previous).max() <= 0.008000001
        assert abs(current).max() <= 0.060000001
        assert not current[12:].any()
        assert gradient[0] @ current[:12] > 0
        previous = current
    for _ in range(8):
        previous = clearance_offset(np.ones(2), gradient, previous)
    np.testing.assert_array_equal(previous, np.zeros(29))


@pytest.mark.parametrize("defect", ["nan", "shape", "dtype", "upper", "trust"])
def test_bad_geometry_or_history_rejected(defect):
    distances, gradient, previous = np.ones(2), np.zeros((2, 12)), np.zeros(29)
    if defect == "nan":
        gradient[0, 0] = np.nan
    elif defect == "shape":
        distances = np.ones(3)
    elif defect == "dtype":
        gradient = gradient.astype(np.float32)
    elif defect == "upper":
        previous[12] = 0.01
    else:
        previous[0] = 0.1
    with pytest.raises(ValueError):
        clearance_offset(distances, gradient, previous)


def test_no_implicit_hyperparameter_search():
    BallClearancePlan().validate()
    with pytest.raises(ValueError):
        BallClearancePlan(0.12).validate()


@pytest.mark.parametrize("anticipatory", [False, True])
def test_closed_nn_clearance_future_matches_own_actual_targets_and_history(tmp_path, anticipatory):
    import xml.etree.ElementTree as ET

    root = ET.fromstring(source_xml())
    root.find(".//body[@name='ball']").set("pos", "0.28 0.06 1")
    for index, side in ((3, "left"), (9, "right")):
        body = root.find(f".//body[@name='link{index}']")
        body.set("name", side + "_knee_link")
        body.find("geom").set("pos", "0.2 0.02 0")
    player, forecaster = setup(
        tmp_path, budget=80, fixture_xml=ET.tostring(root, encoding="unicode")
    )
    # Complete the existing pending target to admit a fresh pre-proposal branch.
    player.world.advance({player.prefix: player.pending.copy()})
    player.commit()
    before = policy_state_hash(player.policy)
    predicted = forecaster.forecast(
        controls=4,
        before_proposal=True,
        clearance_plan=BallClearancePlan(anticipatory_velocity=anticipatory),
    )
    assert policy_state_hash(player.policy) == before
    assert predicted["measured_clearance_feedback"]
    assert any(e["applied"] for e in predicted["executed_target_edits"])
    controller = BallClearanceController(
        player, BallClearancePlan(anticipatory_velocity=anticipatory)
    )
    records, targets = [], []
    for _ in range(4):
        player.propose()
        target, edit = controller.apply()
        assert edit["next_action_history_matches_executed_codec"]
        targets.append(target)
        records.append(player.world.advance({player.prefix: target}))
        player.commit()
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(predicted[key], np.concatenate([r[key] for r in records]))
    np.testing.assert_array_equal(predicted["joint_targets"], np.stack(targets))
