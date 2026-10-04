import numpy as np
import pytest

from rosclaw_soccer.sim.contact_force_observation import maximum_contact_force_by_body


def fixture():
    return dict(
        world_ids=np.array([0, 1, 0, 0, 1]),
        geom_pairs=np.array([[0, 1], [2, 0], [0, 3], [1, 0], [1, 2]]),
        force_torque=np.array(
            [
                [3, 4, 0, 100, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 4, 0, 0, 0],
                [99, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        ),
        geom_body_ids=np.array([10, 20, 30, 20]),
        target_geom_id=0,
        observed_body_ids=(20, 30, 40),
        world_count=2,
    )


def test_world_identity_both_geom_orders_maximum_not_sum_and_no_torque():
    inputs = fixture()
    before = inputs["force_torque"].copy()
    result = maximum_contact_force_by_body(**inputs)
    assert np.array_equal(result, [[5, 0, 0], [0, 2, 0]])
    assert np.array_equal(inputs["force_torque"], before)
    assert not result.flags.writeable


def test_no_active_contacts_yields_owned_zero_observation():
    inputs = fixture()
    for name in ("world_ids", "geom_pairs", "force_torque"):
        inputs[name] = inputs[name][:0]
    assert np.array_equal(maximum_contact_force_by_body(**inputs), np.zeros((2, 3)))


@pytest.mark.parametrize(
    "fault", ["world", "geom", "flex", "nan", "overflow", "shape", "duplicate"]
)
def test_invalid_contact_identity_and_nonfinite_force_rejected(fault):
    inputs = fixture()
    if fault == "world":
        inputs["world_ids"][0] = 2
    elif fault == "geom":
        inputs["geom_pairs"][0, 0] = 4
    elif fault == "flex":
        inputs["geom_pairs"][0, 0] = -1
    elif fault == "nan":
        inputs["force_torque"][0, 0] = np.nan
    elif fault == "overflow":
        inputs["force_torque"][0, 0] = 1e308
    elif fault == "shape":
        inputs["force_torque"] = inputs["force_torque"][:, :3]
    else:
        inputs["observed_body_ids"] = (20, 20)
    with pytest.raises(ValueError):
        maximum_contact_force_by_body(**inputs)
