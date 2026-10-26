import numpy as np
import pytest
from test_shared_native_motor_world import source_xml

from rosclaw_soccer.rsi.reference_support_geometry import (
    fit_support_offset,
    score_support_offset,
    support_distances,
)
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def test_fits_one_offset_and_scores_other_data_without_mutation():
    distances = np.array([[-0.02, -0.04], [0.1, -0.06]], dtype=np.float64)
    labels = np.array([[True, True], [False, True]])
    before = distances.copy()
    fitted = fit_support_offset(distances, labels)
    assert fitted["fitted_offset_m"] == pytest.approx(0.04)
    assert fitted["labeled_train_samples"] == 3
    assert fitted["physical_execution_qualified"] is False
    score = score_support_offset(
        np.array([[-0.03, -0.02]], dtype=np.float64),
        np.array([[True, True]]),
        fitted_offset_m=fitted["fitted_offset_m"],
    )
    assert score["mean_absolute_residual_m"] == pytest.approx(0.015)
    np.testing.assert_array_equal(before, distances)


@pytest.mark.parametrize(
    ("distances", "labels"),
    [
        (np.zeros((2, 2)), np.zeros((2, 2), dtype=bool)),
        (np.array([[np.nan, 0.0]]), np.array([[True, False]])),
        (np.zeros((2, 2)), np.ones((2, 2))),
        (np.zeros((2, 2)), np.ones((1, 2), dtype=bool)),
    ],
)
def test_rejects_invalid_or_empty_labels(distances, labels):
    with pytest.raises(ValueError):
        fit_support_offset(distances, labels)


def test_capsule_clearance_uses_named_geometries_without_integrating():
    import mujoco

    xml = source_xml().replace('name="ball"', 'name="soccer_ball"')
    xml = xml.replace("<worldbody>", '<worldbody><geom name="floor" type="plane" size="1 1 .1"/>')
    feet = "".join(
        f'<geom name="{side}_foot{i}_collision" type="capsule" size=".1 .02" mass=".001"/>'
        for side in ("left", "right")
        for i in range(1, 8)
    )
    xml = xml.replace('<geom type="sphere" size=".05" mass="1"/>', feet)
    model = mujoco.MjModel.from_xml_string(shared_native_scene(xml, prefixes=("a/",)))
    fields = {
        "dof": np.zeros((2, 23), np.float64),
        "root_rot": np.array([[0, 0, 0, 1], [0, 0, 0, 1]], np.float64),
        "root_trans_offset": np.array([[0, 0, 1], [0, 0, 0.5]], np.float64),
        "contact_mask": np.array([[1, 0], [0, 1]], np.float64),
    }
    distances, labels = support_distances(
        model, fields, expected_model_hash=compiled_model_hash(model)
    )
    np.testing.assert_allclose(distances, [[0.88, 0.88], [0.38, 0.38]], atol=1e-12)
    np.testing.assert_array_equal(labels, [[True, False], [False, True]])
