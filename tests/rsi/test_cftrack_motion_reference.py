import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_motion_reference import load_cftrack_motion


def fixture():
    shapes = {
        "base_pos_w": (3,),
        "base_quat_w": (4,),
        "joint_pos": (29,),
        "joint_vel": (29,),
        "body_pos_w": (39, 3),
        "body_quat_w": (39, 4),
        "body_lin_vel_w": (39, 3),
        "body_ang_vel_w": (39, 3),
        "ee_pos_w": (5, 3),
        "ee_quat_w": (5, 4),
        "object_pos_w": (3,),
        "object_quat_w": (4,),
        "object_lin_vel_w": (3,),
        "object_ang_vel_w": (3,),
        "table1_pos_w": (3,),
        "table2_pos_w": (3,),
        "contact_info": (4, 1),
    }
    data = {k: np.zeros((10, *shape), np.float32) for k, shape in shapes.items()}
    for key, array in data.items():
        if "quat" in key:
            array[..., 0] = 1
    data["fps"] = np.array([50.0], np.float32)
    data["contact_info"][8, 1, 0] = 1
    return data


def test_complete_reference_is_not_trimmed_retimed_or_mutable(tmp_path):
    path = tmp_path / "motion.npz"
    data = fixture()
    np.savez(path, **data)
    result = load_cftrack_motion(path)
    assert set(result) == set(data)
    for key, value in result.items():
        np.testing.assert_array_equal(value, data[key])
        assert not value.flags.writeable


@pytest.mark.parametrize(
    "defect",
    ["missing", "fps", "nan", "shape", "quat", "contact", "budget", "object", "symlink"],
)
def test_malformed_or_out_of_budget_reference_is_rejected(tmp_path, defect):
    data = fixture()
    if defect == "missing":
        del data["table1_pos_w"]
    elif defect == "fps":
        data["fps"][0] = 60
    elif defect == "nan":
        data["joint_pos"][0, 0] = np.nan
    elif defect == "shape":
        data["contact_info"] = np.zeros((10, 4), np.float32)
    elif defect == "quat":
        data["body_quat_w"][1, 2, 0] = 0
    elif defect == "contact":
        data["contact_info"][0, 0, 0] = 0.5
    elif defect == "object":
        data["joint_pos"] = np.full((10, 29), "not numeric", object)
    path = tmp_path / "motion.npz"
    np.savez(path, **data)
    if defect == "symlink":
        link = tmp_path / "link.npz"
        link.symlink_to(path)
        path = link
    with pytest.raises(ValueError):
        load_cftrack_motion(path, maximum_frames=9 if defect == "budget" else 540)


def test_redundant_base_arrays_are_preserved_not_silently_substituted(tmp_path):
    data = fixture()
    data["base_pos_w"][0, 0] = 0.004
    path = tmp_path / "motion.npz"
    np.savez(path, **data)
    result = load_cftrack_motion(path)
    np.testing.assert_array_equal(result["base_pos_w"], data["base_pos_w"])
    np.testing.assert_array_equal(result["body_pos_w"], data["body_pos_w"])
