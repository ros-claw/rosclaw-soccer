import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from scripts.rsi_motion_csv_body_compatibility import inspect_motion


def write_motion(tmp_path, *, millimeters=False, wrong_quaternion=False, width=29):
    header = [
        "root_pos_x(m)",
        "root_pos_y(m)",
        "root_pos_z(m)",
        "root_rot_w",
        "root_rot_x",
        "root_rot_y",
        "root_rot_z",
    ]
    header += [f"dof_{name}(rad)" for name in G1_DDS_JOINT_NAMES[:width]]
    if millimeters:
        header[0] = "root_pos_x(mm)"
    values = np.zeros((2, len(header)))
    values[:, 2] = 0.8
    values[:, 3] = 2 if wrong_quaternion else 1
    path = tmp_path / "motion.csv"
    np.savetxt(path, values, delimiter=",", header=",".join(header), comments="")
    return path


@pytest.mark.parametrize(
    "options", [dict(millimeters=True), dict(width=23), dict(wrong_quaternion=True)]
)
def test_invalid_source_contract_rejected_before_world_loading(tmp_path, options):
    path = write_motion(tmp_path, **options)
    with pytest.raises(ValueError):
        inspect_motion(path, tmp_path / "absent-model.mjb")
