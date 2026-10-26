import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from test_shared_native_motor_world import source_xml

from rosclaw_soccer.rsi.reference_support_ik import fit_reference_support_pose
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def _model():
    xml = source_xml().replace('name="ball"', 'name="soccer_ball"')
    xml = xml.replace('armature=".1"/>', 'armature=".1" range="-2 2"/>')
    root = ET.fromstring(xml)
    ET.SubElement(root.find("worldbody"), "geom", name="floor", type="plane", size="1 1 .1")
    body = root.find(".//body[@name='pelvis']")
    for side in ("left", "right"):
        for i in range(1, 8):
            ET.SubElement(
                body,
                "geom",
                name=f"{side}_foot{i}_collision",
                type="capsule",
                size=".1 .02",
                pos="0 0 -.68",
                mass=".001",
            )
    return mujoco.MjModel.from_xml_string(
        shared_native_scene(ET.tostring(root, encoding="unicode"), prefixes=("a/",))
    )


def test_static_support_fit_does_not_mutate_or_authorize_motion():
    model = _model()
    root = np.array([0, 0, 0.7, 1, 0, 0, 0], dtype=np.float64)
    joints = np.zeros(29, dtype=np.float64)
    before_root, before_joints = root.copy(), joints.copy()
    result = fit_reference_support_pose(
        model, root, joints, np.array([True, True]), expected_model_hash=compiled_model_hash(model)
    )
    assert result["reference_geometry_feasible"] is True
    assert result["root_z_change_m"] == pytest.approx(0.1, abs=1e-3)
    assert result["integrated_native_steps"] == 0
    assert result["physical_execution_qualified"] is False
    np.testing.assert_array_equal(root, before_root)
    np.testing.assert_array_equal(joints, before_joints)


def test_unpinned_body_or_missing_support_rejected():
    model = _model()
    root = np.array([0, 0, 0.7, 1, 0, 0, 0], dtype=np.float64)
    joints = np.zeros(29, dtype=np.float64)
    with pytest.raises(ValueError):
        fit_reference_support_pose(
            model,
            root,
            joints,
            np.array([False, False]),
            expected_model_hash=compiled_model_hash(model),
        )
    with pytest.raises(ValueError):
        fit_reference_support_pose(
            model, root, joints, np.array([True, True]), expected_model_hash="sha256:" + "0" * 64
        )
