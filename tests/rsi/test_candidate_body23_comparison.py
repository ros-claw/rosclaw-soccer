import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from test_shared_native_motor_world import source_xml

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.candidate_body23_comparison import compare_candidate_bodies
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def _models():
    root = ET.fromstring(source_xml().replace('name="ball"', 'name="soccer_ball"'))
    ET.SubElement(root.find("worldbody"), "geom", name="floor", type="plane", size="1 1 .1")
    for index, side in ((5, "left"), (11, "right")):
        body = root.find(f".//body[@name='link{index}']")
        body.set("name", side + "_ankle_roll_link")
        for i in range(1, 8):
            ET.SubElement(
                body,
                "geom",
                name=f"{side}_foot{i}_collision",
                type="capsule",
                size=".01 .02",
                mass=".001",
            )
        for _ in range(4):
            ET.SubElement(body, "geom", type="sphere", size=".005", mass=".001")
    original = ET.tostring(root, encoding="unicode")
    canonical = mujoco.MjModel.from_xml_string(shared_native_scene(original, prefixes=("a/",)))
    wrists = set(G1_DDS_JOINT_NAMES[19:22] + G1_DDS_JOINT_NAMES[26:29])
    for parent in root.iter():
        for element in list(parent):
            if element.tag in ("joint", "motor") and element.get("name") in wrists:
                parent.remove(element)
    candidate = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return canonical, candidate


def test_bodies_compare_geometry_only():
    canonical, candidate = _models()
    fields = {
        "dof": np.zeros((2, 23), np.float64),
        "root_rot": np.array([[0, 0, 0, 1], [0, 0, 0, 1]], np.float64),
        "root_trans_offset": np.array([[0, 0, 1], [0, 0, 0.8]], np.float64),
        "contact_mask": np.array([[1, 0], [0, 1]], np.float64),
    }
    result = compare_candidate_bodies(
        canonical,
        candidate,
        fields,
        canonical_hash=compiled_model_hash(canonical),
        candidate_hash=compiled_model_hash(candidate),
    )
    assert result["maximum_ankle_center_difference_m"] == 0
    assert result["claimed_contact_samples"] == 2
    assert result["integrated_native_steps"] == 0
    assert result["source_body_provenance_verified"] is False


def test_unpinned_candidate_model_rejected():
    canonical, candidate = _models()
    with pytest.raises(ValueError, match="pinned"):
        compare_candidate_bodies(
            canonical,
            candidate,
            {},
            canonical_hash=compiled_model_hash(canonical),
            candidate_hash="sha256:" + "0" * 64,
        )
