import mujoco
import numpy as np
import pytest
from test_shared_native_motor_world import source_xml

from rosclaw_soccer.rsi.candidate_motion23_fk_audit import audit_candidate_fk
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def _model():
    xml = source_xml().replace('name="ball"', 'name="soccer_ball"')
    xml = xml.replace("<worldbody>", '<worldbody><geom name="floor" type="plane" size="1 1 .1"/>')
    return mujoco.MjModel.from_xml_string(shared_native_scene(xml, prefixes=("a/",)))


def test_geometry_only_reports_actual_frames_and_never_qualifies():
    model = _model()
    fields = {
        "dof": np.zeros((2, 23), np.float64),
        "root_rot": np.array([[0, 0, 0, 1], [0, 0, 0, 1]], np.float64),
        "root_trans_offset": np.array([[0, 0, 0.5], [0, 0, 0.8]], np.float64),
    }
    result = audit_candidate_fk(model, fields, expected_model_hash=compiled_model_hash(model))
    assert result["source_frames"] == 2
    assert result["root_below_065m_frames"] == 1
    assert result["integrated_native_steps"] == 0
    assert result["physical_execution_qualified"] is False


def test_unknown_model_or_bad_root_state_rejected():
    model = _model()
    fields = {
        "dof": np.zeros((2, 23), np.float64),
        "root_rot": np.array([[0, 0, 0, 1], [0, 0, 0, 1]], np.float64),
        "root_trans_offset": np.zeros((2, 3), np.float64),
    }
    with pytest.raises(ValueError):
        audit_candidate_fk(model, fields, expected_model_hash="sha256:" + "0" * 64)
    fields["root_trans_offset"][0, 0] = np.nan
    with pytest.raises(ValueError):
        audit_candidate_fk(model, fields, expected_model_hash=compiled_model_hash(model))
