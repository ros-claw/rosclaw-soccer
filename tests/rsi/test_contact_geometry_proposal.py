"""Geometric proposal fixtures; no dynamic contact, NN or teacher claims."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_geometry_proposal import contact_geometry_proposal


def fixture():
    q = np.zeros((300, 43), dtype=np.float64)
    q[:, 2] = 0.8
    q[:, 3] = 1
    q[:, 36:39] = [2, 0.02, 0.11]
    v = np.zeros((300, 41), dtype=np.float64)
    toe = np.zeros((300, 3), dtype=np.float64)
    toe[:, 0] = 0.01 * np.arange(300)
    toe[:, 2] = 0.11
    force = np.zeros((300, 10), dtype=np.float64)
    return q, v, toe, force


def propose(data, handoff=50, radius=0.11):
    return contact_geometry_proposal(*data, handoff_frame=handoff, ball_radius_m=radius)


def test_one_deterministic_owned_geometry_request_no_input_mutation():
    data = fixture()
    copies = [a.copy() for a in data]
    result = propose(data)
    assert result == propose(data)
    assert result["applicable"]
    frame = result["selected_global_frame"]
    assert frame in (188, 189)
    assert result["selected_reference_local_frame"] == frame - 50
    np.testing.assert_allclose(result["world_residual_m"], [1.885 - 0.01 * frame, 0.02, 0.0])
    for actual, old in zip(data, copies, strict=True):
        np.testing.assert_array_equal(actual, old)
    result["offset_root_m"][0] = 999
    assert propose(data)["offset_root_m"][0] != 999
    assert result["native_steps"] == result["neural_inferences"] == 0
    assert not result["physics_qualified"] and not result["promotion_authorized"]


def test_actual_root_rotation_transforms_vector_not_ball_or_world():
    data = fixture()
    original = propose(data)
    data[0][:, 3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    rotated = propose(data)
    x, y, z = original["offset_root_m"]
    np.testing.assert_allclose(rotated["offset_root_m"], [y, -x, z], atol=1e-14)


def test_one_explicit_trust_region_projection_not_state_clipping():
    data = fixture()
    data[0][:, 37] = 10
    result = propose(data)
    assert result["applicable"] and result["proposal_projected"]
    assert np.max(abs(np.asarray(result["offset_root_m"]))) <= 0.04
    assert np.linalg.norm(result["offset_root_m"]) <= 0.06
    assert np.all(data[0][:, 37] == 10)


@pytest.mark.parametrize("kind", ["no_handoff", "stationary", "ball_faster", "nonfoot"])
def test_explicit_inapplicability_no_fabricated_offset(kind):
    data = fixture()
    handoff = 50
    if kind == "no_handoff":
        handoff = None
    elif kind == "stationary":
        data[2][:, 0] = 1
    elif kind == "ball_faster":
        data[1][:, 35] = 1.0
    else:
        data[3][0, 0] = 1.001
    result = propose(data, handoff=handoff)
    assert not result["applicable"] and result["offset_root_m"] is None


@pytest.mark.parametrize("kind", ["nan", "float32", "shape", "quat", "force", "entry", "ball"])
def test_invalid_inputs_rejected_not_sanitized(kind):
    data = list(fixture())
    handoff, radius = 50, 0.11
    if kind == "nan":
        data[0][0, 0] = np.nan
    elif kind == "float32":
        data[2] = data[2].astype(np.float32)
    elif kind == "shape":
        data[1] = data[1][20:]
    elif kind == "quat":
        data[0][0, 3] = 0
    elif kind == "force":
        data[3][0, 0] = -1
    elif kind == "entry":
        handoff = True
    else:
        radius = 0.10
    with pytest.raises(ValueError, match="full300"):
        propose(data, handoff, radius)
