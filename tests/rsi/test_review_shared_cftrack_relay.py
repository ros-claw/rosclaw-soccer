import numpy as np
import pytest

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_review_shared_cftrack_relay import compare_prefix, proximity_readback
from scripts.rsi_run_shared_cftrack_relay import validate_replay_binding


def test_physical_prefix_requires_exact_common_history():
    x = {"qpos": np.array([[0], [1], [2]])}
    y = {"qpos": np.array([[0], [1], [3]])}
    compare_prefix(x, y, ("qpos",), 2)
    with pytest.raises(ValueError, match="prefix differs"):
        compare_prefix(x, y, ("qpos",), 3)
    with pytest.raises(ValueError, match="complete named"):
        compare_prefix(x, y, ("qvel",), 1)


def test_proximity_uses_observed_substeps_and_does_not_imply_possession():
    q = np.array([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 1.0, 0.0]])
    v = np.array([[0.0, 0.0, 1.0, 0.0], [0.1, 0.0, 0.4, 0.0], [0.0, 0.0, 0.2, 0.0]])
    result = proximity_readback(q, v, ball_q=2, ball_v=2, root_q=0, root_v=0, first=0)
    assert result["closest_native_substep"] == 1
    assert result["closest_root_ball_planar_m"] == 0.5
    assert result["relative_planar_speed_at_closest_mps"] == pytest.approx(0.3)
    assert result["final_root_ball_planar_m"] == 1.0
    assert result["possession_proven"] is False
    tail = proximity_readback(q, v, ball_q=2, ball_v=2, root_q=0, root_v=0, first=2)
    assert tail["closest_native_substep"] == 2
    with pytest.raises(ValueError):
        proximity_readback(q, v, ball_q=2, ball_v=2, root_q=0, root_v=0, first=3)
    q[0, 0] = np.nan
    with pytest.raises(ValueError):
        proximity_readback(q, v, ball_q=2, ball_v=2, root_q=0, root_v=0, first=0)


def test_replay_rejects_unbound_or_mutated_source_scene(tmp_path):
    import json

    declaration = tmp_path / "declaration.json"
    scene = tmp_path / "scene.xml"
    pins = {"/frozen/asset": "sha256:abc"}
    declaration.write_text(
        json.dumps(dict(input_file_hashes=pins, source_head="head", external_head="external"))
    )
    scene.write_text("<mujoco/>")
    receipt = dict(declaration_hash=digest(declaration), scene_hash=digest(scene))
    validate_replay_binding(tmp_path, receipt, pins, source_head="head", external_head="external")
    with pytest.raises(ValueError, match="exact original"):
        validate_replay_binding(
            tmp_path,
            receipt,
            pins,
            source_head="head",
            external_head="external",
            early_cftrack=True,
        )
    with pytest.raises(ValueError, match="exact original"):
        validate_replay_binding(
            tmp_path,
            receipt,
            pins,
            source_head="head",
            external_head="external",
            protocol_hash="sha256:unexpected",
        )
    with pytest.raises(ValueError, match="bound"):
        validate_replay_binding(tmp_path, {}, pins, source_head="head", external_head="external")
    scene.write_text("<mujoco model='changed'/>")
    with pytest.raises(ValueError, match="bound"):
        validate_replay_binding(
            tmp_path, receipt, pins, source_head="head", external_head="external"
        )
