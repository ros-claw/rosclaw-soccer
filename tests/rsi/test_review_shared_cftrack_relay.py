import numpy as np
import pytest

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_review_shared_cftrack_relay import compare_prefix
from scripts.rsi_run_shared_cftrack_relay import validate_replay_binding


def test_physical_prefix_requires_exact_common_history():
    x = {"qpos": np.array([[0], [1], [2]])}
    y = {"qpos": np.array([[0], [1], [3]])}
    compare_prefix(x, y, ("qpos",), 2)
    with pytest.raises(ValueError, match="prefix differs"):
        compare_prefix(x, y, ("qpos",), 3)
    with pytest.raises(ValueError, match="complete named"):
        compare_prefix(x, y, ("qvel",), 1)


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
