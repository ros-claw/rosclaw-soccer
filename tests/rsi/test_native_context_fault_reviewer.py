"""Reviewer bootstrap and literal feed fixtures, no physics or real ONNX."""

import sys

import numpy as np
import pytest

from scripts import rsi_review_native_context_faults as reviewer


def arguments(tmp_path):
    result = ["reader"]
    for key in (
        "inventory",
        "bank",
        "actor-source",
        "world",
        "model-root",
        "core-source",
        "output",
    ):
        result += ["--" + key, str(tmp_path / key)]
    return result + [
        "--expected-head",
        "expected",
        "--expected-actor-head",
        "actor",
        "--expected-inventory-hash",
        "sha256:" + "1" * 64,
    ]


def test_source_guard_before_helper_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(reviewer.subprocess, "check_output", lambda *a, **k: "wrong\n")
    with pytest.raises(ValueError, match="sealed review source"):
        reviewer.main()


def test_dirty_bootstrap_refused_before_inventory_or_physics(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(
        reviewer.subprocess,
        "check_output",
        lambda cmd, **k: "expected\n" if cmd[1] == "rev-parse" else b"not the script bytes",
    )
    with pytest.raises(ValueError, match="sealed Git blob"):
        reviewer.main()


def test_literal_model_feed_has_original_shapes_and_types():
    q = np.zeros(43)
    q[3] = 1
    context = np.repeat(q[:36][None], 4, axis=0)
    feed = reviewer.literal_feed(np, context, q, (0.3, -0.4, 0.0), 30301)
    assert set(feed) == {
        "context_mujoco_qpos",
        "target_vel",
        "mode",
        "movement_direction",
        "facing_direction",
        "random_seed",
        "has_specific_target",
        "specific_target_positions",
        "specific_target_headings",
        "allowed_pred_num_tokens",
        "height",
    }
    np.testing.assert_array_equal(feed["context_mujoco_qpos"], context[None].astype(np.float32))
    np.testing.assert_allclose(feed["movement_direction"], [[0.6, -0.8, 0.0]], atol=1e-7)
    assert feed["target_vel"][0] == 0.5
    assert feed["mode"][0] == 1 and feed["random_seed"][0] == 30301
    assert feed["context_mujoco_qpos"].dtype == np.float32
    assert feed["allowed_pred_num_tokens"].dtype == np.int64
