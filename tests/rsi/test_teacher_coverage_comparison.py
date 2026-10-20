"""Synthetic artifact-comparison tests, not native robot qualification."""

import copy
import json

import numpy as np
import pytest

from scripts.rsi_probe_teacher_coverage import compare_case


def fixture(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    old = dict(
        course=dict(course_index=0, course=[0.3, -0.24, 0.0]),
        selected=dict(selected_index=0, score=dict(qualified=True), artifact_hash="old-archive"),
        forecast_native_steps=25200,
        endpoint_native_steps=3000,
        shared_endpoint=True,
    )
    row = copy.deepcopy(old)
    row["course"]["course"] = tuple(row["course"]["course"])
    row["selected"]["artifact_hash"] = "new-archive"
    (b / "complete.json").write_text(json.dumps(old))
    for root in (a, b):
        for name in ["own-prefix20.npz", "selected-actual.npz"] + [
            f"candidate{i}.npz" for i in range(9)
        ]:
            np.savez_compressed(root / name, numerical=np.array([-0.0, 1.0]))
    return a, b, row


def test_same_fields_and_metadata_allow_container_hash_difference(tmp_path):
    a, b, row = fixture(tmp_path)
    compare_case(a, b, row)
    compare_case(a, b, row, original=True)


def test_even_numerically_equal_signed_zero_change_is_rejected(tmp_path):
    a, b, row = fixture(tmp_path)
    np.savez_compressed(a / "selected-actual.npz", numerical=np.array([0.0, 1.0]))
    with pytest.raises(AssertionError):
        compare_case(a, b, row, original=True)


def test_changed_score_or_option_never_hidden_by_matching_arrays(tmp_path):
    a, b, row = fixture(tmp_path)
    row["selected"]["score"]["qualified"] = False
    with pytest.raises(ValueError):
        compare_case(a, b, row)
    row["selected"]["score"]["qualified"] = True
    row["selected"]["selected_index"] = 1
    with pytest.raises(ValueError):
        compare_case(a, b, row)


def test_missing_forecast_not_silently_ignored(tmp_path):
    a, b, row = fixture(tmp_path)
    (a / "candidate8.npz").unlink()
    with pytest.raises(FileNotFoundError):
        compare_case(a, b, row)
