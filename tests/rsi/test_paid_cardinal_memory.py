import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_cardinal_memory import (
    CardinalMemoryHead,
    cardinal_certificate,
    cardinal_config,
    fit_cardinal_head,
)
from rosclaw_soccer.rsi.paid_compact_support import CompactTeacherSupport


def dataset():
    x = np.zeros((4, 288), np.float32)
    x[:, 0] = (0, 3, 10, 11)
    y = np.zeros((4, 29), np.float32)
    y[0] = 0.1
    y[1] = -0.1
    return dict(
        features=x,
        target=y,
        active=np.array([True, True, False, False]),
        parent_delta=np.zeros((4, 29), np.float32),
    )


def test_contextual_memory_is_exact_at_anchors_and_preserves_ancestral_inactive():
    z = dataset()
    head, counts = fit_cardinal_head(z)
    assert counts == dict(active_source_rows=2, memory_entries=2, compatible_duplicates_removed=0)
    assert torch.equal(
        head.correction(torch.from_numpy(z["features"][:2])), torch.from_numpy(z["target"][:2])
    )
    prior = torch.full((2, 29), 0.17)
    assert torch.equal(head.combine(torch.from_numpy(z["features"][2:]), prior), prior)
    cert = cardinal_certificate(head, z)
    assert cert["maximum_active_error_rad"] == 0
    assert cert["all_inactive_predictions_exact_parent"]
    assert cert["neural_optimizer_updates"] == 0
    assert not list(head.parameters())
    assert not cardinal_config()["runtime_frame_lookup"]
    assert not cardinal_config()["runtime_teacher"]


def test_inverse_square_interpolation_is_continuous_and_convex_not_a_time_cursor():
    x = torch.zeros(2, 288)
    x[1, 0] = 3
    y = torch.empty(2, 29)
    y[0], y[1] = 0.1, -0.1
    head = CardinalMemoryHead(CompactTeacherSupport(x, torch.ones(2, dtype=torch.float64)), x, y)
    middle = torch.zeros(1, 288)
    middle[0, 0] = 1.5
    assert head.correction(middle).count_nonzero() == 0
    near = x[:1].clone()
    near[0, 0] = 1e-7
    assert torch.allclose(head.correction(near), y[:1], atol=1e-7, rtol=0)
    repeat = head.correction(near)
    head.correction(x[1:])
    assert torch.equal(head.correction(near), repeat)
    out = head.correction(torch.randn(32, 288))
    assert torch.isfinite(out).all() and out.abs().max() <= 0.1


def test_noisy_conflicting_identical_context_labels_are_not_averaged():
    z = dataset()
    z["features"][1] = z["features"][0]
    with pytest.raises(ValueError, match="conflicting"):
        fit_cardinal_head(z)
    z["target"][1] = z["target"][0]
    head, counts = fit_cardinal_head(z)
    assert counts["memory_entries"] == counts["compatible_duplicates_removed"] == 1
    assert cardinal_certificate(head, z)["maximum_active_error_rad"] == 0


def test_final_authority_cannot_expand_and_bad_contexts_fail_closed():
    z = dataset()
    head, _ = fit_cardinal_head(z)
    prior = torch.full((1, 29), 0.19)
    predicted = head.combine(torch.from_numpy(z["features"][:1]), prior)
    assert predicted.abs().max() <= 0.2
    with pytest.raises(ValueError, match="finite"):
        head.correction(torch.full((1, 288), float("nan")))
    with pytest.raises(ValueError, match="bounded"):
        head.combine(torch.zeros(1, 288), torch.full((1, 29), 0.3))


def test_zero_memory_control_does_not_pretend_to_fit_actions():
    z = dataset()
    head, _ = fit_cardinal_head(z, control=True)
    cert = cardinal_certificate(head, z, control=True)
    assert head.coefficients.count_nonzero() == 0
    assert cert["memory_entries_fit"] == 0
    assert not cert["new_data_learning_effect"]
    assert cert["maximum_active_error_rad"] > 0
    with pytest.raises(ValueError, match="typed"):
        cardinal_config(control=1)


def test_dirty_fitter_has_no_data_model_or_physics_side_effect(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_fit_paid_cardinal_memory as runner

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--data-root",
            str(tmp_path),
            "--data-hash",
            "hash",
            "--expected-head",
            "frozen",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="clean frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()
