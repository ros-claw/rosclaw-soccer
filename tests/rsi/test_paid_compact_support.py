import pytest
import torch

from rosclaw_soccer.rsi.paid_compact_support import (
    CompactSupportedHead,
    CompactTeacherSupport,
    nearest_protected_radii,
)


def fixture():
    positive = torch.zeros(2, 288)
    positive[1, 0] = 2
    negative = torch.zeros(3, 288)
    negative[:, 0] = torch.tensor([1.0, 3.0, -1.0])
    return positive, negative


def test_support_opens_all_teachers_and_exactly_rejects_all_protected_anchors():
    p, n = fixture()
    radii = nearest_protected_radii(p, n)
    assert torch.equal(radii, torch.full((2,), 0.45, dtype=torch.float64))
    s = CompactTeacherSupport(p, radii)
    assert torch.equal(s(p), torch.ones(2, 1))
    assert torch.equal(s(n), torch.zeros(3, 1))
    assert not list(s.parameters())
    assert torch.equal(s(torch.full((1, 288), 1e3)), torch.zeros(1, 1))


def test_support_is_continuous_within_teacher_neighbourhood_not_exact_state_lookup():
    p, n = fixture()
    s = CompactTeacherSupport(p, nearest_protected_radii(p, n))
    q = p[:1].clone()
    q[0, 0] += 0.225
    assert float(s(q)) == pytest.approx(0.5)
    assert torch.equal(s(torch.cat([q, p])), torch.cat([s(q), s(p)]))


def test_no_dead_gradient_at_teacher_and_exact_old_action_after_actual_update():
    p, n = fixture()
    m = CompactSupportedHead(CompactTeacherSupport(p, nearest_protected_radii(p, n)))
    prior = torch.full((2, 29), 0.01)
    before = m.net[-1].weight.detach().clone()
    optimizer = torch.optim.Adam(m.net.parameters(), lr=0.001)
    optimizer.zero_grad()
    (m.combine(p, prior) - 0.1).square().mean().backward()
    assert m.net[-1].weight.grad.abs().max() > 0
    optimizer.step()
    assert not torch.equal(before, m.net[-1].weight)
    old = torch.full((3, 29), -0.02)
    assert torch.equal(m.combine(n, old), old)
    assert m.combine(p, prior).abs().max() <= 0.2


def test_collision_fails_closed_instead_of_inventing_a_radius():
    p, _ = fixture()
    with pytest.raises(ValueError, match="collision"):
        nearest_protected_radii(p, p)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_radii_rejected(bad):
    p, _ = fixture()
    with pytest.raises(ValueError):
        CompactTeacherSupport(p, torch.full((2,), bad, dtype=torch.float64))


def test_compact_support_does_not_alias_input_memory():
    p, n = fixture()
    radii = nearest_protected_radii(p, n)
    s = CompactTeacherSupport(p, radii)
    p.fill_(42)
    radii.fill_(42)
    assert float(s.centres[0, 0]) == 0
    assert float(s.radii[0]) == 0.45


def test_inference_rejects_nonfinite_and_wrong_feature_shapes():
    p, n = fixture()
    s = CompactTeacherSupport(p, nearest_protected_radii(p, n))
    for x in (p.double(), p[:, :4], torch.full_like(p, float("nan")), p[:0]):
        with pytest.raises(ValueError):
            s(x)


def test_dirty_learner_rejected_before_output_or_numerical_work(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_train_paid_compact_support as runner

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--data-root",
            str(tmp_path / "data"),
            "--data-hash",
            "hash",
            "--expected-head",
            "frozen",
            "--seed",
            "0",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr("scripts.rsi_native_context_recovery_common.digest", lambda _: "hash")
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()
