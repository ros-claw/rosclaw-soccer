"""Summary admission/timing fixtures, not rollout evidence."""

import sys

import pytest

from scripts import rsi_summarize_close_contact_robustness as summary


@pytest.mark.parametrize(
    "handoff,native,before,without",
    [
        (None, None, False, False),
        (None, 359, False, True),
        (68, 660, True, False),
        (68, 680, False, False),
        (68, 2260, False, False),
    ],
)
def test_measured_contact_timing_not_causal_overclaim(handoff, native, before, without):
    result = summary.contact_timing(
        dict(
            handoff_frame=handoff,
            strike=dict(
                first_foot_contact_native_index=native,
                first_nonfoot_robot_contact_native_index=native,
            ),
        )
    )
    for kind in ("foot", "nonfoot_robot"):
        assert result[f"{kind}_contact_before_handoff"] is before
        assert result[f"{kind}_contact_without_handoff"] is without


@pytest.mark.parametrize("kind", ["source", "used_output", "missing_cluster"])
def test_summary_requires_sealed_source_unused_output_all_four(tmp_path, monkeypatch, kind):
    arguments = [
        "summary",
        "--expected-head",
        "expected",
        "--expected-producer-head",
        "producer",
        "--expected-review-head",
        "review",
        "--output",
        str(tmp_path / "out"),
    ]
    for i in range(3 if kind == "missing_cluster" else 4):
        arguments += ["--shard", str(tmp_path / f"p{i}"), str(tmp_path / f"r{i}")]
    if kind == "used_output":
        (tmp_path / "out").mkdir()
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        summary.subprocess,
        "check_output",
        lambda *a, **k: "wrong\n" if kind == "source" else "expected\n",
    )
    with pytest.raises(ValueError, match="sealed summary source"):
        summary.main()
