"""Admission only; stop before model loading or physical output creation."""

import pytest

from scripts import rsi_qualify_velocity_native_branch as runner


class SourceAdmissionReached(Exception):
    pass


def invoke(monkeypatch, flags):
    def stop_before_assets(head):
        raise SourceAdmissionReached

    monkeypatch.setattr("scripts.rsi_native_fresh_common.clean", stop_before_assets)
    monkeypatch.setattr(
        "sys.argv",
        [
            "qualify",
            "--effect-review",
            "unused",
            "--reference-record",
            "unused",
            "--output",
            "not-created",
            "--effect-review-hash",
            "unused",
            "--expected-head",
            "unused",
            *flags,
        ],
    )
    runner.main()


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--trajectory-profile", "--component-audit", "unused", "--component-audit-hash", "unused"],
    ],
)
def test_profiles_stay_separately_explicit(monkeypatch, flags):
    with pytest.raises(SourceAdmissionReached):
        invoke(monkeypatch, flags)


@pytest.mark.parametrize(
    "flags",
    [
        ["--trajectory-profile"],
        ["--trajectory-profile", "--component-audit", "unused"],
        ["--trajectory-profile", "--component-audit-hash", "unused"],
        ["--component-audit", "unused"],
        ["--component-audit-hash", "unused"],
        ["--component-audit", "unused", "--component-audit-hash", "unused"],
    ],
)
def test_no_implicit_multicontrol_or_partial_certificate(monkeypatch, flags):
    with pytest.raises(ValueError):
        invoke(monkeypatch, flags)
