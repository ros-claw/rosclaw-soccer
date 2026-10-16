from dataclasses import asdict

import numpy as np
import pytest

from rosclaw_soccer.rsi import postcontact_paired_study as study
from rosclaw_soccer.rsi.postcontact_navigation_probe import PostcontactNavigationProbe
from rosclaw_soccer.rsi.postcontact_paired_study import require_declaration, verify_precontact_pair


def declaration():
    return dict(
        postcontact_study_groups=[4, 12, 32, 52],
        max_attempts=10,
        neural_training_authorized=False,
        teacher_auto_admission=False,
        postcontact_probe=asdict(PostcontactNavigationProbe()),
    )


def test_exact_failure_directed_plan_exercises_actual_interface(monkeypatch):
    calls = []
    original = PostcontactNavigationProbe.navigation

    def measured(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(PostcontactNavigationProbe, "navigation", measured)
    require_declaration(declaration())
    assert len(calls) == 12


def test_balanced_retention_profile_requires_all_sixteen_states_and_fixed_budget():
    config = declaration()
    config.update(
        postcontact_study_profile="balanced_public_sixteen_retention",
        postcontact_study_groups=list(range(0, 64, 4)),
        max_attempts=34,
    )
    require_declaration(config)
    config["postcontact_study_groups"] = [4, 12, 32, 52]
    with pytest.raises(ValueError):
        require_declaration(config)
    config["postcontact_study_groups"] = list(range(0, 64, 4))
    config["max_attempts"] = 1000
    with pytest.raises(ValueError):
        require_declaration(config)


@pytest.mark.parametrize("profile", [None, True, [], "fresh", "unknown"])
def test_unknown_study_profile_cannot_change_scope(profile):
    config = declaration()
    config["postcontact_study_profile"] = profile
    with pytest.raises(ValueError):
        require_declaration(config)


@pytest.mark.parametrize("gain,changed", [(True, False), (False, False), (True, True)])
def test_balanced_preflight_binds_qualified_prior_without_neural_allocation(
    tmp_path, monkeypatch, gain, changed
):
    config = declaration()
    prior_path = tmp_path / "prior.json"
    config.update(
        postcontact_study_profile="balanced_public_sixteen_retention",
        postcontact_study_groups=list(range(0, 64, 4)),
        max_attempts=34,
        prior_study_report_path=str(prior_path),
        prior_study_report_hash="fixture-seal",
    )
    prior = dict(
        schema="soccer.rsi.qualified_postcontact_paired_study.v1",
        report_hash="fixture-seal",
        completed_pairs=4,
        source_physical_executions=8,
        source_independent_physical_substeps=24000,
        source_foundation_recomputed_calls=2400,
        source_sampling_draws_reconstructed=2160,
        baseline_high_quality=0,
        probe_high_quality=int(gain),
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    pins = {}
    digest_reads = []

    class Preparation:
        def sealed(self, path):
            assert path == prior_path
            return prior

        def digest(self, path):
            assert path == prior_path
            digest_reads.append(path)
            if changed and len(digest_reads) > 1:
                return "changed-file-hash"
            return "fixture-file-hash"

        def add_pins(self, value):
            pins.update(value)

    monkeypatch.setattr(study, "prepare", lambda _: (Preparation(), None, None, config))
    if gain and not changed:
        study.run(tmp_path / "config.json", preflight=True)
        assert pins == {str(prior_path): "fixture-file-hash"}
    else:
        with pytest.raises(ValueError, match="qualified SIM pilot"):
            study.run(tmp_path / "config.json", preflight=True)


@pytest.mark.parametrize(
    "key,value",
    [
        ("postcontact_study_groups", [4, 12]),
        ("max_attempts", 100),
        ("neural_training_authorized", True),
        ("teacher_auto_admission", True),
        ("predecessor", {}),
        ("postcontact_probe", {}),
    ],
)
def test_no_silent_plan_expansion_or_teacher_activation(key, value):
    config = declaration()
    config[key] = value
    with pytest.raises(ValueError):
        require_declaration(config)


@pytest.mark.parametrize("first", [None, 0, 72, 299])
def test_prefix_checks_through_contact_frame_and_all_fields(tmp_path, first):
    a = dict(force_n=np.zeros((300, 1, 6)), latent=np.zeros((300, 1, 2)))
    if first is not None:
        a["force_n"][first, 0, 0] = 10.0
    b = {k: v.copy() for k, v in a.items()}
    pa, pb = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(pa, **a)
    receipt_a = dict(
        outcome=dict(first_foot_contact_frame=first), witnesses=dict(trace=dict(path=pa))
    )
    receipt_b = dict(
        outcome=dict(first_foot_contact_frame=first), witnesses=dict(trace=dict(path=pb))
    )
    if first is not None and first < 299:
        b["latent"][first + 1 :] = 1.0
    np.savez(pb, **b)
    verify_precontact_pair(receipt_a, receipt_b)
    bad_frame = 299 if first is None else first
    b["latent"][bad_frame, 0, 0] = 2.0
    np.savez(pb, **b)
    with pytest.raises(ValueError, match="pre-intervention"):
        verify_precontact_pair(receipt_a, receipt_b)


def test_nonfoot_contact_is_actual_intervention_boundary(tmp_path):
    a = dict(force_n=np.zeros((300, 1, 6)), latent=np.zeros((300, 1, 2)))
    a["force_n"][20, 0, 4] = 5.0
    a["force_n"][40, 0, 0] = 5.0
    b = {k: v.copy() for k, v in a.items()}
    b["latent"][21:] = 1.0
    pa, pb = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(pa, **a)
    np.savez(pb, **b)
    verify_precontact_pair(
        dict(outcome=dict(first_foot_contact_frame=40), witnesses=dict(trace=dict(path=pa))),
        dict(outcome=dict(first_foot_contact_frame=None), witnesses=dict(trace=dict(path=pb))),
    )
