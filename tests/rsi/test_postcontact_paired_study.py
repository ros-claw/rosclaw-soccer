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


def closing_declaration():
    config = declaration()
    del config["postcontact_probe"]
    config.update(
        postcontact_study_profile=study.CLOSING_PROFILE,
        postcontact_study_groups=study.CLOSING_GROUPS.copy(),
        closing_gain_probe_s_inv=3.0,
        max_attempts=24,
    )
    return config


def test_closing_changes_only_gain_not_lateral_swing_limits_or_postcontact():
    from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl

    config = closing_declaration()
    baseline = study.paired_candidate(config, "baseline")
    probe = study.paired_candidate(config, "probe")
    assert set(baseline) == set(probe) == {"parameters", "swing"}
    assert baseline["swing"] == probe["swing"]
    changed = [
        k for k in baseline["parameters"] if baseline["parameters"][k] != probe["parameters"][k]
    ]
    assert changed == ["closing_gain_s_inv"]
    a = BoundedFirstTouchControl(**baseline["parameters"])
    b = BoundedFirstTouchControl(**probe["parameters"])
    for gap in (0.2, 0.5, 1.0, 2.8):
        for before in (True, False):
            kwargs = dict(gap_x=gap, gap_y=0.1, ball_vx=0.6, before_contact=before)
            assert a.navigation(**kwargs)[1][1:] == b.navigation(**kwargs)[1][1:]
            if not before:
                assert a.navigation(**kwargs) == b.navigation(**kwargs)


@pytest.mark.parametrize(
    "key,value",
    [
        ("postcontact_probe", {}),
        ("closing_gain_probe_s_inv", 3),
        ("closing_gain_probe_s_inv", 2.0),
        ("postcontact_study_groups", [28, 44]),
        ("max_attempts", 100),
        ("neural_training_authorized", True),
    ],
)
def test_closing_fixed_factor_coverage_and_budget_cannot_drift(key, value):
    config = closing_declaration()
    config[key] = value
    with pytest.raises(ValueError):
        require_declaration(config)


def test_initial_pair_requires_all_initial_physical_fields_but_not_identical_commands(tmp_path):
    keys = (
        "root_pose_xyzw_m",
        "root_velocity_world",
        "ball_position_before_step_m",
        "ball_linear_velocity_before_step_m_s",
        "foot_geometry_position_before_step_m",
        "joint_position_rad",
        "joint_velocity_rad_s",
        "canonical_qpos",
        "canonical_qvel",
    )
    a = {key: np.zeros((300, 1, 3)) for key in keys}
    b = {key: value.copy() for key, value in a.items()}
    b["navigation_command"] = np.ones((300, 1, 3))
    b["canonical_qpos"][1:] = 1.0
    pa, pb = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(pa, **a)
    np.savez(pb, **b)
    ra, rb = (dict(witnesses=dict(trace=dict(path=p))) for p in (pa, pb))
    study.verify_initial_pair(ra, rb)
    b["canonical_qpos"][0, 0, 0] = 0.01
    np.savez(pb, **b)
    with pytest.raises(ValueError, match="initial paired physical state"):
        study.verify_initial_pair(ra, rb)


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
