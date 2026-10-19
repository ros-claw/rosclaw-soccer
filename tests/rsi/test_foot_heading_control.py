"""Values-only request checks; no simulator allocation or execution permission."""

import argparse
import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from rosclaw_soccer.rsi import native_first_touch_episode as native
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe, require_heading_request
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
from rosclaw_soccer.rsi.planner_navigation_pulse import PlannerNavigationPulse
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_swing_release_default_arithmetic import OffPath


@pytest.mark.parametrize("enabled", [False, True])
def test_heading_contract_is_exact_frozen_and_not_an_authority_grant(enabled):
    probe = FootHeadingContactProbe(enabled)
    c = probe.contract()
    assert c["contract_hash"] == hash_json({k: v for k, v in c.items() if k != "contract_hash"})
    assert c["parameters"] == dict(enabled=enabled)
    assert c["first_possible_frame"] == 30
    assert c["original_positive_ik_planner_pulse_required"]
    assert c["mesh_contact_point_not_position_constrained"]
    assert c["original_six_channel_contact_gate_unchanged"]
    assert c["dense_unmonitored_contact_review_required"]
    assert "first_any_robot_contact_ends_heading_updates" not in c
    assert c["source_hash"] != c["numerical_source_hash"]
    for k in (
        "trained_policy",
        "physics_qualified",
        "runtime_execution_authorized",
        "training_authorized",
        "hardware_authorized",
        "promotion_authorized",
    ):
        assert c[k] is False
    with pytest.raises(FrozenInstanceError):
        probe.enabled = not enabled


@pytest.mark.parametrize("value", [0, 1, None, "true", [], {}])
def test_truthy_flags_are_not_explicit_probe_parameters(value):
    with pytest.raises(ValueError):
        FootHeadingContactProbe(value)


def test_derived_types_do_not_open_an_alternative_control_implementation():
    class Untrusted(FootHeadingContactProbe):
        pass

    with pytest.raises(ValueError):
        Untrusted(True)


@pytest.mark.parametrize("enabled", [False, True])
def test_heading_requires_full_original_native_boundary_before_output_or_allocation(
    tmp_path, enabled
):
    with pytest.raises(ValueError, match="original-IK planner SIM request"):
        native.run_native_first_touch_episode(
            argparse.Namespace(output_root=tmp_path / "unused"),
            course=(2.65, 0.1, 0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            control_profile=BoundedFirstTouchControl(),
            record_default_planner_events=True,
            original_ik_planner_pulse=OriginalIkPlannerPulse(0.08),
            foot_heading_control=FootHeadingContactProbe(enabled),
        )
    assert not (tmp_path / "unused").exists()


def test_default_enabled_heading_cannot_be_hidden_by_off_path_projection():
    f = next(
        n
        for n in ast.parse(Path(native.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_native_first_touch_episode"
    )
    i = next(i for i, a in enumerate(f.args.kwonlyargs) if a.arg == "foot_heading_control")
    f.args.kw_defaults[i] = ast.Dict(keys=[], values=[])
    with pytest.raises(AssertionError):
        OffPath().visit(f)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "zero_pulse",
        "legacy_pulse",
        "reference",
        "profile",
        "capture",
        "untyped",
        "tampered_flag",
    ],
)
def test_both_arms_bind_the_same_positive_pulse_original_ik_and_fixed_b2_profile(enabled, mutation):
    probe, pulse, model, profile, record = (
        FootHeadingContactProbe(enabled),
        OriginalIkPlannerPulse(0.08),
        None,
        BoundedFirstTouchControl(),
        True,
    )
    if mutation == "zero_pulse":
        pulse = OriginalIkPlannerPulse(0.0)
    elif mutation == "legacy_pulse":
        pulse = PlannerNavigationPulse(0.08)
    elif mutation == "reference":
        model = {"model_hash": "not-allowed"}
    elif mutation == "profile":
        profile = BoundedFirstTouchControl(lateral_offset_m=0.01)
    elif mutation == "capture":
        record = False
    elif mutation == "untyped":
        probe = {"enabled": enabled}
    elif mutation == "tampered_flag":
        object.__setattr__(probe, "enabled", 1)
    if mutation is None:
        assert (
            require_heading_request(
                probe,
                pulse=pulse,
                reference_model=model,
                profile=profile,
                record_planner_events=record,
            )
            is None
        )
    else:
        with pytest.raises(ValueError):
            require_heading_request(
                probe,
                pulse=pulse,
                reference_model=model,
                profile=profile,
                record_planner_events=record,
            )
