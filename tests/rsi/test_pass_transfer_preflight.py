import copy
import json

import pytest

from scripts.rsi_audit_pass_transfer import verify_json_preflight


def declaration():
    return dict(
        initial_conditions=8,
        worlds=32,
        course_binding="sha256:binding",
        native_steps=0,
        ball_dimensions=dict(
            radius_m=0.11,
            body_mass_kg=0.43,
            principal_inertia_kg_m2=(0.0020812, 0.0020812, 0.0020812),
        ),
    )


def test_exact_json_roundtrip_of_tuple_inertia_is_not_physical_mismatch():
    expected = declaration()
    recorded = json.loads(json.dumps(expected))
    assert expected != recorded
    verify_json_preflight(expected, recorded)


@pytest.mark.parametrize("fault", ["radius", "mass", "inertia", "binding", "steps", "states"])
def test_actual_numeric_or_domain_mismatch_still_rejected(fault):
    expected = declaration()
    recorded = json.loads(json.dumps(copy.deepcopy(expected)))
    if fault in ("radius", "mass", "inertia"):
        key = dict(radius="radius_m", mass="body_mass_kg", inertia="principal_inertia_kg_m2")[fault]
        recorded["ball_dimensions"][key] = 999
    else:
        key = dict(binding="course_binding", steps="native_steps", states="initial_conditions")[
            fault
        ]
        recorded[key] = 999
    with pytest.raises(ValueError, match="physical conditions"):
        verify_json_preflight(expected, recorded)
