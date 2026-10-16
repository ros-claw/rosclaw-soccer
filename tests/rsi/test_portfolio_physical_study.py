import copy
import json
from dataclasses import asdict

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_outcome import outcome_contract
from rosclaw_soccer.rsi.initial_context_control_portfolio import (
    ControlCase,
    ControlOutcome,
    InitialContextControlPortfolio,
)
from rosclaw_soccer.rsi.portfolio_physical_study import (
    GROUPS,
    PROFILE,
    PreparedPortfolio,
    logical_document,
    prefix_query,
    publish_selection,
    validate_option,
    verify_selected_receipt,
)
from rosclaw_soccer.rsi.postcontact_paired_study import require_declaration
from rosclaw_soccer.sim.contracts import hash_json


def fixture():
    trace = {
        name: np.zeros((3, 1, size))
        for name, size in (
            ("canonical_qpos", 43),
            ("canonical_qvel", 41),
            ("root_pose_xyzw_m", 7),
            ("ball_position_before_step_m", 3),
            ("ball_linear_velocity_before_step_m_s", 3),
        )
    }
    trace["ball_position_before_step_m"][:, 0, 0] = 2.2
    trace["canonical_qpos"][:, 0, :7] = (0, 0, 0.793, 1, 0, 0, 0)
    trace["canonical_qpos"][:, 0, 36:] = (2.2, 0, 0.13, 1, 0, 0, 0)
    trace["root_pose_xyzw_m"][:, 0] = (0, 0, 0.793, 0, 0, 0, 1)
    trace["ball_position_before_step_m"][:, 0, 2] = 0.13
    commit = dict(
        recurrent_sampling_factory=dict(complete_canonical_mean_hash=hash_json("mean")),
        step_model_hash=hash_json("view"),
        compiled_model_hash=hash_json("world"),
        foundation_hash=hash_json("foundation"),
        late_actor_hash=hash_json("late"),
        physics={},
    )
    law = dict(schema="soccer.rsi.recurrent_motor_sampling.v1", std_raw=0.1, rho=0.9)
    view = dict(
        law,
        mean_model={"$rosclaw_shared_payload": hash_json("mean")},
        model_hash=commit["step_model_hash"],
        seed=1,
    )
    binding = dict(
        initial_robot_qpos=trace["canonical_qpos"][0, 0, :36].tolist(),
        initial_robot_qvel=[0.0] * 35,
        world_hash=commit["compiled_model_hash"],
        canonical_mean_hash=hash_json("mean"),
        foundation_hash=commit["foundation_hash"],
        late_actor_hash=commit["late_actor_hash"],
        physics={},
        sampling_law=law,
        outcome_contract_hash=outcome_contract()["contract_hash"],
    )
    options = (
        None,
        dict(parameters=asdict(BoundedFirstTouchControl()), swing="bounded_contact_diagnostic"),
    )
    cases = tuple(
        ControlCase(
            (2.2 + i * 0.1, 0.0, 0.0),
            (ControlOutcome(True, False, -3.0), ControlOutcome(True, True, 3.0)),
            tuple(hash_json([i, j]) for j in range(2)),
        )
        for i in range(4)
    )
    model = InitialContextControlPortfolio.fit(
        cases=cases,
        option_hashes=tuple(hash_json(c) for c in options),
        execution_binding_hash=hash_json(binding),
    )
    return model, options, trace, commit, view, binding


def envelope(document):
    value = dict(
        schema="rosclaw.growth.shared_json_payload.v1",
        stripped_document=document,
        payload_hash=hash_json("mean"),
        logical_document_hash=hash_json("logical"),
        location=["mean_model"],
    )
    return dict(value, envelope_hash=hash_json(value))


def test_prefix_only_and_no_future_outcome_input():
    model, options, trace, commit, view, binding = fixture()
    query = prefix_query(model, options, trace, commit, view, expected_binding=binding)
    assert query["model_query"]["option_index"] == 1
    for value in trace.values():
        value[1:] = np.nan
    trace["future_labels"] = object()
    assert prefix_query(model, options, trace, commit, view, expected_binding=binding) == query
    query["candidate"]["parameters"]["forward_cap_m_s"] = 0.8
    assert options[1]["parameters"]["forward_cap_m_s"] == 1.4


@pytest.mark.parametrize(
    "change", ["sampling", "parent", "world", "robot", "unsupported", "option"]
)
def test_wrong_binding_support_or_control_rejected_before_execution(change):
    model, options, trace, commit, view, binding = fixture()
    if change == "sampling":
        view["rho"] = 0.8
    elif change == "parent":
        commit["foundation_hash"] = hash_json("other")
    elif change == "world":
        commit["compiled_model_hash"] = hash_json("other")
    elif change == "robot":
        trace["canonical_qpos"][0, 0, 0] = 0.01
    elif change == "unsupported":
        trace["ball_position_before_step_m"][0, 0, 0] = 9.0
    else:
        options = copy.deepcopy(options)
        options[1]["parameters"]["forward_cap_m_s"] = 0.8
    with pytest.raises(ValueError):
        prefix_query(model, options, trace, commit, view, expected_binding=binding)


def test_option_domain_and_owned_copy():
    assert validate_option(None) is None
    candidate = fixture()[1][1]
    result = validate_option(candidate)
    result["parameters"]["forward_cap_m_s"] = 0.8
    assert candidate["parameters"]["forward_cap_m_s"] == 1.4
    for value in (
        {"torque": 1.0},
        dict(candidate, hardware_authorized=True),
        dict(candidate, postcontact={}),
    ):
        with pytest.raises(ValueError):
            validate_option(value)


@pytest.mark.parametrize("change", ["height", "spin", "lateral_velocity", "orientation"])
def test_unmodelled_ball_initial_conditions_cannot_enter_native_portfolio(change):
    model, options, trace, commit, view, binding = fixture()
    if change == "height":
        trace["canonical_qpos"][0, 0, 38] = 0.3
        trace["ball_position_before_step_m"][0, 0, 2] = 0.3
    elif change == "spin":
        trace["canonical_qvel"][0, 0, 39] = 1.0
    elif change == "lateral_velocity":
        trace["canonical_qvel"][0, 0, 36] = 0.1
        trace["ball_linear_velocity_before_step_m_s"][0, 0, 1] = 0.1
    else:
        trace["canonical_qpos"][0, 0, 39:43] = (0, 0, 0, 1)
    with pytest.raises(ValueError, match="initial canonical"):
        prefix_query(model, options, trace, commit, view, expected_binding=binding)


def test_selection_immutable_and_rechecked_against_actual_initial_state(tmp_path):
    model, options, trace, commit, view, binding = fixture()
    selection = dict(
        prefix_query(model, options, trace, commit, view, expected_binding=binding), group=0
    )
    portfolio = PreparedPortfolio(model, options, hash_json("fit"), binding, {0: selection})
    folder = tmp_path / "attempt0"
    witness = publish_selection(folder, selection)
    np.savez(folder / "physical_trace.npz", **trace)
    (folder / "commitment.json").write_text(json.dumps(commit))
    (folder / "report.json").write_text(
        json.dumps(envelope(dict(executed_motor_policy=dict(step_motor_proof=dict(model=view)))))
    )
    receipt = dict(
        candidate=selection["candidate"],
        group=0,
        portfolio_selection_witness=witness,
        witnesses={
            k: dict(path=str(folder / name))
            for k, name in (
                ("trace", "physical_trace.npz"),
                ("commitment", "commitment.json"),
                ("report", "report.json"),
            )
        },
    )
    verify_selected_receipt(receipt, selection, portfolio, tmp_path)
    changed = dict(selection, group=4)
    with pytest.raises(ValueError):
        publish_selection(folder, changed)
    trace["ball_position_before_step_m"][0, 0, 0] += 0.01
    np.savez(folder / "physical_trace.npz", **trace)
    with pytest.raises(ValueError, match="actual initial state"):
        verify_selected_receipt(receipt, selection, portfolio, tmp_path)


def test_envelope_metadata_reader_is_not_full_restoration_and_rejects_tamper(tmp_path):
    path = tmp_path / "view.json"
    raw = envelope(dict(seed=1))
    path.write_text(json.dumps(raw))
    assert logical_document(path) == dict(seed=1)
    raw["stripped_document"]["seed"] = 2
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        logical_document(path)


def test_fixed_portfolio_scope_and_no_mixed_interventions():
    config = dict(
        postcontact_study_profile=PROFILE,
        postcontact_study_groups=GROUPS,
        max_attempts=42,
        neural_training_authorized=False,
        teacher_auto_admission=False,
        portfolio_control_arm_count=8,
        physical_payload_reader_mode="fixed_after_first_qualified",
        **dict.fromkeys(
            (
                "portfolio_model_path",
                "portfolio_fit_report_path",
                "portfolio_model_hash",
                "portfolio_fit_report_hash",
                "portfolio_fit_source_root",
                "portfolio_fit_source_head",
            ),
            "fixture",
        ),
    )
    require_declaration(config)
    for key, value in (
        ("postcontact_study_groups", GROUPS[:16]),
        ("portfolio_control_arm_count", True),
        ("max_attempts", 999),
        ("postcontact_probe", {}),
        ("physical_payload_reader_mode", "default"),
    ):
        with pytest.raises(ValueError):
            require_declaration(dict(config, **{key: value}))
