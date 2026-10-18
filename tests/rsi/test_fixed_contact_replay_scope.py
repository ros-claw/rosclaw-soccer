import ast
import hashlib
from pathlib import Path

import pytest

from scripts.rsi_replay_first_touch_contacts import (
    fixed_replay_scope,
    verify_none_demonstration_sources,
)


def config(surface):
    count = 6 if surface else 4
    names = (
        (
            "surface_raw_8",
            "surface_raw_56",
            "surface_context_0",
            "surface_context_1",
            "surface_raw_4",
            "surface_raw_32",
        )
        if surface
        else tuple(f"old{i}" for i in range(4))
    )
    result = dict(
        new_replay_budget=count,
        new_substep_budget=count * 3000,
        cases=[dict(case_id=n, receipt_hash=str(i)) for i, n in enumerate(names)],
    )
    if surface:
        result["replay_profile"] = "surface-six"
    return result


@pytest.mark.parametrize("surface", [False, True])
def test_exact_scope_preserves_old_four_and_separately_declares_six(surface):
    assert fixed_replay_scope(config(surface)) == ((6, 18000) if surface else (4, 12000))


@pytest.mark.parametrize("surface", [False, True])
@pytest.mark.parametrize(
    "tamper", ["partial", "budget", "steps", "duplicate", "receipt", "retag", "truthy"]
)
def test_reject_partial_duplicate_or_retagged_scope(surface, tamper):
    c = config(surface)
    if tamper == "partial":
        c["cases"].pop()
    elif tamper == "budget":
        c["new_replay_budget"] += 1
    elif tamper == "steps":
        c["new_substep_budget"] -= 1
    elif tamper == "duplicate":
        c["cases"][1]["case_id"] = c["cases"][0]["case_id"]
    elif tamper == "receipt":
        c["cases"][1]["receipt_hash"] = c["cases"][0]["receipt_hash"]
    elif tamper == "retag":
        c["replay_profile"] = "legacy-four" if surface else "surface-six"
    else:
        c["new_replay_budget"] = True
    with pytest.raises(ValueError):
        fixed_replay_scope(c)


def test_cannot_reorder_fixed_six_or_request_an_unbounded_profile():
    c = config(True)
    c["cases"] = c["cases"][::-1]
    with pytest.raises(ValueError):
        fixed_replay_scope(c)
    c["replay_profile"] = "any-number-of-replays"
    with pytest.raises(ValueError):
        fixed_replay_scope(c)


def test_model_allocation_integration_outputs_and_failure_tail_match_qualified_four_case_replay():
    # Derived from the independently frozen 933505fe replay source. Scope
    # validation may change, but all original model/physics/witness arithmetic
    # from root allocation onward must stay byte-for-byte identical as AST.
    from scripts import rsi_replay_first_touch_contacts as module

    function = next(
        n
        for n in ast.parse(Path(module.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    start = next(
        i
        for i, n in enumerate(function.body)
        if isinstance(n, ast.Expr) and ast.unparse(n.value) == "root.mkdir()"
    )
    body = ast.dump(
        ast.Module(body=function.body[start:], type_ignores=[]), include_attributes=False
    )
    assert (
        hashlib.sha256(body.encode()).hexdigest()
        == "94a24c5e8430401162225a0f3ac0dee8ad270f7078942972163ce290fdec46fd"
    )


def test_missing_three_baselines_have_separate_finite_budget_not_repeated_old_replays():
    c = dict(
        replay_profile="surface-missing-three",
        new_replay_budget=3,
        new_substep_budget=9000,
        cases=[
            dict(case_id=n, receipt_hash=str(i))
            for i, n in enumerate(("baseline_context_1", "baseline_raw_4", "baseline_raw_32"))
        ],
    )
    assert fixed_replay_scope(c) == (3, 9000)
    c["cases"][0]["case_id"] = "surface_raw_8"
    with pytest.raises(ValueError):
        fixed_replay_scope(c)
    c["cases"].pop()
    with pytest.raises(ValueError):
        fixed_replay_scope(c)


def none_five_config():
    return dict(
        replay_profile="demonstration-missing-five",
        new_replay_budget=5,
        new_substep_budget=15000,
        cases=[
            dict(case_id=n, receipt_hash=str(i))
            for i, n in enumerate(
                ("none_raw_4", "none_raw_8", "none_raw_32", "none_raw_56", "none_context_1")
            )
        ],
    )


def test_five_original_none_missing_endpoints_are_not_new_policy_courses():
    c = none_five_config()
    assert fixed_replay_scope(c) == (5, 15000)
    c["cases"][0]["case_id"] = "none_context_0"
    with pytest.raises(ValueError):
        fixed_replay_scope(c)


@pytest.mark.parametrize("tamper", ["partial", "order", "duplicate", "budget", "bool"])
def test_none_five_rejects_scope_tamper(tamper):
    c = none_five_config()
    if tamper == "partial":
        c["cases"].pop()
    elif tamper == "order":
        c["cases"].reverse()
    elif tamper == "duplicate":
        c["cases"][1]["receipt_hash"] = c["cases"][0]["receipt_hash"]
    elif tamper == "budget":
        c["new_substep_budget"] += 3000
    else:
        c["new_replay_budget"] = True
    with pytest.raises(ValueError):
        fixed_replay_scope(c)


@pytest.mark.parametrize(
    "tamper", [None, "reader", "schema", "rows", "complete", "flag", "job", "audit"]
)
def test_none_five_requires_both_whole_original_studies(tmp_path, tamper):
    jobs = [f"original-public-{n}" for n in (4, 8, 32, 56)] + ["control-context-1"]
    receipts = [dict(receipt_hash=str(i), job=j, outcome={}) for i, j in enumerate(jobs)]
    documents, cfg, pins = {}, {}, {}
    for prefix, count, subset, schema in (
        ("nominal", 4, receipts[:4], "soccer.rsi.mesh_original_public_terminal_readback.v1"),
        (
            "baseline",
            18,
            receipts[4:],
            "soccer.rsi.mesh_option4_teacher_screen_terminal_readback.v1",
        ),
    ):
        reader = tmp_path / prefix
        reader.write_text("frozen reader")
        reader_hash = "sha256:" + hashlib.sha256(reader.read_bytes()).hexdigest()
        results = list(subset) if count == 4 else [{}, {}, subset[0]] + [{}] * 15
        documents[prefix + "-complete"] = dict(results=results, report_hash=prefix + "-c")
        documents[prefix + "-audit"] = dict(
            schema=schema,
            report_hash=prefix + "-a",
            complete_hash=prefix + "-c",
            reader_source_hash=reader_hash,
            source_native_bodies=count,
            whole_input_and_witness_files_verified=True,
            shared_payload_logical_seals_rebuilt=True,
            prior_independent_replays_authenticated=True,
            rows=list(subset),
        )
        cfg.update(
            {
                f"{prefix}_reader_source": str(reader),
                f"{prefix}_terminal_readback": prefix + "-audit",
                f"{prefix}_complete": prefix + "-complete",
                f"{prefix}_terminal_readback_hash": prefix + "-a",
            }
        )
    audit = documents["baseline-audit"]
    if tamper == "reader":
        audit["reader_source_hash"] = "other"
    elif tamper == "schema":
        audit["schema"] = "different"
    elif tamper == "rows":
        audit["rows"] = []
    elif tamper == "complete":
        documents["baseline-complete"]["results"][2] = {}
    elif tamper == "flag":
        audit["shared_payload_logical_seals_rebuilt"] = False
    elif tamper == "job":
        receipts[4]["job"] = "teacher-context-1"
    elif tamper == "audit":
        cfg["nominal_terminal_readback_hash"] = "other"
    if tamper:
        with pytest.raises(ValueError):
            verify_none_demonstration_sources(cfg, receipts, documents.__getitem__, pins)
    else:
        verify_none_demonstration_sources(cfg, receipts, documents.__getitem__, pins)
        assert len(pins) == 2
