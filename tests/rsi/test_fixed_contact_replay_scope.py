import ast
import hashlib
from pathlib import Path

import pytest

from scripts.rsi_replay_first_touch_contacts import fixed_replay_scope


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
