import copy
import json
from pathlib import Path

import pytest
from test_paid_ancestor_study import sources

from rosclaw_soccer.rsi.paid_ancestor_study import ancestor_train_cases
from rosclaw_soccer.rsi.paid_fresh_protocol import (
    CASE_SEEDS,
    PROTOCOL,
    ROUTER_FRESH_PROTOCOL,
    fresh_lifecycle_cases,
)
from rosclaw_soccer.rsi.paid_router_fresh import condition_hash, router_fresh_cases


def test_fixed_v2_is_distinct_from_all_consumed_and_fitted_conditions_without_resampling():
    source = sources()
    old, train = fresh_lifecycle_cases(source), ancestor_train_cases(source)
    before = copy.deepcopy((source, old, train))
    cases, metadata = router_fresh_cases(source, old, train)
    assert cases == fresh_lifecycle_cases(source, protocol=ROUTER_FRESH_PROTOCOL)
    assert len(cases) == 32 and len(metadata["excluded_condition_hashes"]) == 52
    assert len(metadata["excluded_original_geometry_hashes"]) == 16
    assert not set(map(condition_hash, cases)) & set(map(condition_hash, old + train))
    assert metadata["resampled_cases"] == 0
    assert CASE_SEEDS[PROTOCOL] == 2026102207
    assert CASE_SEEDS[ROUTER_FRESH_PROTOCOL] == 2026102301
    assert (source, old, train) == before
    assert all("obsolete_outcome" not in c for c in metadata["source_cases"])
    assert all(
        sum(c["fresh_class"] == label for c in cases) == 8
        for label in ("near_toward", "near_away", "far_toward", "far_away")
    )


@pytest.mark.parametrize("fault", ["missing", "changed_old", "overlap", "duplicate"])
def test_invalid_exclusion_contract_rejects_whole_declaration(fault):
    source = sources()
    old, train = fresh_lifecycle_cases(source), ancestor_train_cases(source)
    if fault == "missing":
        old.pop()
    elif fault == "changed_old":
        old[0]["goal_world"][0] += 0.01
    elif fault == "overlap":
        train[0] = fresh_lifecycle_cases(source, protocol=ROUTER_FRESH_PROTOCOL)[0]
    else:
        train[0] = train[1]
    with pytest.raises(ValueError):
        router_fresh_cases(source, old, train)


def test_protocol_cannot_be_silently_switched_or_custom_seed_swept():
    with pytest.raises(ValueError, match="versioned"):
        fresh_lifecycle_cases(sources(), protocol="try_seed_until_easy")


@pytest.mark.parametrize("fault", ["status", "all_three", "primary", "read_fresh", "binding"])
def test_unqualified_or_reselected_student_is_rejected_before_fresh_sources(tmp_path, fault):
    from rosclaw_soccer.rsi.paid_fresh_evidence import digest
    from rosclaw_soccer.rsi.paid_router_fresh import router_fresh_commitment

    pipeline = tmp_path / "pipeline"
    student = pipeline / "trained"
    student.mkdir(parents=True)
    (student / "complete.json").write_text("{}")
    student_hash = digest(student / "complete.json")
    root = tmp_path / "review"
    root.mkdir()
    declaration = dict(primary_seed=0, seeds=[0, 1, 2], pipelines=[str(pipeline)])
    review = dict(
        status="ROUTER_FRONTIER_PASS_PENDING_NEW_FRESH",
        all_three_seeds_pass=True,
        primary_seed=0,
        Fresh_rows_read=0,
        input_file_hashes={str(student / "complete.json"): student_hash},
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if fault == "status":
        review["status"] = "ROUTER_FRONTIER_REJECTED_KEEP_PARENT"
    elif fault == "all_three":
        review["all_three_seeds_pass"] = False
    elif fault == "primary":
        declaration["primary_seed"] = 1
    elif fault == "read_fresh":
        review["Fresh_rows_read"] = 1
    else:
        review["input_file_hashes"] = {}
    (root / "complete.json").write_text(json.dumps(review))
    (root / "declaration.json").write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="qualified"):
        router_fresh_commitment(
            student_root=student,
            student_hash=student_hash,
            sources=[],
            review_root=root,
            review_hash=digest(root / "complete.json"),
            previous_root=Path("never_read"),
            previous_hash="never_read",
            train_root=Path("never_read"),
            train_hash="never_read",
        )
