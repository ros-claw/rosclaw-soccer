import copy

import pytest

from rosclaw_soccer.rsi.protected_online_evidence import validate_feedback


def feedback():
    return dict(
        parent_model_hash="sha256:" + "a" * 64,
        generation=0,
        runtime_selection_authorized=False,
        samples=[dict(course_index=i, candidate=c) for i in (0, 2, 5) for c in range(12)],
    )


def test_complete_matched_physical_feedback_batch():
    value = feedback()
    validate_feedback(value, value["parent_model_hash"], 0)


@pytest.mark.parametrize("change", ["duplicate", "missing", "parent", "generation", "authority"])
def test_count_alone_cannot_certify_coverage_lineage_or_authority(change):
    value = feedback()
    if change == "duplicate":
        value["samples"][-1] = copy.deepcopy(value["samples"][0])
    elif change == "missing":
        value["samples"].pop()
    elif change == "parent":
        value["parent_model_hash"] = "sha256:" + "b" * 64
    elif change == "generation":
        value["generation"] = 2
    else:
        value["runtime_selection_authorized"] = True
    with pytest.raises(ValueError):
        validate_feedback(value, "sha256:" + "a" * 64, 0)
