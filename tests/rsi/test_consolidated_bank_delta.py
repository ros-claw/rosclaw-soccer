"""All affected courses must be retained, including the known new out-of-play."""

from copy import deepcopy

import pytest

from scripts.rsi_validate_consolidated_bank_delta import DECLARED_INDICES, selected_indices


def banks():
    rows = [
        dict(
            index=i,
            seed=42 + i,
            lane=0,
            warm=dict(high_quality=False, clean_foot_only=True, maximum_lateral_excursion_m=3.0),
            candidate=dict(
                high_quality=False, clean_foot_only=True, maximum_lateral_excursion_m=3.0
            ),
        )
        for i in range(52)
    ]
    for i in (21, 23, 35, 42):
        rows[i]["candidate"]["high_quality"] = True
    parent = {"rows": rows}
    rejected = deepcopy(parent)
    for i in range(52):
        rejected["rows"][i]["warm"] = deepcopy(rows[i]["candidate"])
    for i in (21, 23, 35):
        rejected["rows"][i]["candidate"]["high_quality"] = False
    for i in (6, 49):
        rejected["rows"][i]["candidate"]["high_quality"] = True
    rejected["rows"][25]["candidate"]["maximum_lateral_excursion_m"] = 4.3
    return parent, rejected


def test_complete_preregistered_union_includes_all_gains_losses_and_out() -> None:
    assert selected_indices(*banks()) == DECLARED_INDICES


def test_hiding_known_out_of_play_is_rejected() -> None:
    parent, rejected = banks()
    rejected["rows"][25]["candidate"]["maximum_lateral_excursion_m"] = 3.9
    with pytest.raises(ValueError, match="seven-course"):
        selected_indices(parent, rejected)


def test_new_other_regression_cannot_be_silently_truncated() -> None:
    parent, rejected = banks()
    rejected["rows"][5]["candidate"]["clean_foot_only"] = False
    with pytest.raises(ValueError, match="seven-course"):
        selected_indices(parent, rejected)


def test_stale_parent_or_misaligned_identity_rejected() -> None:
    parent, rejected = banks()
    rejected["rows"][21]["warm"]["high_quality"] = False
    with pytest.raises(ValueError, match="unaligned"):
        selected_indices(parent, rejected)
    parent, rejected = banks()
    rejected["rows"][21]["lane"] = 2
    with pytest.raises(ValueError, match="unaligned"):
        selected_indices(parent, rejected)


def test_incomplete_banks_rejected() -> None:
    parent, rejected = banks()
    parent["rows"].pop()
    with pytest.raises(ValueError, match="complete"):
        selected_indices(parent, rejected)
