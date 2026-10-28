import pytest

from rosclaw_soccer.rsi.relay_snapshot_cases import select_relay_snapshot_cases


def records():
    return [
        {"course": {"context": c, "lateral_m": side, "arm": {"receiver_lateral_offset_m": offset}}}
        for c in (2, 3, 5, 7)
        for side in (-0.35, 0.35)
        for offset in (-0.35, -0.12, 0.12, 0.35)
    ]


def test_expansion_is_complete_balanced_and_disjoint():
    bank = records()
    default = select_relay_snapshot_cases(bank, nondefault=False)
    expanded = select_relay_snapshot_cases(bank, nondefault=True)
    assert len(default) == 8 and len(expanded) == 24
    assert all(x not in default for x in expanded)


def test_missing_or_duplicated_case_fails_entire_declaration():
    bank = records()
    bank.pop(0)
    bank.append(bank[0])
    with pytest.raises(ValueError, match="complete unique balanced"):
        select_relay_snapshot_cases(bank, nondefault=True)
