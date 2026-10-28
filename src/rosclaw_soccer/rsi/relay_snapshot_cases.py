"""Select balanced already-recorded development cases, never fresh holdout."""


def select_relay_snapshot_cases(
    records: list, *, nondefault: bool, contexts: tuple[int, ...] = (2, 3, 5, 7)
) -> list:
    if type(records) is not list or type(nondefault) is not bool:
        raise ValueError("explicit recorded development case bank required")
    if contexts not in ((2, 3, 5, 7), (8, 9, 11, 13), (15, 16, 17, 18)):
        raise ValueError("explicitly declared development contexts required, never sealed fresh")
    offsets = (-0.35, -0.12, 0.35) if nondefault else (0.12,)
    chosen = [r for r in records if r["course"]["arm"]["receiver_lateral_offset_m"] in offsets]
    identities = {
        (
            r["course"]["context"],
            r["course"]["lateral_m"],
            r["course"]["arm"]["receiver_lateral_offset_m"],
        )
        for r in chosen
    }
    expected = {
        (context, side, offset)
        for context in contexts
        for side in (-0.35, 0.35)
        for offset in offsets
    }
    if len(chosen) != len(expected) or identities != expected:
        raise ValueError("complete unique balanced recorded development conditions required")
    return chosen
