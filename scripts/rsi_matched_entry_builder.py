"""Matched-time ablation from OWN current forecast, retaining all old anchors."""


def make_builder(source_poses, limits, source_evidence_hash):
    import numpy as np

    from rosclaw_soccer.providers.g1.sonic_matched_entry import (
        InfeasibleMatchedBridge,
        prepare_matched_entry_pair,
    )
    from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import quarantine_wrist_reference
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_response_cem_builder import certified_anchors

    def build(predict, rows, prefix, source, item, folder):
        anchors = certified_anchors(predict, rows, source, item)
        availability = []
        for index, law in ((0, 0), (2, 3), (4, 4)):
            anchor = anchors[index]
            handoff = anchor.report["handoff_frame"]
            if handoff is None:
                availability.append(
                    dict(law_index=law, status="NO_ORIGINAL_HANDOFF", candidate_indices=[])
                )
                continue
            q = anchor.arrays["canonical_qpos"][handoff - 20].copy()
            v = anchor.arrays["canonical_qvel"][handoff - 20].copy()
            held, _ = quarantine_wrist_reference(source_poses, q[:36])
            try:
                pair = prepare_matched_entry_pair(
                    held[286:767], q[:36], v, limits, source_evidence_hash=source_evidence_hash
                )
            except InfeasibleMatchedBridge as error:
                availability.append(
                    dict(
                        law_index=law,
                        status="NO_FEASIBLE_MATCHED_BRIDGE",
                        candidate_indices=[],
                        feasibility_attempts=list(error.attempts),
                    )
                )
                continue
            indices = []
            for mode in ("matched_position", "matched_velocity"):
                report = pair.report(mode)
                indices.append(len(rows))
                future = predict(
                    law,
                    (0.0, 0.0, 0.0),
                    dict(method="matched_entry_reference", mode=mode, planned_adaptation=report),
                    entry_reference_mode=mode,
                )
                if (
                    future.report["handoff_frame"] != handoff
                    or future.report["entry_reference_adaptation"] != report
                ):
                    raise ValueError("new mode did not reconstruct its OWN forecast handoff plan")
                for key in future.arrays:
                    np.testing.assert_array_equal(
                        future.arrays[key][: handoff - 20], anchor.arrays[key][: handoff - 20]
                    )
            availability.append(
                dict(
                    law_index=law,
                    status="MATCHED_PAIR_EXECUTED",
                    entry_intervals=pair.entry_intervals,
                    candidate_indices=indices,
                    position_initial_velocity_gap_rad_s=pair.position_initial_velocity_gap_rad_s,
                    velocity_initial_velocity_gap_rad_s=pair.velocity_initial_velocity_gap_rad_s,
                )
            )
        write_once(
            folder / "matched-entry-availability.json",
            dict(
                results=availability,
                source="own_zero_C_native_forecast_not_course_lookup",
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        return availability

    return build


def choose_matched_candidate(rows):
    import math

    if (
        not 6 <= len(rows) <= 12
        or len(rows) % 2
        or [r["law_index"] for r in rows[:6]] != [0, 0, 3, 3, 4, 4]
    ):
        raise ValueError("six anchors and at most three matched pairs required")
    if [r["offset_root_m"] for r in rows[:6]] != [[0.0, 0.0, 0.0], [0.0, -0.025, -0.025]] * 3:
        raise ValueError("old zero/LOWER anchors must remain available")
    seen = set()
    for i in range(6, len(rows), 2):
        position, velocity = rows[i : i + 2]
        law = position["law_index"]
        if (
            law in seen
            or law not in (0, 3, 4)
            or velocity["law_index"] != law
            or position.get("entry_reference_mode") != "matched_position"
            or velocity.get("entry_reference_mode") != "matched_velocity"
            or any(r["offset_root_m"] != [0.0, 0.0, 0.0] for r in (position, velocity))
        ):
            raise ValueError("complete declared same-law position/velocity pairs required")
        seen.add(law)
    for r in rows:
        s = r["score"]
        if type(s["qualified"]) is not bool or (
            s["qualified"] and (type(s["cost"]) not in (float, int) or not math.isfinite(s["cost"]))
        ):
            raise ValueError("typed original qualified/cost required")
    eligible = [i for i, r in enumerate(rows) if r["score"]["qualified"]]
    return min(eligible, key=lambda i: (rows[i]["score"]["cost"], i)) if eligible else 0
