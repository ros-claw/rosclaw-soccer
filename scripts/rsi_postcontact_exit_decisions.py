"""All300 actual postcontact decisions and raw exit banks, no new physics/NN."""

import numpy as np

from scripts.rsi_early_impact_execution_alignment import compare_postcontact_execution
from scripts.rsi_read_early_feedback_screen_arrays import verify_postcontact_exit_prediction


def read_exit_decisions(trace, evidence_root, *, expected_bank, sealed, arrays):
    decisions = [sealed(evidence_root / f"frame-{f:03d}.json") for f in range(300)]
    counts = trace["postcontact_exit_forecast_steps"][:, 0, 0]
    accumulated = np.cumsum(counts)
    selections = []
    for f, d in enumerate(decisions):
        if (
            d["schema"] != "soccer.rsi.preaction_postcontact80ms_exit_decision.v1"
            or type(d["frame"]) is not int
            or d["frame"] != f
            or d["persisted_before_current_native_substeps"] is not True
            or any(
                d[k] is not False
                for k in (
                    "future_source_frame_or_target_used",
                    "full_h300_gain_proven",
                    "neural_policy_trained",
                    "teacher_qualified",
                    "hardware_authorized",
                    "promotion_authorized",
                )
            )
            or np.asarray(d["requested_offset_nm"], dtype=np.float64).tobytes()
            != trace["postcontact_exit_requested_offset_nm"][f, 0].tobytes()
            or type(d["forecast_substeps"]) is not int
            or d["forecast_substeps"] != counts[f]
            or type(d["accumulated_forecast_substeps"]) is not int
            or d["accumulated_forecast_substeps"] != accumulated[f]
        ):
            raise ValueError("same-frame actual preaction postcontact decision required")
        witnesses = d["raw_witnesses"]
        if len(witnesses) * 40 != counts[f] or bool(witnesses) != (d["selected_index"] is not None):
            raise ValueError("all actual exit forecasts must have raw preaction witnesses")
        if not witnesses:
            continue
        if f == 0 or not np.any(trace["force_n"][f - 1, 0, :2] > 1):
            raise ValueError("raw exit bank must follow actual completed foot contact")
        if expected_bank is None or [w["name"] for w in witnesses] != [
            name for name, _ in expected_bank
        ]:
            raise ValueError("complete independent current-pose Jacobian bank required")
        raw = []
        for j, (w, (_, offset)) in enumerate(zip(witnesses, expected_bank, strict=True)):
            if w["path"] != str(evidence_root / f"frame-{f:03d}-candidate-{j:02d}.npz"):
                raise ValueError("fixed ordered actual exit witness paths required")
            a = arrays(w)
            if not np.array_equal(a["offset_nm"], offset):
                raise ValueError("raw exit action differs from independent current-pose Jacobian")
            verify_postcontact_exit_prediction(
                a, dict(w, is_zero=bool(not np.any(offset))), trace["force_n"][f - 1, 0]
            )
            raw.append(a)
        if (
            len({w["donor_hash"] for w in witnesses}) != 1
            or raw[0].keys() != raw[-1].keys()
            or any(
                raw[0][k].shape != raw[-1][k].shape
                or raw[0][k].dtype != raw[-1][k].dtype
                or raw[0][k].tobytes() != raw[-1][k].tobytes()
                for k in raw[0]
            )
        ):
            raise ValueError("same-donor whole independent repeated-zero exit capture required")
        feasible = [
            (j, w["score"]["proxy_cost"])
            for j, w in enumerate(witnesses[:-1])
            if w["score"]["feasible_predicted_exit"]
        ]
        best = 0
        if feasible:
            best, cost = min(feasible, key=lambda item: item[1])
            if (
                witnesses[0]["score"]["feasible_predicted_exit"]
                and witnesses[0]["score"]["proxy_cost"] - cost <= 1e-6
            ):
                best = 0
        if (
            type(d["selected_index"]) is not int
            or d["selected_index"] != best
            or d["selected_name"] != witnesses[best]["name"]
        ):
            raise ValueError("actual exit action differs from declared stable ranking/fallback")
        cancel = next(
            (
                n
                for n in range(f + 1, f + 3)
                if decisions[n]["status"] == "CANCELLED_OBSERVED_UNSAFE_NO_RETRY"
            ),
            None,
        )
        expected = np.zeros((300, 1, 29))
        expected[f : f + 3 if cancel is None else cancel, 0] = raw[best]["offset_nm"]
        if not np.array_equal(expected, trace["postcontact_exit_requested_offset_nm"]):
            raise ValueError("actual exit action differs from selected bounded lease")
        selections.append(
            compare_postcontact_execution(trace, raw[best], f, safety_cancel_frame=cancel)
        )
    if len(selections) > 1 or (
        np.any(trace["postcontact_exit_requested_offset_nm"]) and not selections
    ):
        raise ValueError("at most one complete exit search and proof for every correction required")
    return selections
