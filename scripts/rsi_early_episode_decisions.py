"""Read all H300 preaction decisions; no physics, learning or data admission."""

import numpy as np

from scripts.rsi_early_impact_execution_alignment import compare_execution
from scripts.rsi_early_impact_h300 import intervention_frames
from scripts.rsi_read_early_feedback_screen_arrays import verify_early_prediction


def read_decisions(trace, evidence_root, *, sealed, arrays):
    """Callbacks authenticate files only; neither is passed into an actor."""
    forecast_steps = trace["early_impact_feedback_forecast_steps"]
    requested, _ = intervention_frames(trace)
    decisions = [sealed(evidence_root / f"frame-{f:03d}.json") for f in range(300)]
    selections = []
    for f, decision in enumerate(decisions):
        if (
            type(decision["frame"]) is not int
            or decision["frame"] != f
            or decision["persisted_before_current_native_substeps"] is not True
            or decision["future_source_frame_or_target_used"] is not False
            or np.asarray(decision["requested_offset_nm"], dtype=np.float64).tobytes()
            != trace["early_impact_feedback_requested_offset_nm"][f, 0].tobytes()
        ):
            raise ValueError("each same-frame source-enforced preaction decision required")
        raw = []
        for j, w in enumerate(decision["raw_witnesses"]):
            if w["path"] != str(evidence_root / f"frame-{f:03d}-candidate-{j:02d}.npz"):
                raise ValueError("same ordered raw preaction witness paths required")
            a = arrays(w)
            metadata = dict(w, is_zero=bool(not np.any(a["requested_offset_nm"][0])))
            verify_early_prediction(a, metadata, int(trace["foot_heading_state"][f, 0, 1]))
            raw.append(a)
        selected = decision["selected_index"]
        if (
            type(decision["forecast_substeps"]) is not int
            or decision["forecast_substeps"] != len(raw) * 40
            or decision["forecast_substeps"] != forecast_steps[f, 0, 0]
        ):
            raise ValueError("each actual forecast window must have raw preaction evidence")
        if selected is not None:
            if type(selected) is not int or not 0 <= selected < len(raw):
                raise ValueError("actual selected action index required")
            witnesses = decision["raw_witnesses"]
            if (
                not 2 <= len(raw) <= 14
                or witnesses[0]["name"] != "baseline"
                or witnesses[-1]["name"] != "repeat_zero"
                or len({w["donor_hash"] for w in witnesses}) != 1
                or set(raw[0]) != set(raw[-1])
                or any(
                    raw[0][k].shape != raw[-1][k].shape
                    or raw[0][k].dtype != raw[-1][k].dtype
                    or raw[0][k].tobytes() != raw[-1][k].tobytes()
                    for k in raw[0]
                )
            ):
                raise ValueError("complete same-donor bank and whole repeated zero required")
            eligible = [
                (j, w["score"]["proxy_cost"])
                for j, w in enumerate(witnesses[:-1])
                if w["score"]["feasible_predicted_impact"]
            ]
            if not eligible or not witnesses[0]["score"]["feasible_predicted_impact"]:
                raise ValueError("selection requires a feasible complete baseline impact")
            best, cost = min(eligible, key=lambda item: item[1])
            if witnesses[0]["score"]["proxy_cost"] - cost <= 1e-6:
                best = 0
            if selected != best:
                raise ValueError("actual selection differs from declared stable local ranking")
            cancel = next(
                (
                    n
                    for n in range(f + 1, f + 3)
                    if decisions[n]["status"] == "CANCELLED_CURRENT_UNSAFE_CONTACT_OR_PELVIS"
                ),
                None,
            )
            expected_requests = np.zeros((300, 1, 29))
            lease_periods = 3 if cancel is None else cancel - f
            expected_requests[f : f + lease_periods, 0] = raw[selected]["requested_offset_nm"][0]
            if not np.array_equal(
                expected_requests, trace["early_impact_feedback_requested_offset_nm"]
            ):
                raise ValueError("actual requests differ from selected bounded action lease")
            selections.append(
                compare_execution(trace, raw[selected], f, safety_cancel_frame=cancel)
            )
    if len(selections) > 1 or (requested is not None and not selections):
        raise ValueError(
            "at most one complete selection and evidence for every correction required"
        )
    return selections
