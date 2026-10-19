"""Fixed selected-action continuation parity; no search, learning or promotion.

Gold predictions remain comparison-only, outside the controller. Selection is
retrospective from the closed development study, not a deployable actor input.
"""

import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_audit_early_neural_components import load_inputs
from scripts.rsi_early_feedback_calibration import preserve_forecast
from scripts.rsi_feedback_branch_inputs import prepare as original_prepare
from scripts.rsi_owned_feedback_parity_inputs import require_scope as zero_scope
from scripts.rsi_stage_release_contact_replay import digest


def require_scope(cfg):
    fixed = dict(
        profile="six-selected-owned-continuation-parity-v1",
        frame_selection="CLOSED_H300_SELECTED_ACTION_COMPARISON_ONLY",
        selected_branches_per_case=2,
        reuse_owned_updates_only_on_second_branch=True,
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("fixed whole-six selected-action paired diagnostic required")
    if "zero_branches_per_case" in cfg or "contact_frame_lookback" in cfg:
        raise ValueError("selected actions must not be declared zero or contact-minus-two")
    normalized = dict(cfg)
    normalized.update(
        profile="six-early-owned-continuation-parity-v1",
        frame_selection="RETROSPECTIVE_CONTACT_MINUS_TWO_DIAGNOSTIC_ONLY",
        zero_branches_per_case=2,
        contact_frame_lookback=2,
    )
    normalized.pop("selected_branches_per_case")
    zero_scope(normalized)


def compare_arrays(actual, expected):
    """Strict bytes, shapes and dtypes, including all intermediate NN records."""
    if set(actual) != set(expected):
        raise ValueError("whole selected forecast field set differs")
    for key in actual:
        a, b = actual[key], expected[key]
        if (
            a.shape != b.shape
            or a.dtype != b.dtype
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
            or a.tobytes() != b.tobytes()
        ):
            raise ValueError("selected forecast exact arrays differ: " + key)


def bind_case(case, row, alignment, decision, gold, actual):
    """Bind the selected current state to the untouched original prefix."""
    selections = alignment["selected_window_alignment"]
    if (
        row["job"] != case["job"]
        or alignment["job"] != case["job"]
        or row["source_receipt_hash"] != case["receipt"]["receipt_hash"]
        or len(selections) != 1
        or selections[0]["full80ms_execution_alignment_verified"] is not True
        or selections[0]["actual_safety_cancel_frame"] is not None
    ):
        raise ValueError("same complete selected source case required")
    frame = selections[0]["forecast_start_frame"]
    if (
        type(frame) is not int
        or not 30 <= frame <= 294
        or frame // 20 != (frame + 3) // 20
        or decision["frame"] != frame
        or decision["persisted_before_current_native_substeps"] is not True
        or decision["future_source_frame_or_target_used"] is not False
    ):
        raise ValueError("same complete preaction current-state window required")
    source = case["trace"]
    for key in set(source) & set(actual):
        a, b = source[key], actual[key]
        if a.shape[:1] == (300,) and b.shape[:1] == (300,):
            compare_arrays({key: a[:frame]}, {key: b[:frame]})
    # All completed feedback, body and network memory is bound above. These are
    # current BEFORE-STEP values, unlike this frame's intervened torque/forces.
    from rosclaw_soccer.rsi.feedback_branch_control import BODY_SHAPES

    current_keys = (
        set(BODY_SHAPES)
        | {
            "canonical_qpos",
            "canonical_qvel",
            "joint_target_rad",
            "motor_delta_rad",
            "recurrent_residual_hidden_state",
            "foundation_navigation_joint_target_rad",
            "taskspace_joint_increment_rad",
        }
        | {k for k in source if k.startswith("foundation_neural_")}
    )
    compare_arrays(
        {k: source[k][frame] for k in current_keys},
        {k: actual[k][frame] for k in current_keys},
    )
    offset = gold["offset_nm"]
    if offset.shape != (29,) or offset.dtype != np.float64 or not np.isfinite(offset).all():
        raise ValueError("complete selected torque vector required")
    requested = gold["requested_offset_nm"]
    if (
        requested.shape != (40, 29)
        or not np.array_equal(requested[:30], np.tile(offset, (30, 1)))
        or np.any(requested[30:])
    ):
        raise ValueError("selected60ms torque lease plus20ms unmodified tail required")
    if not np.array_equal(offset, np.asarray(decision["requested_offset_nm"], dtype=np.float64)):
        raise ValueError("selected raw and preaction decision differ")
    case["original_contact_frame"] = case["frame"]
    case["frame"] = frame
    case["selected_offset_nm"] = offset.copy()


def prepare(path):
    p = original_prepare(path, scope_validator=require_scope)
    _, _, pins, complete, modelroot = load_inputs(p.cfg, path, p.source, p.root)
    if modelroot != p.model_root:
        raise ValueError("same complete Foundation assets required")
    for name, h in pins.items():
        if name in p.pins and p.pins[name] != h:
            raise ValueError("conflicting closed source evidence")
        p.pins[name] = h
    readback = json.loads(Path(p.cfg["readback_path"]).read_text())

    def linked(name, expected=None):
        name = str(name)
        h = digest(name)
        if name not in p.pins or p.pins[name] != h or (expected is not None and expected != h):
            raise ValueError("selected comparison must belong to whole closed input chain")
        return Path(name)

    for i, (case, row, alignment) in enumerate(
        zip(p.cases, complete["rows"], readback["rows"], strict=True)
    ):
        selections = alignment["selected_window_alignment"]
        if len(selections) != 1:
            raise ValueError("exactly one selected window for each original case required")
        frame = selections[0]["forecast_start_frame"]
        decision = json.loads(
            linked(
                Path(p.cfg["producer_root"]) / f"case-{i}-preaction-evidence/frame-{frame:03d}.json"
            ).read_text()
        )
        if decision["report_hash"] != hash_json(
            {k: v for k, v in decision.items() if k != "report_hash"}
        ):
            raise ValueError("selected preaction seal differs")
        selected = decision["selected_index"]
        if type(selected) is not int or not 0 <= selected < len(decision["raw_witnesses"]):
            raise ValueError("registered selected action required")
        witness = decision["raw_witnesses"][selected]
        goldpath = linked(witness["path"], witness["sha256"])
        with (
            np.load(goldpath, allow_pickle=False) as a,
            np.load(
                linked(row["arrays"]["path"], row["arrays"]["sha256"]), allow_pickle=False
            ) as b,
        ):
            bind_case(case, row, alignment, decision, dict(a), dict(b))
        case["selected_gold_witness"] = dict(witness)
    p.selected_h300_hash = complete["report_hash"]
    return p


def preserve_and_compare_selected(root, index, branch, out, case, pins):
    witness = preserve_forecast(root, index, branch, out)
    gold = case["selected_gold_witness"]
    if (
        digest(gold["path"]) != pins[gold["path"]]
        or out.donor_integration_state_hash != gold["donor_hash"]
    ):
        raise ValueError("selected original donor or gold bytes changed")
    with (
        np.load(witness["path"], allow_pickle=False) as a,
        np.load(gold["path"], allow_pickle=False) as b,
    ):
        expected = dict(b)
        offset = expected.pop("offset_nm")
        compare_arrays({"offset": case["selected_offset_nm"]}, {"offset": offset})
        compare_arrays(dict(a), expected)
    witness.update(selected_gold_sha256=gold["sha256"], all_selected_gold_arrays_bit_exact=True)
    return witness
