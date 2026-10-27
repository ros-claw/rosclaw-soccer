"""Review every post-touch candidate against audited exact-parent physics."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.relay_follow_comparison import (
    equal_prefix,
    pair_metrics,
    untouched_prefix_length,
)
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report


def validate_prefix(root: Path, baseline: dict, candidate: dict) -> tuple[int, bool]:
    bp = bind(root, baseline, "physics")
    if bp is None:
        raise ValueError("baseline physics required")
    with np.load(bp, allow_pickle=False) as f:
        count, touched = untouched_prefix_length(f["ball_forces"])
    for name in ("physics", "passer_nn", "receiver_velocity", "receiver_shadow"):
        a, b = bind(root, baseline, name), bind(root, candidate, name)
        if a is None or b is None:
            raise ValueError("complete causal physical and neural prefixes required")
        with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
            if not equal_prefix(dict(left), dict(right), count):
                raise ValueError("candidate differs before first real receiver contact")
            if not touched and any(not np.array_equal(left[k], right[k]) for k in left.files):
                raise ValueError("teacher changed a no-contact world")
    if baseline["artifacts"]["initial"]["hash"] != candidate["artifacts"]["initial"]["hash"]:
        raise ValueError("paired physical initial states differ")
    return count, touched


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=4,
        action="append",
        required=True,
        metavar=("SOURCE", "AUDIT", "OLD", "OLD_AUDIT"),
    )
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--dataset-hash", required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.group) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.dataset.is_symlink()
        or digest(a.dataset) != a.dataset_hash
    ):
        raise ValueError("all three full paired studies and pinned data required")
    ds = json.loads(a.dataset.read_text())
    if ds.get("status") != "PAIRED_RELAY_BALANCE_DECISIONS_EXPORTED_NOT_TRAINED":
        raise ValueError("original audited paired dataset required")
    pins = {(r["block"], r["expanded"]): r for r in ds["source_pins"]}
    seen, rows, frozen = set(), [], []
    for sr, ar, oldr, olda in a.group:
        root, audit, old, old_audit = map(Path, (sr, ar, oldr, olda))
        dec = json.loads((root / "declaration.json").read_text())
        block = dec["development_context_block"]
        if block in seen or (block, False) not in pins:
            raise ValueError("unique declared development blocks required")
        seen.add(block)
        audith = digest(audit)
        source, _, sh = report(
            root, audit, audith, "NEURAL_RELAY_POST_TOUCH_FOLLOW32_COLLECTED_PENDING_OWN_REPLAY"
        )
        oldpin = pins[(block, False)]
        historical, _, oldh = report(
            old,
            old_audit,
            oldpin["audit_hash"],
            "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY",
        )
        if oldh != oldpin["source_hash"] or source["source_head"] != a.producer_head:
            raise ValueError("exact source, parent and producer required")
        current = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["receiver_follow_gain"],
            ): r
            for r in source["records"]
        }
        parents = {
            (r["course"]["context"], r["course"]["lateral_m"]): r
            for r in historical["records"]
            if r["course"]["arm"]["balance_teacher"] == -1
        }
        if len(current) != 32 or len(parents) != 8:
            raise ValueError("all expected paired factorial conditions required")
        for condition, oldrow in sorted(parents.items()):
            base = current[(*condition, 0.0)]
            if any(
                base[k] != oldrow[k]
                for k in ("termination", "controls", "native_steps", "handoff", "result")
            ):
                raise ValueError("zero-follow baseline changed old outcome or boundary")
            for name, old_art in oldrow["artifacts"].items():
                new_art = base["artifacts"][name]
                if (
                    (old_art is None) != (new_art is None)
                    or old_art is not None
                    and old_art["hash"] != new_art["hash"]
                ):
                    raise ValueError("zero-follow baseline is not byte-identical")
                if old_art is not None:
                    bind(old, oldrow, name)
                    bind(root, base, name)
            parent_metrics = measured(root, base)
            for gain in (0.25, 0.5, 0.75):
                candidate = current[(*condition, gain)]
                count, touched = validate_prefix(root, base, candidate)
                child_metrics = measured(root, candidate)
                rows.append(
                    dict(
                        block=block,
                        context=condition[0],
                        side=condition[1],
                        gain=gain,
                        same_precontact_controls=count,
                        baseline_contact=touched,
                        baseline=parent_metrics,
                        candidate=child_metrics,
                        changes=pair_metrics(parent_metrics, child_metrics),
                    )
                )
        frozen.extend(
            (
                (root / "complete.json", sh),
                (audit, audith),
                (old / "complete.json", oldh),
                (old_audit, oldpin["audit_hash"]),
            )
        )
    if (
        seen != {0, 1, 2}
        or len(rows) != 72
        or digest(a.dataset) != a.dataset_hash
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("all 24 paired conditions and unchanged provenance required")
    summaries = {}
    for gain in (0.25, 0.5, 0.75):
        subset = [r for r in rows if r["gain"] == gain]
        counters = {k: sum(r["changes"][k] for r in subset) for k in subset[0]["changes"]}
        counters["eligible_for_further_data_research"] = bool(
            counters["handoff_rescue"] > 0
            and counters["precise_regression"] == 0
            and counters["body_failure_regression"] == 0
        )
        summaries[str(gain)] = counters
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RELAY_POST_TOUCH_FOLLOW_PAIRED_DEVELOPMENT_NOT_PROMOTED",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            summaries=summaries,
            records=rows,
            original_baselines_raw_byte_identical=True,
            paired_physical_and_private_precontact_prefix_exact=True,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
