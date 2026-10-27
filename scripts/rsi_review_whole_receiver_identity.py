"""Review all paired zero-whole motor identity cases and historic baselines."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.relay_whole_identity_comparison import compare_receiver_traces
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def artifact_equal(left_root: Path, left: dict, right_root: Path, right: dict, name: str) -> bool:
    a, b = bind(left_root, left, name), bind(right_root, right, name)
    if a is None or b is None:
        return a is None and b is None
    if digest(a) == digest(b):
        return True
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        return set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=4,
        action="append",
        required=True,
        metavar=("ROOT", "AUDIT", "OLD", "OLD_AUDIT"),
    )
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--dataset-hash", required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.group) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.dataset.is_symlink()
        or digest(a.dataset) != a.dataset_hash
    ):
        raise ValueError("all three audited source blocks and exact parent data required")
    dataset = json.loads(a.dataset.read_text())
    if dataset.get("status") != "PAIRED_RELAY_BALANCE_DECISIONS_EXPORTED_NOT_TRAINED":
        raise ValueError("original audited parent dataset required")
    pins = {(r["block"], r["expanded"]): r for r in dataset["source_pins"]}
    seen, rows, frozen = set(), [], []
    for sr, ar, oldr, olda in a.group:
        root, own, old, old_own = map(Path, (sr, ar, oldr, olda))
        dec = json.loads((root / "declaration.json").read_text())
        block = dec["development_context_block"]
        if block in seen or (block, False) not in pins:
            raise ValueError("unique admitted original development blocks required")
        seen.add(block)
        oh = digest(own)
        source, _, sh = report(
            root, own, oh, "NEURAL_RELAY_WHOLE_RECEIVER_IDENTITY16_PENDING_OWN_REPLAY", worlds=16
        )
        pin = pins[(block, False)]
        parent, _, ph = report(
            old,
            old_own,
            pin["audit_hash"],
            "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY",
        )
        if ph != pin["source_hash"] or source["source_head"] != a.producer_head:
            raise ValueError("exact original parent and identity source required")
        current = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["receiver_whole"],
            ): r
            for r in source["records"]
        }
        parents = {
            (r["course"]["context"], r["course"]["lateral_m"]): r
            for r in parent["records"]
            if r["course"]["arm"]["balance_teacher"] == -1
        }
        if len(current) != 16 or len(parents) != 8:
            raise ValueError("complete identity source factorial required")
        for condition, historical in sorted(parents.items()):
            base, whole = current[(*condition, False)], current[(*condition, True)]
            outcomes = all(
                base[k] == whole[k]
                for k in ("termination", "controls", "native_steps", "handoff", "result")
            )
            parent_outcomes = all(
                base[k] == historical[k]
                for k in ("termination", "controls", "native_steps", "handoff", "result")
            )
            exact = {
                name: artifact_equal(root, base, root, whole, name)
                for name in (
                    "initial",
                    "physics",
                    "passer_nn",
                    "receiver_shadow",
                    "receiver_return_nn",
                )
            }
            parent_exact = True
            for name, old_art in historical["artifacts"].items():
                new_art = base["artifacts"][name]
                if (old_art is None) != (new_art is None):
                    parent_exact = False
                elif old_art is not None:
                    bind(root, base, name)
                    bind(old, historical, name)
                    parent_exact = parent_exact and old_art["hash"] == new_art["hash"]
            bp, wp = bind(root, base, "receiver_velocity"), bind(root, whole, "receiver_velocity")
            if bp is None or wp is None:
                raise ValueError("complete velocity neural traces required")
            with np.load(bp, allow_pickle=False) as x, np.load(wp, allow_pickle=False) as y:
                trace = compare_receiver_traces(dict(x), dict(y))
            rows.append(
                dict(
                    block=block,
                    context=condition[0],
                    side=condition[1],
                    physical_and_other_traces=exact,
                    candidate_outcomes_exact=outcomes,
                    historic_parent_outcomes_exact=parent_outcomes,
                    historic_parent_all_traces_exact=parent_exact,
                    receiver_trace=trace,
                )
            )
        frozen.extend(
            (
                (root / "complete.json", sh),
                (own, oh),
                (old / "complete.json", ph),
                (old_own, pin["audit_hash"]),
            )
        )
    if (
        seen != {0, 1, 2}
        or len(rows) != 24
        or digest(a.dataset) != a.dataset_hash
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("complete unchanged 24 condition identity study required")
    passed = all(
        all(r["physical_and_other_traces"].values())
        and r["candidate_outcomes_exact"]
        and r["historic_parent_outcomes_exact"]
        and r["historic_parent_all_traces_exact"]
        and r["receiver_trace"]["common_fields_exact"]
        and r["receiver_trace"]["action_is_actual_onnx_reference"]
        and r["receiver_trace"]["extra_joint_delta_exact_zero"]
        for r in rows
    )
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RELAY_WHOLE_RECEIVER_ZERO_IDENTITY_QUALIFIED"
            if passed
            else "RELAY_WHOLE_RECEIVER_ZERO_IDENTITY_REJECTED",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            conditions=24,
            records=rows,
            own_physics_and_neural_replay_complete=True,
            identity_passed=passed,
            trained=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
