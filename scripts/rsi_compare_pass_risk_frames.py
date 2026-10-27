"""Compare two audited feature frames on identical held-out pass contexts."""

import argparse
import json
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("world", "world-audit", "body", "body-audit"):
        parser.add_argument(f"--{name}", type=Path, action="append", required=True)
        parser.add_argument(f"--{name}-hash", action="append", required=True)
    parser.add_argument("--dataset-hash", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    keys = ("world", "world_audit", "body", "body_audit")
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(len(getattr(args, k)) != 4 or len(getattr(args, k + "_hash")) != 4 for k in keys)
    ):
        raise ValueError("four paired shards per frame and unused output required")
    pins, frame_rows = {}, {}
    for frame in ("world", "body"):
        rows = {}
        for i in range(4):
            source = getattr(args, frame)[i] / "complete.json"
            audit_path = getattr(args, frame + "_audit")[i] / "complete.json"
            source_hash = getattr(args, frame + "_hash")[i]
            audit_hash = getattr(args, frame + "_audit_hash")[i]
            if (
                source.is_symlink()
                or audit_path.is_symlink()
                or digest(source) != source_hash
                or digest(audit_path) != audit_hash
            ):
                raise ValueError("exact source and prediction-replay audit required")
            model, audit = (json.loads(p.read_text()) for p in (source, audit_path))
            if (
                model["status"] != "ACTUAL_PASS_CONTACT_RISK_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL"
                or model["dataset_hash"] != args.dataset_hash
                or model.get("feature_frame", "world") != frame
                or model["shard_index"] != i
                or model["shard_count"] != 4
                or audit["status"]
                != "ACTUAL_PASS_CONTACT_RISK_HOLDOUT_PREDICTIONS_REPLAYED_NOT_CERTIFIED"
                or audit["source_hash"] != source_hash
                or audit["dataset_hash"] != args.dataset_hash
                or audit.get("feature_frame", "world") != frame
            ):
                raise ValueError("audited same-data, same-shard feature-frame lineage required")
            audited = {(r["heldout_context"], r["seed"]): r for r in audit["rows"]}
            if len(audited) != len(audit["rows"]) or len(model["rows"]) != len(audited):
                raise ValueError("unique fully audited models required")
            for row in model["rows"]:
                key = row["heldout_context"], row["seed"]
                if (
                    key in rows
                    or key not in audited
                    or any(
                        row[k] != audited[key][k]
                        for k in ("tp", "fn", "fp", "tn", "checkpoint_hash")
                    )
                ):
                    raise ValueError("unique exact replayed held-out counts required")
                rows[key] = row
            pins[str(source)], pins[str(audit_path)] = source_hash, audit_hash
        frame_rows[frame] = rows
    if (
        set(frame_rows["world"]) != set(frame_rows["body"])
        or len(frame_rows["world"]) != 48
        or len({k[0] for k in frame_rows["world"]}) != 16
    ):
        raise ValueError("same sixteen held-out contexts and all three seeds required")
    result = []
    for key in sorted(frame_rows["world"]):
        world, body = frame_rows["world"][key], frame_rows["body"][key]
        if any(
            world[k] != body[k]
            for k in ("train_rows", "heldout_rows", "train_positive", "heldout_positive")
        ):
            raise ValueError("exact same train/held-out sample split required")
        result.append(
            dict(
                context=key[0],
                seed=key[1],
                heldout_rows=world["heldout_rows"],
                heldout_positive=world["heldout_positive"],
                world={k: world[k] for k in ("tp", "fn", "fp", "tn")},
                body={k: body[k] for k in ("tp", "fn", "fp", "tn")},
            )
        )
    if any(digest(Path(p)) != h for p, h in pins.items()):
        raise ValueError("model or audit changed during comparison")
    args.output.mkdir()
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_RISK_FEATURE_FRAME_MATCHED_HOLDOUT_NOT_CONTROL",
            source_head=args.expected_head,
            dataset_hash=args.dataset_hash,
            input_hashes=pins,
            rows=result,
            contexts=16,
            seeds=3,
            threshold=0.5,
            per_transition_results_not_independent_matches=True,
            safety_certificate=False,
            controller_interface=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
