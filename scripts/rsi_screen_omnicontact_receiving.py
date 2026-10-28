"""Screen only declared training motions for reception-like reference events."""

import argparse
import csv
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.omnicontact_receive_candidates import candidate_events
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    root = a.dataset_root.resolve()
    split_path = root / "metadata" / "splits.csv"
    data_root = root / "npz" / "soccer"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or split_path.is_symlink()
        or digest(split_path) != a.split_hash
        or not data_root.is_dir()
    ):
        raise ValueError("unchanged split, soccer corpus, and unused external output required")
    with split_path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    splits = {r["capture_id"]: r["split"] for r in records}
    if len(splits) != len(records) or any(
        s not in ("train", "val", "test") for s in splits.values()
    ):
        raise ValueError("unique declared source splits required")
    paths = sorted(data_root.glob("*/*_with_contact.npz"))
    if not paths or len(paths) > 1000:
        raise ValueError("bounded published soccer corpus required")
    rows, failed, pins = [], [], []
    split_counts = {"train": 0, "val": 0, "test": 0}
    for path in paths:
        capture = path.name.removesuffix("_with_contact.npz")
        if path.is_symlink() or capture not in splits:
            raise ValueError("ordinary file with explicitly declared split required")
        split = splits[capture]
        split_counts[split] += 1
        if split != "train":
            continue
        file_hash = digest(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                fps = np.asarray(data["fps"])
                if fps.shape != (1,) or fps.dtype.kind not in "iu":
                    raise ValueError("declared integer sample rate required")
                events = candidate_events(
                    data["object_pos_w"],
                    data["object_lin_vel_w"],
                    data["body_pos_w"],
                    data["contact_info"],
                    int(fps[0]),
                )
        except (ValueError, KeyError, OSError) as exc:
            failed.append(dict(capture=capture, hash=file_hash, error=str(exc)))
            events = []
        if digest(path) != file_hash:
            raise ValueError("input changed while screening")
        pins.append((path, file_hash))
        rows.append(
            dict(
                capture=capture,
                category=path.parent.name,
                source_hash=file_hash,
                source_relative_path=str(path.relative_to(root)),
                events=events,
            )
        )
    if not rows or digest(split_path) != a.split_hash or any(digest(p) != h for p, h in pins):
        raise ValueError("complete unchanged training source required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="OMNICONTACT_TRAIN_RECEIVING_REFERENCE_SCREEN_NOT_PHYSICS",
            source_head=a.expected_head,
            split_hash=a.split_hash,
            split_counts=split_counts,
            processed_train=len(rows),
            rejected_inputs=failed,
            candidate_event_count=sum(len(r["events"]) for r in rows),
            rows=rows,
            validation_or_test_motion_opened=False,
            reference_only_not_actual_contact=True,
            noncommercial_source=True,
            commercial_publicity_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
