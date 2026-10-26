"""Bounded, batched anchor screen of every football train/retention pair."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--prior-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.output.exists() or a.prior.is_symlink() or digest(a.prior) != a.prior_hash:
        raise ValueError("unused output and exact prior completed report required")
    prior = json.loads(a.prior.read_text())
    if (
        prior["status"] != "FIRST_TOUCH_DATA_SPLIT_SCREEN_COMPLETED_NOT_INDEPENDENCE_PROOF"
        or prior["audit"]["total_clips"] != 1203
        or prior["audit"]["split_counts"] != {"train": 973, "retention": 230}
    ):
        raise ValueError("exact completed declared football source inventory required")
    import numpy as np

    from rosclaw_soccer.data.motion_anchor_guard import MotionAnchorGuard
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.first_touch_teacher_portfolio import (
        _CATEGORY_BY_DIRECTORY,
        _EXPECTED_COLUMNS,
        MotionDecodeClip,
        _frame_count_and_header,
        _hash_file,
        _split_for_path,
    )

    root = a.dataset.resolve()
    football_root = root / "extracted_s118" / "3.3.3.Football"
    clips = []
    for directory, category in _CATEGORY_BY_DIRECTORY.items():
        for path in sorted((football_root / directory).glob("*.csv")):
            count, header = _frame_count_and_header(path)
            if header != _EXPECTED_COLUMNS or not 8 <= count <= 60000:
                raise ValueError("bounded canonical motion frames and joint header required")
            relative = path.relative_to(root).as_posix()
            clips.append(
                MotionDecodeClip(
                    relative, _hash_file(path), category, _split_for_path(relative), count
                )
            )
    if (
        hash_json([asdict(c) for c in clips]) != prior["audit"]["inventory_hash"]
        or sum(c.split == "train" for c in clips) != 973
        or sum(c.split == "retention" for c in clips) != 230
    ):
        raise ValueError("dataset inventory changed; no split screen started")
    train = [c for c in clips if c.split == "train"]
    held = [c for c in clips if c.split == "retention"]
    batch_size = 64
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="motiondecode.full_batched_anchored_split_screen.v1",
            source_head=a.expected_head,
            prior_hash=a.prior_hash,
            inventory_hash=prior["audit"]["inventory_hash"],
            train_count=len(train),
            retention_count=len(held),
            train_batch_size=batch_size,
            window=8,
            stride=32,
            decimals=5,
            translation_columns=[0, 1],
            bounded_candidate_and_hit_budgets=True,
            filter_incomplete_not_independence_proof=True,
            retention_read_for_dedup_not_task_outcomes=True,
            optimization_updates=0,
            native_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )

    def load(clip):
        path = root / clip.relative_path
        if path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("bounded non-symlink CSV required")
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        if (
            values.shape != (clip.frame_count, len(_EXPECTED_COLUMNS))
            or _hash_file(path) != clip.source_hash
        ):
            raise ValueError(f"source changed during screen: {clip.relative_path}")
        return values

    records = []
    current_batch = None
    current_clip = None
    try:
        for start in range(0, len(train), batch_size):
            current_batch = start // batch_size
            selected = train[start : start + batch_size]
            guard = MotionAnchorGuard(
                {clip.relative_path: load(clip) for clip in selected}, translation_columns=(0, 1)
            )
            receipts = []
            for clip in held:
                current_clip = clip.relative_path
                row = guard.inspect(clip.relative_path, load(clip))
                receipts.append({**row, "source_hash": clip.source_hash})
            current_clip = None
            batch = dict(
                batch=current_batch,
                train_count=len(selected),
                retention_count=len(receipts),
                index_entries=guard.index_entries,
                receipts=receipts,
            )
            records.append(batch)
            write_once(a.output / f"batch-{current_batch:02d}.json", batch)
            print(
                json.dumps(
                    dict(
                        batch=current_batch,
                        completed=len(records),
                        total=(len(train) + 63) // 64,
                        hits=sum(r["anchor_hits"] for r in receipts),
                    )
                ),
                flush=True,
            )
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="ALL_TRAIN_RETENTION_ANCHORED_SCREEN_COMPLETED_NOT_INDEPENDENCE_PROOF",
                source_head=a.expected_head,
                prior_hash=a.prior_hash,
                inventory_hash=prior["audit"]["inventory_hash"],
                records=records,
                train_count=len(train),
                retention_count=len(held),
                source_independence_proven=False,
                matching_filter_incomplete=True,
                optimization_updates=0,
                native_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="FULL_SPLIT_SCREEN_FAILED_CLOSED",
                source_head=a.expected_head,
                prior_hash=a.prior_hash,
                completed_batches=len(records),
                current_batch=current_batch,
                current_clip=current_clip,
                error_type=type(error).__name__,
                reason=str(error),
                optimization_updates=0,
                native_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
