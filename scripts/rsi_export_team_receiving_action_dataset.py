"""Export authenticated historical team receiving rows as SIM training data.

The source has report-level physical trace hashes, not raw state trajectories;
this export is not independent physics verification or a policy promotion.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.team_receiving_action_dataset import import_team_action_map
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1])
        or args.source.is_symlink()
        or not args.source.is_file()
    ):
        raise ValueError("unused external output and regular source required")
    source_bytes_hash = digest(args.source)
    dataset = import_team_action_map(json.loads(args.source.read_text()))
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_bytes_hash=source_bytes_hash,
            source_report_hash=dataset["source_hash"],
            source_head=args.expected_head,
            courses=12,
            actions=16,
            episodes=192,
            source_report_only_not_independent_replay=True,
            hardware_authorized=False,
        ),
    )
    path = args.output / "dataset.npz"
    np.savez_compressed(
        path,
        features=dataset["features"],
        action_offsets=dataset["actions"],
        rewards=dataset["rewards"],
        safe=dataset["safe"],
        eligible=dataset["eligible"],
    )
    if digest(args.source) != source_bytes_hash:
        raise ValueError("source report changed during export")
    write_once(
        args.output / "complete.json",
        dict(
            status="TEAM_RECEIVING_PAIRED_DATASET_REPORT_ONLY_NOT_PROMOTED",
            source_bytes_hash=source_bytes_hash,
            source_report_hash=dataset["source_hash"],
            source_head=args.expected_head,
            dataset_hash=digest(path),
            courses=dataset["courses"],
            action_ids=dataset["action_ids"],
            physical_trace_hashes=dataset["trace_hashes"],
            rewards=dataset["rewards"].tolist(),
            safe=dataset["safe"].tolist(),
            eligible=dataset["eligible"].tolist(),
            source_raw_trajectories_available=False,
            independent_replay_verified=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
