"""Audit a second candidate execution against every saved original array."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def compare_arrays(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
    if set(first) != set(second) or not first:
        raise ValueError("exact nonempty trajectory array keys required")
    for key in first:
        a, b = first[key], second[key]
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(f"repeat trajectory differs: {key}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("first", "repeat", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("first-hash", "repeat-hash", "trial-head", "source-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    paths = [a.first / "complete.json", a.repeat / "complete.json"]
    hashes = [a.first_hash, a.repeat_hash]
    if any(p.is_symlink() or digest(p) != h for p, h in zip(paths, hashes, strict=True)):
        raise ValueError("exact candidate and repeat receipts required")
    first, repeat = [json.loads(p.read_text()) for p in paths]
    for data in (first, repeat):
        if (
            data.get("status") != "RELAY_NONDEFAULT_SMOOTH_RETURN_PHYSICAL_CANDIDATE_NOT_SUCCESS"
            or data.get("source_head") != a.trial_head
            or data.get("source_hash") != a.source_hash
            or len(data.get("rows", [])) != 24
            or any(
                data.get(k) is not False
                for k in (
                    "hardware_authorized",
                    "promotion_authorized",
                    "goal_complete",
                    "independent_fresh",
                )
            )
        ):
            raise ValueError("same unpromoted development trial required")
    for key in set(first) | set(repeat):
        if key != "rows" and first.get(key) != repeat.get(key):
            raise ValueError(f"repeat receipt metadata differs: {key}")
    first_rows = {r["index"]: r for r in first["rows"]}
    repeat_rows = {r["index"]: r for r in repeat["rows"]}
    if len(first_rows) != 24 or set(first_rows) != set(repeat_rows):
        raise ValueError("24 distinct identical case indices required")
    rows = []
    for index in sorted(first_rows):
        one, two = first_rows[index], repeat_rows[index]
        ignored = {"physical_trajectory_hash", "physical_trajectory"}
        if {k: v for k, v in one.items() if k not in ignored} != {
            k: v for k, v in two.items() if k not in ignored
        }:
            raise ValueError("repeat case outcome differs")
        arrays = []
        for directory, row in ((a.first, one), (a.repeat, two)):
            name = row["physical_trajectory"]
            path = directory / name
            if (
                Path(name).name != name
                or path.is_symlink()
                or digest(path) != row["physical_trajectory_hash"]
            ):
                raise ValueError("unchanged physical evidence required")
            with np.load(path, allow_pickle=False) as archive:
                arrays.append({k: archive[k] for k in archive.files})
            if digest(path) != row["physical_trajectory_hash"]:
                raise ValueError("trajectory changed during audit")
        compare_arrays(*arrays)
        rows.append(dict(index=index, arrays_exact=True, controls=one["consecutive_controls"]))
    if any(digest(p) != h for p, h in zip(paths, hashes, strict=True)):
        raise ValueError("candidate or repeat receipt changed during audit")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RETURN_SMOOTH_SAME_SOURCE_REPEAT_EXACT_NOT_FRESH",
            source_head=a.expected_head,
            trial_head=a.trial_head,
            source_hash=a.source_hash,
            first_hash=a.first_hash,
            repeat_hash=a.repeat_hash,
            rows=rows,
            independent_fresh=False,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
