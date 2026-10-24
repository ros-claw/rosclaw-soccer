"""Await all three existing native experiments and bind their complete readback.

No physics retries, training, promotion, or agent wakeup is provided here.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.native_role_learning_review import review_native_role_learning
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("new all-seed review output required")
    reports, hashes = [], {}
    deadline = time.monotonic() + 3600
    for seed in (926061, 926062, 926063):
        root = a.root / f"rsi-native8-residual-learning-{seed}-20261025"
        complete = root / "complete.json"
        while not complete.exists():
            state = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    f"rosclaw-native8-residual-learning-{seed}-20261025.service",
                    "-p",
                    "ActiveState",
                    "--value",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            if (
                state.stdout.strip() not in ("active", "activating")
                or (root / "failed.json").exists()
                or time.monotonic() > deadline
            ):
                raise ValueError("incomplete/failed experiment retained; no automatic retry")
            time.sleep(2)
        if complete.is_symlink() or (root / "failed.json").exists():
            raise ValueError("ordinary completed experiments only")
        report = json.loads(complete.read_text())
        fit_path = root / "fit/complete.json"
        if fit_path.is_symlink() or digest(fit_path) != report["fit_hash"]:
            raise ValueError("fit commitment changed")
        fit = json.loads(fit_path.read_text())
        if (
            report["seed"] != seed
            or fit["seed"] != seed
            or fit["source_head"] != report["source_head"]
            or fit["numeric_report"] != report["numeric_report"]
        ):
            raise ValueError("own source and actual fit counters required")
        check_pins(fit["input_file_hashes"])
        for item in (fit["policy"], fit["checkpoint"], fit["dataset"]):
            path = fit_path.parent / item["path"]
            if (
                path.parent.resolve() != fit_path.parent.resolve()
                or path.is_symlink()
                or digest(path) != item["hash"]
            ):
                raise ValueError("learned weights/checkpoint changed")
        for case in report["results"]:
            path = root / case["course"] / "complete.json"
            if path.is_symlink() or digest(path) != case["source_hash"]:
                raise ValueError("physical case report changed")
            source = json.loads(path.read_text())
            if not source["fresh_reexecution_exact"] or len(source["records"]) != 4:
                raise ValueError("all paired source/fresh physics required")
            check_pins(source["input_file_hashes"])
            for row in source["records"]:
                path = root / case["course"] / row["artifact"]["path"]
                if (
                    path.parent.resolve() != (root / case["course"]).resolve()
                    or path.is_symlink()
                    or digest(path) != row["artifact"]["hash"]
                ):
                    raise ValueError("physical arrays changed")
        reports.append(report)
        hashes[str(complete)] = digest(complete)
    result = review_native_role_learning(reports)
    clean(a.expected_head)
    result.update(input_complete_hashes=hashes, source_head=a.expected_head)
    write_once(a.output, result)


if __name__ == "__main__":
    main()
