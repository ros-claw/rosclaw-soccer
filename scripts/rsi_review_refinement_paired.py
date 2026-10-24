"""Read-only scientific successor after both paired native integrations close."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source-root", "own-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from rosclaw_soccer.rsi.refinement_paired_review import paired_review
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only review output required")
    pins, reports = {}, []
    for root in (a.source_root, a.own_root):
        if root.is_symlink() or (root / "failed.json").exists():
            raise ValueError("ordinary closed native exam required")
        path, dp = root / "complete.json", root / "declaration.json"
        pins[str(path)], pins[str(dp)] = file_hash(path), file_hash(dp)
        r, d = [json.loads(f.read_text()) for f in (path, dp)]
        if (
            r["source_head"] != a.producer_head
            or r["declaration_hash"] != file_hash(dp)
            or d["input_file_hashes"] != r["input_file_hashes"]
            or any(r[k] != d[k] for k in ("config", "source_head", "training_hash"))
        ):
            raise ValueError("exact predeclared native producer and declaration required")
        for k, v in r["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting reviewed ancestry")
            pins[k] = v
        reports.append(r)
    if reports[1]["source_hash"] != pins[str(a.source_root / "complete.json")]:
        raise ValueError("own replay must bind exact completed source")
    source_pins(pins)
    check_pins(pins)
    result = paired_review(*reports)
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        result
        | dict(
            source_head=a.expected_head,
            producer_head=a.producer_head,
            input_file_hashes=pins,
            source_hash=pins[str(a.source_root / "complete.json")],
            own_hash=pins[str(a.own_root / "complete.json")],
        ),
    )


if __name__ == "__main__":
    main()
