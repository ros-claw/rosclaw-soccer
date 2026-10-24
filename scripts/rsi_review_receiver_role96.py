"""Read-only all96 paired role analysis after BOTH native/law pairs close.

Authenticate exact declarations, all referenced source primitives and inherited
pins. No world allocation, rerun, labels changed, model update, or promotion.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("factual-source-root", "factual-own-root", "source-root", "own-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("factual-own-hash", "producer-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.receiver_role_review import paired_role_review
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only role review output required")
    pins, reports = {}, []
    roots = (a.factual_source_root, a.factual_own_root, a.source_root, a.own_root)
    for root in roots:
        if root.is_symlink() or (root / "failed.json").exists():
            raise ValueError("ordinary closed full native/law reports required")
        rp, dp = root / "complete.json", root / "declaration.json"
        rh, dh = file_hash(rp), file_hash(dp)
        r, d = [json.loads(path.read_text()) for path in (rp, dp)]
        if (
            r["declaration_hash"] != dh
            or r["input_file_hashes"] != d["input_file_hashes"]
            or any(r[k] != d[k] for k in ("config", "source_head", "training_hash"))
        ):
            raise ValueError("exact declared native producer required")
        if root in (a.source_root, a.own_root) and r["source_head"] != a.producer_head:
            raise ValueError("exact predeclared new-role producer required")
        for k, v in r["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting reviewed ancestry")
            pins[k] = v
        pins.update({str(rp): rh, str(dp): dh})
        reports.append(r)
    if pins[str(a.factual_own_root / "complete.json")] != a.factual_own_hash:
        raise ValueError("exact predeclared reused factual own proof required")
    for root, report, audit in (
        (a.factual_source_root, reports[0], reports[1]),
        (a.source_root, reports[2], reports[3]),
    ):
        for r in report["records"]:
            if set(r["artifacts"]) != {"A", "B", "physical", "sampling", "initial"}:
                raise ValueError("complete physical/private primitives required")
            for art in r["artifacts"].values():
                path = root / art["path"]
                if (
                    path.is_symlink()
                    or path.resolve().parent != (root / f"trial{r['index']}").resolve()
                    or audit["input_file_hashes"].get(str(path)) != art["hash"]
                ):
                    raise ValueError(
                        "every exact confined primitive must be pinned in independent audit"
                    )
                if str(path) in pins and pins[str(path)] != art["hash"]:
                    raise ValueError("conflicting primitive hash")
                pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    result = paired_role_review(
        *reports,
        factual_source_hash=pins[str(a.factual_source_root / "complete.json")],
        source_hash=pins[str(a.source_root / "complete.json")],
    )
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        result
        | dict(
            source_head=a.expected_head,
            producer_head=a.producer_head,
            input_file_hashes=pins,
            factual_source_hash=pins[str(a.factual_source_root / "complete.json")],
            factual_own_hash=a.factual_own_hash,
            source_hash=pins[str(a.source_root / "complete.json")],
            own_hash=pins[str(a.own_root / "complete.json")],
        ),
    )


if __name__ == "__main__":
    main()
