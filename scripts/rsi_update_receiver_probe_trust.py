"""Derive one bounded SIM probe from exact paired positive/negative readbacks."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.receiver_probe_trust_update import development_trust_update
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("positive", "negative", "positive-candidate", "negative-candidate", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "positive-hash",
        "negative-hash",
        "positive-candidate-hash",
        "negative-candidate-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    inputs = (
        (a.positive, a.positive_hash),
        (a.negative, a.negative_hash),
        (a.positive_candidate, a.positive_candidate_hash),
        (a.negative_candidate, a.negative_candidate_hash),
    )
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external result required")
    for path, expected in inputs:
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError("exact bound physical input required")
    plus, minus, pc, mc = (json.loads(path.read_text()) for path, _ in inputs)
    positive_vector = [0.0, 0.0, 0.0, 0.1, -0.05, 0.0] * 2
    negative_vector = [-v for v in positive_vector]
    if (
        pc.get("probe_leg_rad") != positive_vector
        or mc.get("probe_leg_rad") != negative_vector
        or any(
            x.get("status") != "MATCHED_RECEIVER_PHYSICAL_PROBE_NOT_CATCH_CERTIFICATE"
            for x in (plus, minus)
        )
        or any(
            x.get("status") != "RELAY_LEG_RESIDUAL_PHYSICAL_PROBE_NOT_FOOTBALL_SUCCESS"
            for x in (pc, mc)
        )
        or plus.get("candidate_hash") != a.positive_candidate_hash
        or minus.get("candidate_hash") != a.negative_candidate_hash
        or any(
            plus.get(k) != minus.get(k) for k in ("baseline_hash", "snapshot_hash", "scene_hash")
        )
        or any(
            pc.get(k) != mc.get(k)
            for k in ("source_head", "source_hash", "audit_hash", "snapshot_hash", "frontier_hash")
        )
        or any(
            x.get(k) is not False
            for x in (plus, minus, pc, mc)
            for k in ("promotion_authorized", "hardware_authorized", "goal_complete")
        )
    ):
        raise ValueError("exact symmetric measured development probes required")
    result = development_trust_update(plus, minus)
    for path, expected in inputs:
        if digest(path) != expected:
            raise ValueError("physical input changed during optimizer update")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            **result,
            source_head=a.expected_head,
            positive_hash=a.positive_hash,
            negative_hash=a.negative_hash,
            positive_candidate_hash=a.positive_candidate_hash,
            negative_candidate_hash=a.negative_candidate_hash,
            baseline_hash=plus["baseline_hash"],
        ),
    )


if __name__ == "__main__":
    main()
