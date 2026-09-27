"""Fit a bounded SIM_ONLY speed selector from consumed paired Isaac reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contact_speed_actor import fit_contact_speed_actor
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _verified_report(path: Path) -> dict[str, Any]:
    if path.name != "report.json" or not path.is_file():
        raise ValueError("existing Isaac report.json required")
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    stored = report.pop("report_hash")
    if stored != hash_json(report):
        raise ValueError(f"report integrity mismatch: {path}")
    trace = path.with_name("trajectory.npz")
    if hash_bytes(trace.read_bytes()) != report["trajectory_hash"]:
        raise ValueError(f"trajectory integrity mismatch: {trace}")
    report["report_hash"] = stored
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.pair) < 4:
        parser.error("new actor path and at least four consumed report pairs required")
    pairs = [(_verified_report(slow), _verified_report(fast)) for slow, fast in args.pair]
    actor = fit_contact_speed_actor(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(actor, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(actor, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
