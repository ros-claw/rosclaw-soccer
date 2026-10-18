"""Read-only next-work handoff after all directional pilot diagnostics.

Does not simulate, retry, learn, promote, wake an agent or launch work.
"""

import argparse
import hashlib
import json
from pathlib import Path

from rosclaw_soccer.rsi.contact_exit_pilot_handoff import contact_exit_pilot_handoff
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def digest(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "summary", "diagnostic", "dense", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("unused once-only handoff output required")
    pins = {str(args.config): digest(args.config)}
    source = Path(__file__).resolve().parents[1]
    for path in (Path(__file__), source / "src/rosclaw_soccer/rsi/contact_exit_pilot_handoff.py"):
        pins[str(path)] = digest(path)

    def sealed(path):
        value = json.loads(Path(path).read_text())
        if value.get("report_hash") != hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        ):
            raise ValueError("original complete input seal required")
        pins[str(path)] = digest(path)
        for filename, expected in value.get("input_file_hashes", {}).items():
            if digest(filename) != expected:
                raise ValueError("original audited input/source file changed")
            pins[filename] = expected
        return value

    config = json.loads(args.config.read_text())
    complete = sealed(Path(config["output_root"]) / "complete.json")
    summary, diagnostic, dense = (sealed(p) for p in (args.summary, args.diagnostic, args.dense))
    surface = config.get("use_mesh_surface_swing", False)
    if type(surface) is not bool:
        raise ValueError("explicit boolean surface handoff scope required")
    if (
        diagnostic.get("schema")
        != (
            "soccer.rsi.mesh_surface_contact_diagnostic_complete.v1"
            if surface
            else "soccer.rsi.directional_compliance_contact_diagnostic_complete.v1"
        )
        or diagnostic.get("metrics_hash") != dense["report_hash"]
        or diagnostic.get("upstream_readback_hash") != summary["terminal_readback_hash"]
        or diagnostic.get("new_fixed_control_replay_substeps") != (18000 if surface else 12000)
        or diagnostic.get("new_closed_loop_policy_trials") != 0
        or diagnostic.get("new_optimizer_updates") != 0
    ):
        raise ValueError("same whole diagnostic and terminal-readback chain required")
    result = contact_exit_pilot_handoff(summary, complete, dense, config["baseline_receipts"])
    if any(digest(path) != expected for path, expected in pins.items()):
        raise ValueError("handoff input/source changed during classification")
    result["input_file_hashes"] = pins
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result["next_agent_action"], result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
