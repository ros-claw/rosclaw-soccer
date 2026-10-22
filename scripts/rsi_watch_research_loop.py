"""Observe multiple declared SIM campaigns; never launch or retry physics."""

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from rosclaw_soccer.rsi.research_loop_dependencies import annotate_dependencies
from rosclaw_soccer.rsi.research_loop_dispatch import dispatch_state
from rosclaw_soccer.rsi.research_loop_watch import assess, collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", nargs=2, action="append", required=True, metavar=("UNIT", "ROOT")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependency-plan", type=Path)
    arguments = parser.parse_args()
    names = [name for name, _ in arguments.campaign]
    if len(names) != len(set(names)):
        parser.error("unique campaign units required")
    output = arguments.output
    previous = json.loads(output.read_text()) if output.exists() else {"campaigns": {}}
    now = time.time()
    campaigns = {}
    for unit, root in arguments.campaign:
        try:
            campaigns[unit] = assess(
                collect(unit, Path(root)), previous["campaigns"].get(unit), now=now
            )
        except Exception as error:
            # One unreadable campaign must not make the other worker disappear.
            campaigns[unit] = dict(
                phase="OBSERVATION_FAILED",
                error_type=type(error).__name__,
                error=str(error),
                next_agent_action="REPAIR_OBSERVER_NOT_PHYSICS_RETRY",
                goal_complete=False,
                hardware_authorized=False,
            )
    plan_hash = None
    if arguments.dependency_plan is not None:
        path = arguments.dependency_plan
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 262144:
            raise ValueError("bounded ordinary read-only dependency declaration required")
        raw = path.read_bytes()
        plan_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
        campaigns = annotate_dependencies(campaigns, json.loads(raw))
    report = dict(
        schema="soccer.rsi.multi_campaign_watch.v1",
        time_unix=now,
        campaigns=campaigns,
        dispatch=dispatch_state(campaigns),
        observer_only=True,
        agent_wakeup_provided_by_this_script=False,
        scientific_successor_automated=False,
        dependency_plan_hash=plan_hash,
        dependency_plan_is_scientific_verification=False,
        goal_complete=False,
        hardware_authorized=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
