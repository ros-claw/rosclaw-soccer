"""Observe multiple declared SIM campaigns; never launch or retry physics."""

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

from rosclaw_soccer.rsi.research_loop_continuity import continuity_state
from rosclaw_soccer.rsi.research_loop_dependencies import annotate_dependencies
from rosclaw_soccer.rsi.research_loop_dispatch import dispatch_state
from rosclaw_soccer.rsi.research_loop_watch import assess, collect


def resume_campaigns(path: Path):
    """Read the current checkpoint's observed jobs, never executable commands.

    Read once per tick so a successor does not require editing the timer unit.
    A changed pointer is metadata, not a reason to clear the idle deadline.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError("ordinary resume pointer required")
    with path.open("rb") as stream:
        raw = stream.read(262145)
    if len(raw) > 262144:
        raise ValueError("bounded resume pointer required")
    pointer = json.loads(raw)
    if (
        type(pointer) is not dict
        or pointer.get("schema") != "soccer.agent.resume_pointer.v1"
        or pointer.get("goal_status") != "active"
        or type(pointer.get("observation_campaigns")) is not list
        or len(pointer["observation_campaigns"]) > 128
    ):
        raise ValueError("active resume pointer with explicit bounded observed job list required")
    campaigns = []
    for row in pointer["observation_campaigns"]:
        if (
            type(row) is not dict
            or set(row) != {"unit", "root"}
            or type(row["unit"]) is not str
            or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.@-]{0,190}\.service", row["unit"]) is None
            or type(row["root"]) is not str
            or not row["root"]
            or "\x00" in row["root"]
            or not Path(row["root"]).is_absolute()
            or ".." in Path(row["root"]).parts
        ):
            raise ValueError("exact read-only service/root declaration required; no commands")
        campaigns.append((row["unit"], row["root"]))
    if len({unit for unit, _ in campaigns}) != len(campaigns):
        raise ValueError("unique observed service units required")
    return campaigns, "sha256:" + hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--campaign", nargs=2, action="append", metavar=("UNIT", "ROOT"))
    selection.add_argument("--resume-pointer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependency-plan", type=Path)
    arguments = parser.parse_args()
    pointer_hash = None
    selected = arguments.campaign
    if arguments.resume_pointer is not None:
        selected, pointer_hash = resume_campaigns(arguments.resume_pointer)
    names = [name for name, _ in selected]
    if len(names) != len(set(names)):
        parser.error("unique campaign units required")
    output = arguments.output
    previous = json.loads(output.read_text()) if output.exists() else {"campaigns": {}}
    now = time.time()
    campaigns = {}
    for unit, root in selected:
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
    dispatch = dispatch_state(campaigns)
    report = dict(
        schema="soccer.rsi.multi_campaign_watch.v1",
        time_unix=now,
        campaigns=campaigns,
        dispatch=dispatch,
        continuity=continuity_state(dispatch, previous.get("continuity"), now=now),
        observer_only=True,
        agent_wakeup_provided_by_this_script=False,
        scientific_successor_automated=False,
        dependency_plan_hash=plan_hash,
        resume_pointer_hash=pointer_hash,
        resume_pointer_is_scientific_progress=False,
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
