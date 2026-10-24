"""Authenticate source/own pairs before comparing physical motor growth.

Callers must authenticate receipt hashes and merge their input pins first.
This only reads evidence: it never loads a motor, executes physics or promotes.
"""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.dual_motor_checkpoint import digest


def qualified_native_growth_pair(
    passed: dict[str, Any],
    retained: dict[str, Any],
    *,
    kind: str,
    pins: Mapping[str, str],
    resolve: Callable[[str], tuple[Path, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        passed["status"] != "TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH"
        or retained["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
        or len(passed["records"]) != 80
        or retained["worlds"] != 44
        or retained["independent_whole_world_replay"] is not True
        or retained["checkpoint_kind"] != kind
        or passed["training_hash"] != retained["training_hash"]
    ):
        raise ValueError("same motor with complete independent pass80 and retention44 required")
    pass_path, source = resolve(passed["producer_hash"])
    old_path, old_source = resolve(retained["source_hash"])
    _, training = resolve(passed["training_hash"])
    if (
        training["checkpoint_kind"] != kind
        or source["training_hash"] != passed["training_hash"]
        or source["checkpoint_kind"] != kind
        or [{k: v for k, v in row.items() if k != "artifacts"} for row in source["records"]]
        != passed["records"]
        or source["summary"] != passed["summary"]
        or source["counts"] != passed["counts"]
        or old_source["training_hash"] != passed["training_hash"]
        or training["checkpoint_hash"] != retained["checkpoint_hash"]
        or old_source["checkpoint_hash"] != training["checkpoint_hash"]
        or len(old_source["records"]) != 44
        or old_source["whole_repeats_exact"] is not True
        or retained["results"]
        != [
            dict(
                context=x["context"],
                repeat=x["repeat"],
                quality=x["healthy_quality"],
                body_failed=x["result"]["body_failed"],
            )
            for x in old_source["records"]
        ]
        or retained["counts"]["native"] != old_source["actual_native_steps"]
        or retained["counts"]["neural"] != old_source["actual_actor_decisions"]
        or any(
            d[k] is not False
            for d in (passed, retained, source, old_source, training)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("exact unpromoted source and independent own results required")

    def declaration(path: Path, report: dict[str, Any]) -> dict[str, Any]:
        dp = path.parent / "declaration.json"
        if (
            dp.is_symlink()
            or pins.get(str(dp)) != digest(dp)
            or report["declaration_hash"] != digest(dp)
        ):
            raise ValueError("source-bound ordinary original task declaration required")
        return dict(json.loads(dp.read_text()))

    pd, rd = declaration(pass_path, source), declaration(old_path, old_source)
    if (
        pd["protocol"] != "dual256.team_pass_unassisted_TRAIN80.v1"
        or pd["training_hash"] != passed["training_hash"]
        or pd["checkpoint_kind"] != kind
        or pd["maximum_worlds"] != 80
        or pd["maximum_native_steps"] != 320000
        or pd["maximum_actor_calls"] != 64000
        or pd["formal_pass_receive_controls"] != 300
        or pd["required_A_neural_recovery_controls"] != 100
        or pd["teacher_enabled"] is not False
        or rd["protocol"] != "dual256.unassisted_TRAIN20_and_consumed_run2.v1"
        or rd["training_hash"] != passed["training_hash"]
        or rd["checkpoint_kind"] != kind
        or len(rd["contexts"]) != 22
        or rd["repeats"] != 2
        or rd["hidden_width"] != 256
        or rd["maximum_native_steps"] != 176000
        or rd["maximum_actor_decisions"] != 17600
        or rd["online_teacher_enabled"] is not False
        or rd["Fresh_cases"] != 0
        or any(
            d[k] is not False
            for d in (pd, rd)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("unchanged original pass80/retention44 protocols and budgets required")
    return training, rd
