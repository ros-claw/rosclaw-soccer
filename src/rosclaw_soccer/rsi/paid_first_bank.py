"""Versioned original16 fixtures with explicit measured velocities, not new outcomes."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "paid.full_entry.original16.first_enabled_fixture.v1"
STATUS = "PAID_FIRST_ENABLED_ORIGINAL16_BANK_PREREGISTERED_NO_ROLLOUT"
EXAM_STATUS = "PAID_FULL_ENTRY_ORIGINAL16_WORLDS_CLOSED_PENDING_AUDIT"
AUDIT_STATUS = "PAID_FULL_ENTRY_ORIGINAL16_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS"


def load_motor_exam_plan(root: Path, expected: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rp, dp = root / "complete.json", root / "declaration.json"

    def digest(f: Path) -> str:
        return "sha256:" + hashlib.sha256(f.read_bytes()).hexdigest()

    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != expected
        or (root / "failed.json").exists()
    ):
        raise ValueError("exact closed motor exam plan required")
    r = json.loads(rp.read_text())
    if r.get("status") != STATUS:
        from rosclaw_soccer.rsi.paid_ancestor_study import load_ancestor_plan

        return load_ancestor_plan(root, expected)
    d = json.loads(dp.read_text())
    if (
        d.get("schema") != SCHEMA
        or d.get("bank") not in ("train", "dev")
        or r.get("bank") != d["bank"]
        or r.get("declaration_hash") != digest(dp)
        or r.get("source_head") != d.get("source_head")
        or r.get("body_scene_hash") != d.get("body_scene_hash")
        or d.get("cases_predeclared") != 16
        or r.get("cases_preflighted") != 16
        or r.get("original_seven_entry_fields_exact") is not True
        or r.get("explicit_velocity_preserves_full_native_entry") is not True
        or r.get("own_native_steps") != 0
        or r.get("own_neural_rows") != 0
        or any(
            o.get(k) is not False
            for o in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete zero-step original-bank fixture proof required")
    cases = d.get("cases", [])
    if (
        len(cases) != 16
        or [c["case"] for c in cases] != list(range(16))
        or len({c["source"] for c in cases}) != 8
    ):
        raise ValueError("all16 distinct ordered original fixtures required")
    entries = r.get("entries", [])
    if len(entries) != 16 or [e.get("case") for e in entries] != list(range(16)):
        raise ValueError("all16 actual measured entries required")
    for c, e in zip(cases, entries, strict=True):
        if c.get("profile") != "cached-standard" or c.get("ball_mass_kg") != 0.43:
            raise ValueError("unchanged original physical recipe required")
        for name, width in (
            ("ball_world_initial", 3),
            ("goal_world", 3),
            ("velocity_world_initial", 2),
        ):
            x = np.asarray(c[name], float)
            if x.shape != (width,) or not np.isfinite(x).all():
                raise ValueError("finite explicit original geometry/velocity required")
        if (
            c["ball_world_initial"][2] != 0.11
            or c["goal_world"][2] != 0.11
            or np.linalg.norm(c["velocity_world_initial"]) > 0.6
        ):
            raise ValueError("unchanged sphere height/bounded measured velocity required")
        if (
            e.get("position") != c["ball_world_initial"]
            or e.get("velocity") != c["velocity_world_initial"]
        ):
            raise ValueError("measured entry and explicit fixture differ")
    return r, d
