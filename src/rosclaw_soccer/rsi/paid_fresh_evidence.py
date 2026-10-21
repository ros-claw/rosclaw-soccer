"""Hash-bound Fresh declaration and frozen three-arm simulation admission."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_fresh_protocol import CASE_SEEDS, ROUTER_FRESH_PROTOCOL


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_fresh_declaration(root: Path, expected: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != expected
        or (root / "failed.json").exists()
    ):
        raise ValueError("exact closed Fresh preflight required")
    r, d = json.loads(rp.read_text()), json.loads(dp.read_text())
    protocol = d.get("schema")
    if protocol not in CASE_SEEDS:
        raise ValueError("explicit versioned Fresh protocol required")
    if protocol == ROUTER_FRESH_PROTOCOL:
        from rosclaw_soccer.rsi.paid_router_fresh import validate_router_fresh_declaration

        validate_router_fresh_declaration(d)
    if (
        r.get("status") != "PAID_FRESH_ALL32_CASES_DECLARED_AND_PREFLIGHT_PASS_NO_ROLLOUT"
        or r.get("declaration_hash") != digest(dp)
        or r.get("source_head") != d.get("source_head")
        or r.get("protocol") != protocol
        or d.get("arms") != ["r2", "parent", "progressive"]
        or d.get("repeats") != 2
        or d.get("skills") != 3
        or d.get("standing_controls") != 100
        or d.get("body_resets") != 0
        or type(d.get("candidate_seed_index")) is not int
        or d["candidate_seed_index"] != 0
        or d.get("candidate_frozen_before_generation") is not True
        or d.get("candidate_seed_chosen_on_fresh") is not False
        or d.get("case_generation_seed") != CASE_SEEDS[protocol]
        or d.get("evaluation")
        != dict(
            candidate_stable_required=32,
            no_case_regression_vs_R2=True,
            no_case_regression_vs_parent=True,
            third_gain_vs_R2_minimum=4,
            third_gain_vs_parent_minimum=1,
            all_independent_audits_required=True,
        )
        or r.get("own_native_steps") != 0
        or r.get("Fresh_episodes_executed") != 0
        or any(obj.get("Fresh_episodes_fit") != 0 for obj in (r, d))
        or any(
            obj.get(k) is not False
            for obj in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("unchanged complete Fresh preregistration required")
    cases, entries = d.get("cases", []), r.get("native_entries", [])
    if (
        len(cases) != 32
        or len(entries) != 32
        or [c.get("case") for c in cases] != list(range(32))
        or [c.get("case") for c in entries] != list(range(32))
    ):
        raise ValueError("every predeclared context must be physically admitted")
    classes = {"near_toward", "near_away", "far_toward", "far_away"}
    clusters: dict[int, set[str]] = {}
    sources: dict[int, str] = {}
    for c, entry in zip(cases, entries, strict=True):
        cluster, role = c.get("source_cluster"), c.get("fresh_class")
        if (
            type(cluster) is not int
            or cluster not in range(8)
            or role not in classes
            or c.get("profile") != "cached-standard"
            or c.get("ball_mass_kg") != 0.43
        ):
            raise ValueError("unchanged32-context coverage required")
        old = clusters.setdefault(cluster, set())
        if role in old or (cluster in sources and sources[cluster] != c["source"]):
            raise ValueError("duplicated/mixed motion-source cluster")
        old.add(role)
        sources[cluster] = c["source"]
        for name, width in (
            ("ball_world_initial", 3),
            ("goal_world", 3),
            ("velocity_world_initial", 2),
        ):
            x = np.asarray(c[name], dtype=float)
            if x.shape != (width,) or not np.isfinite(x).all():
                raise ValueError("finite explicit Fresh geometry required")
        if (
            entry["position"] != c["ball_world_initial"]
            or entry["velocity"] != c["velocity_world_initial"]
            or c["ball_world_initial"][2] != 0.11
            or c["goal_world"][2] != 0.11
            or np.linalg.norm(c["velocity_world_initial"]) > 0.6
        ):
            raise ValueError("declared Fresh entry was not actually admitted")
    if (
        len(sources) != 8
        or len(set(sources.values())) != 8
        or any(v != classes for v in clusters.values())
    ):
        raise ValueError("eight distinct known motion clusters required")
    return r, d


def load_fresh_actor(policy: Path, declaration: dict[str, Any], arm: str) -> tuple[Any, int]:
    from rosclaw_soccer.rsi.paid_contact_residual import FrozenContactResidualActor
    from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student

    if arm not in ("r2", "parent", "progressive"):
        raise ValueError("one of three committed arms required")
    root = Path(declaration["student_root"])
    receipt = json.loads((root / "complete.json").read_text())
    if (
        digest(root / "complete.json") != declaration["student_hash"]
        or receipt["checkpoint_hash"] != declaration["checkpoint_hash"]
        or receipt["parent_hash"] != declaration["parent_hash"]
    ):
        raise ValueError("frozen candidate/parent changed")
    actor, _ = load_paid_student(
        policy,
        checkpoint=Path(receipt["checkpoint_path"]),
        checkpoint_hash=receipt["checkpoint_hash"],
        training_receipt=root / "complete.json",
        training_receipt_hash=declaration["student_hash"],
        scene_hash=declaration["unchanged_body_scene_hash"],
    )
    if arm == "parent":
        if declaration["schema"] == ROUTER_FRESH_PROTOCOL:
            from rosclaw_soccer.rsi.paid_ancestor_router_student import FrozenAncestralRouterActor
            from rosclaw_soccer.rsi.paid_progressive_student import FrozenProgressiveActor

            if type(actor) is not FrozenAncestralRouterActor:
                raise ValueError("explicit admitted router with current compound parent required")
            actor = FrozenProgressiveActor(
                actor.base, actor.residual.parent, actor.get_buffer("action_scale")
            ).eval()
        else:
            actor = FrozenContactResidualActor(
                actor.base, actor.residual.parent, actor.action_scale
            ).eval()
    return actor, 1 if arm == "r2" else 2
