"""New fixed Fresh contexts, qualified seed0 binding, and consumed-context exclusion."""

import hashlib
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.paid_ancestor_study import case_condition, load_ancestor_plan
from rosclaw_soccer.rsi.paid_fresh_protocol import (
    PROTOCOL,
    ROUTER_FRESH_PROTOCOL,
    fresh_lifecycle_cases,
)
from rosclaw_soccer.sim.contracts import hash_json


def condition_hash(c: dict[str, Any]) -> str:
    return str(hash_json(case_condition(c)))


def geometry_hash(c: dict[str, Any]) -> str:
    return str(hash_json({k: c[k] for k in ("source", "ball_world_initial", "goal_world")}))


def router_fresh_cases(
    sources: list[dict[str, Any]], consumed: list[dict[str, Any]], train: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(consumed) != 32 or len(train) != 20:
        raise ValueError("all consumed32 and fitted20 context declarations required")
    if consumed != fresh_lifecycle_cases(sources):
        raise ValueError("consumed v1 must use these exact frozen source anchors")
    source_fields = (
        "case",
        "source",
        "profile",
        "ball_mass_kg",
        "ball_world_initial",
        "goal_world",
    )
    clean = [{k: c[k] for k in source_fields} for c in sources]
    excluded = sorted({condition_hash(c) for c in consumed + train})
    geometries = sorted({geometry_hash(c) for c in clean})
    if len(excluded) != 52 or len(geometries) != 16:
        raise ValueError("complete distinct consumed and fitted context exclusions required")
    cases = fresh_lifecycle_cases(clean, protocol=ROUTER_FRESH_PROTOCOL)
    actual = [condition_hash(c) for c in cases]
    if (
        len(set(actual)) != 32
        or set(actual) & set(excluded)
        or any(geometry_hash(c) in geometries for c in cases)
    ):
        raise ValueError("Fresh overlap: reject entire fixed declaration, never resample")
    return cases, dict(
        source_cases=clean,
        excluded_condition_hashes=excluded,
        excluded_original_geometry_hashes=geometries,
        resampled_cases=0,
        parent_semantics="exact_current_progressive_parent",
        candidate_semantics="frozen_ancestral_router_seed0",
    )


def router_fresh_commitment(
    *,
    student_root: Path,
    student_hash: str,
    sources: list[dict[str, Any]],
    review_root: Path,
    review_hash: str,
    previous_root: Path,
    previous_hash: str,
    train_root: Path,
    train_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    from rosclaw_soccer.rsi.paid_fresh_evidence import load_fresh_declaration

    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    rp, dp = review_root / "complete.json", review_root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != review_hash
        or (review_root / "failed.json").exists()
    ):
        raise ValueError("exact closed router frontier review required")
    review, declaration = [json.loads(f.read_text()) for f in (rp, dp)]
    if (
        review["status"] != "ROUTER_FRONTIER_PASS_PENDING_NEW_FRESH"
        or review["all_three_seeds_pass"] is not True
        or review["primary_seed"] != 0
        or review["Fresh_rows_read"] != 0
        or declaration["primary_seed"] != 0
        or declaration["seeds"] != [0, 1, 2]
        or student_root != Path(declaration["pipelines"][0]) / "trained"
        or review["input_file_hashes"].get(str(student_root / "complete.json")) != student_hash
        or digest(student_root / "complete.json") != student_hash
        or any(review[k] is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError("predeclared primary seed0 with all three qualified frontiers required")
    # Only the prior declaration is consumed, never its outcome labels/actions.
    prior_schema = json.loads((previous_root / "declaration.json").read_text())["schema"]
    if prior_schema != PROTOCOL:
        raise ValueError("only consumed fixed v1 declaration is the exclusion source")
    _, previous = load_fresh_declaration(previous_root, previous_hash)
    _, train = load_ancestor_plan(train_root, train_hash)
    cases, metadata = router_fresh_cases(sources, previous["cases"], train["cases"])
    metadata.update(
        qualification_review_root=str(review_root),
        qualification_review_hash=review_hash,
        previous_fresh_root=str(previous_root),
        previous_fresh_hash=previous_hash,
        new_train_plan_root=str(train_root),
        new_train_plan_hash=train_hash,
    )
    pins = dict(review["input_file_hashes"])
    for root in (review_root, previous_root, train_root):
        for name in ("complete.json", "declaration.json"):
            f = root / name
            pins[str(f)] = digest(f)
    return cases, metadata, pins


def validate_router_fresh_declaration(d: dict[str, Any]) -> None:
    cases, metadata, _ = router_fresh_commitment(
        student_root=Path(d["student_root"]),
        student_hash=d["student_hash"],
        sources=d["source_cases"],
        review_root=Path(d["qualification_review_root"]),
        review_hash=d["qualification_review_hash"],
        previous_root=Path(d["previous_fresh_root"]),
        previous_hash=d["previous_fresh_hash"],
        train_root=Path(d["new_train_plan_root"]),
        train_hash=d["new_train_plan_hash"],
    )
    if d["cases"] != cases or any(d.get(k) != v for k, v in metadata.items()):
        raise ValueError("exact new generator and exclusions must be independently recomputed")
