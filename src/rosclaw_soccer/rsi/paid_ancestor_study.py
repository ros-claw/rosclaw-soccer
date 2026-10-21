"""New TRAIN contexts for same-entry ancestor competence, not consumed Fresh fit."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEDULES = {
    "grown": ("base", "student", "student"),
    "ancestor_third": ("base", "student", "base"),
    "r2_all": ("base", "base", "base"),
}


def ancestor_train_cases(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if (
        len(sources) != 16
        or [c["case"] for c in sources] != list(range(16))
        or len({c["source"] for c in sources}) != 8
        or any(sources[2 * i]["source"] != sources[2 * i + 1]["source"] for i in range(8))
    ):
        raise ValueError("all original TRAIN source entries required")
    rng = np.random.Generator(np.random.PCG64(2026102211))
    cases: list[dict[str, Any]] = []
    entries = [(row, (-1 if i % 2 == 0 else 1), "balanced") for i, row in enumerate(sources)]
    target = [c for c in sources if c["source"] == "soccer-standard-009_right.npz"]
    if len(target) != 2:
        raise ValueError("declared source009 pair required, no source replacement")
    entries += [(target[1], 1, "source009_coverage") for _ in range(4)]
    for row, sign, group in entries:
        ball, goal = (
            np.asarray(row["ball_world_initial"], float),
            np.asarray(row["goal_world"], float),
        )
        if (
            row["profile"] != "cached-standard"
            or row["ball_mass_kg"] != 0.43
            or ball.shape != (3,)
            or goal.shape != (3,)
            or not np.isfinite(ball).all()
            or not np.isfinite(goal).all()
            or ball[2] != 0.11
            or goal[2] != 0.11
            or np.linalg.norm(goal[:2]) < 4
        ):
            raise ValueError("unchanged original TRAIN physical recipe required")
        forward = goal[:2] / np.linalg.norm(goal[:2])
        side = np.array([-forward[1], forward[0]])
        position = ball.copy()
        position[:2] += rng.uniform(-0.04, 0.04) * forward + rng.uniform(-0.04, 0.04) * side
        speed = rng.uniform(0.13, 0.29) if group == "balanced" else rng.uniform(0.08, 0.32)
        velocity = sign * speed * forward + rng.uniform(-0.05, 0.05) * side
        angle = rng.uniform(-0.04, 0.04)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        target_goal = goal.copy()
        target_goal[:2] = rot @ target_goal[:2]
        cases.append(
            dict(
                case=len(cases),
                source=row["source"],
                profile="cached-standard",
                ball_mass_kg=0.43,
                ball_world_initial=position.tolist(),
                velocity_world_initial=velocity.tolist(),
                goal_world=target_goal.tolist(),
                cohort=group,
                bank="train",
                not_fresh=True,
            )
        )
    return cases


def case_condition(case: dict[str, Any]) -> dict[str, Any]:
    return {
        k: case[k] for k in ("source", "ball_world_initial", "velocity_world_initial", "goal_world")
    }


def ancestor_outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summaries retain partial legal worlds, including no third kick."""
    return dict(
        stable=sum(r["stable"] for r in rows),
        quality_by_kick=[
            sum(len(r["quality_by_kick"]) > k and r["quality_by_kick"][k] for r in rows)
            for k in range(3)
        ],
    )


def load_ancestor_plan(root: Path, expected_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from rosclaw_soccer.sim.contracts import hash_json

    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or (root / "failed.json").exists()
        or digest(rp) != expected_hash
    ):
        raise ValueError("exact closed all-case TRAIN preflight required")
    r, d = [json.loads(f.read_text()) for f in (rp, dp)]
    if (
        r["status"] != "PAID_TRAIN20_ANCESTOR_STUDY_PREREGISTERED_NO_ROLLOUT"
        or d["schema"] != "paid.ancestor_counterfactual.TRAIN20.v1"
        or d["repeats"] != 2
        or d["shards"] != 4
        or d["cases_per_shard"] != 5
        or d["schedules"] != {k: list(v) for k, v in SCHEDULES.items()}
        or r["declaration_hash"] != digest(dp)
        or r["source_head"] != d["source_head"]
        or r["cases_preflighted"] != 20
        or r["own_native_steps"] != 0
        or r["rollouts"] != 0
        or any(o["bank"] != "train" or o["Fresh_fit_rows"] != 0 for o in (r, d))
        or any(
            o[k] is not False
            for o in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("unchanged TRAIN20 ancestor experiment commitment required")
    cases = d["cases"]
    excluded = d["excluded_context_hashes"]
    if (
        len(cases) != 20
        or [c["case"] for c in cases] != list(range(20))
        or len(excluded) != 32
        or len(set(excluded)) != 32
        or len({c["source"] for c in cases[:16]}) != 8
        or any(c["source"] != "soccer-standard-009_right.npz" for c in cases[16:])
    ):
        raise ValueError("all20 cases and consumed-test exclusion required")
    for c in cases:
        if hash_json(case_condition(c)) in excluded:
            raise ValueError("test/training overlap")
        if (
            c["bank"] != "train"
            or c["not_fresh"] is not True
            or c["profile"] != "cached-standard"
            or c["ball_mass_kg"] != 0.43
        ):
            raise ValueError("new TRAIN-only physical recipe required")
        for key, size in (
            ("ball_world_initial", 3),
            ("goal_world", 3),
            ("velocity_world_initial", 2),
        ):
            x = np.asarray(c[key], float)
            if x.shape != (size,) or not np.isfinite(x).all():
                raise ValueError("finite explicit geometry required")
        if (
            c["ball_world_initial"][2] != 0.11
            or c["goal_world"][2] != 0.11
            or np.linalg.norm(c["velocity_world_initial"]) > 0.6
        ):
            raise ValueError("unchanged ball and bounded velocity required")
    return r, d


def load_ancestor_actor(policy: Path, declaration: dict[str, Any]) -> Any:
    from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student

    root = Path(declaration["student_root"])
    path = root / "complete.json"
    receipt = json.loads(path.read_text())
    if (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != declaration["student_hash"]
        or receipt["checkpoint_hash"] != declaration["checkpoint_hash"]
    ):
        raise ValueError("frozen grown parent changed")
    actor, _ = load_paid_student(
        policy,
        checkpoint=Path(receipt["checkpoint_path"]),
        checkpoint_hash=receipt["checkpoint_hash"],
        training_receipt=path,
        training_receipt_hash=declaration["student_hash"],
        scene_hash=declaration["body_scene_hash"],
    )
    return actor


def ancestor_comparison(groups: list[dict[str, Any]]) -> dict[str, Any]:
    def good(w: dict[str, Any]) -> bool:
        return w["stable"] and len(w["quality_by_kick"]) == 3 and w["quality_by_kick"][2]

    rescues = []
    harms = []
    earlier = []
    for group in groups:
        a, b, c = (good(group["arms"][k]) for k in ("grown", "ancestor_third", "r2_all"))
        i = group["case"]
        if b and not a:
            rescues.append(i)
        if a and not b:
            harms.append(i)
        if c and not a and not b:
            earlier.append(i)
    return dict(
        summaries={arm: ancestor_outcome([g["arms"][arm] for g in groups]) for arm in SCHEDULES},
        same_entry_ancestor_rescues=rescues,
        same_entry_ancestor_harms=harms,
        full_R2_only_rescues=earlier,
        reproducible_teacher_exists=bool(rescues),
        conclusion="SAME_ENTRY_TEACHER_EXISTS"
        if rescues
        else "NO_SAME_ENTRY_RESCUE_NO_ROUTER_TRAINING_YET",
        training_authorized_by_this_diagnostic=False,
        promotion_authorized=False,
    )
