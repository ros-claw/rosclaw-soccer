"""Typed contact candidates; never relax a frozen unseen pair's model lock."""

import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.paid_canonical_behavior import digest


def contact_exam_candidate(
    root: Path, expected: str, plan: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any], dict[str, str]]:
    path = root / "complete.json"
    if path.is_symlink() or digest(path) != expected or (root / "failed.json").exists():
        raise ValueError("exact closed simulation candidate required")
    r = json.loads(path.read_text())
    from rosclaw_soccer.rsi.paid_replay_neural_student import AUDITED_STATUS

    if r.get("status") == AUDITED_STATUS:
        return neural_train_candidate(root, expected, r, plan)
    from rosclaw_soccer.rsi.paid_cardinal_memory import GROWING_CONTROL_STATUS, GROWING_STATUS
    from rosclaw_soccer.rsi.paid_growth_memory_data import growth_exam_candidate

    if r.get("status") not in (GROWING_STATUS, GROWING_CONTROL_STATUS):
        raise ValueError("unknown candidate type is not simulation admission")
    return growth_exam_candidate(root, expected, plan)


def neural_train_candidate(
    root: Path, expected: str, r: dict[str, Any], plan: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any], dict[str, str]]:
    from rosclaw_soccer.rsi.paid_neighborhood_study import neighborhood_config
    from rosclaw_soccer.rsi.paid_replay_neural import FIT_STATUS, replay_config
    from rosclaw_soccer.rsi.paid_replay_neural_student import AUDITED_STATUS

    # The old v4 probe is intentionally NOT usable for design changes made
    # after looking at its outcome. Its frozen pair remains immutable.
    if plan.get("plan_kind") is not None or plan.get("config") != neighborhood_config():
        raise ValueError("neural candidate only admitted to original consumed TRAIN32 fixture")
    cfg = replay_config(
        r.get("config", {}).get("seed_index"),
        tail_replay=r.get("config", {}).get("tail_replay", False),
    )
    if (
        r.get("status") != AUDITED_STATUS
        or r.get("config") != cfg
        or r.get("CPU_fit_reconstruction") is not True
        or r.get("own_native_steps") != 0
        or r.get("optimizer_updates") != 0
        or r.get("runtime_admitted") is not False
        or r.get("body_scene_hash") != plan["body_scene_hash"]
        or r.get("parent_hash") != plan["specifications"]["models"]["learned_memory"]["hash"]
        or any(r.get(k) is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError("exact independently checked same-body current-parent neural fit required")
    fp = Path(r["fit_root"]) / "complete.json"
    cp = Path(r["checkpoint_path"])
    if (
        fp.is_symlink()
        or digest(fp) != r["fit_hash"]
        or cp.is_symlink()
        or cp.parent != fp.parent
        or cp.stat().st_size > 16000000
        or digest(cp) != r["checkpoint_hash"]
        or (fp.parent / "failed.json").exists()
    ):
        raise ValueError("exact confined fit and neural checkpoint required")
    fit = json.loads(fp.read_text())
    if fit.get("status") != FIT_STATUS or any(
        fit.get(k) != r.get(k)
        for k in (
            "config",
            "certificate",
            "counts",
            "body_scene_hash",
            "parent_hash",
            "checkpoint_path",
            "checkpoint_hash",
        )
    ):
        raise ValueError("neural audit differs from its complete fit")
    pins = dict(r["input_file_hashes"])
    pins.update(
        {
            str(root / "complete.json"): expected,
            str(fp): r["fit_hash"],
            str(cp): r["checkpoint_hash"],
        }
    )
    config = neighborhood_config()
    config.update(
        protocol="paid.contact_neighborhood.soft_replay_neural_train_candidate.v5",
        candidate_kind="soft_replay_neural",
        candidate_hash=expected,
        candidate_seed_index=cfg["seed_index"],
        frozen_parent_hash=r["parent_hash"],
        consumed_for_training=True,
        heldout_generalization_claim=False,
        maximum_native_steps_per_arm=248960,
        maximum_episodes=192,
        maximum_native_steps=746880,
        arms=["seed0", "seed1", "seed2"],
        best_seed_selection_authorized=False,
    )
    if cfg.get("tail_replay", False):
        config.update(
            candidate_tail_replay=True, candidate_neural_training_protocol=cfg["protocol"]
        )
    return dict(root=str(root), hash=expected), config, pins
