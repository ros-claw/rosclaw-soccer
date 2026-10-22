"""A frozen-pair unseen local development probe, NOT the formal Fresh exam."""

import copy
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.paid_canonical_behavior import digest

STATUS = "PAID_GROWTH_UNSEEN32_PROBE_PREREGISTERED_NO_ROLLOUT"
KIND = "frozen_growth_pair_unseen_local_probe32"
IDENTITIES = ("probe_base", "probe_learned", "probe_zero")


def probe_config(hashes: dict[str, str]) -> dict[str, Any]:
    from rosclaw_soccer.rsi.paid_neighborhood_study import neighborhood_config

    if set(hashes) != {"learned", "zero-control"} or any(
        type(h) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", h) is None
        for h in hashes.values()
    ):
        raise ValueError("exact pre-frozen learned/control commitments required")
    cfg = neighborhood_config()
    cfg.update(
        protocol="paid.growth.unseen_local32_probe.v4",
        seed=2026102207,
        position_half_range_m=0.03,
        velocity_half_range_mps=0.06,
        sampling="PythonRandomMT19937_fixed_seed_16_per_source_no_resampling",
        arms=["learned", "zero-control"],
        maximum_episodes=128,
        maximum_native_steps=497920,
        maximum_native_steps_per_arm=248960,
        locked_candidate_hashes=hashes.copy(),
        consumed_for_training=False,
        formal_Fresh_claim=False,
        statistical_independence_not_established=True,
    )
    return cfg


def probe_cases(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(sources) != 2 or [c["case"] for c in sources] != [4, 5]:
        raise ValueError("original paired source contexts required")
    rng = random.Random(2026102207)
    rows: list[dict[str, Any]] = []
    for source in sources:
        if source["profile"] != "cached-standard" or source["ball_mass_kg"] != 0.43:
            raise ValueError("unchanged physical source profile required")
        for k, n in (("goal_world", 3), ("ball_world_initial", 3), ("velocity_world_initial", 2)):
            if len(source[k]) != n or any(
                type(x) not in (int, float) or not math.isfinite(x) for x in source[k]
            ):
                raise ValueError("explicit finite source geometry and velocity required")
        ball, goal, v = [
            source[k] for k in ("ball_world_initial", "goal_world", "velocity_world_initial")
        ]
        length = math.hypot(*goal[:2])
        if length < 4 or goal[2] != 0.11 or ball[2] != 0.11:
            raise ValueError("unchanged goal and standard ball height required")
        f = [q / length for q in goal[:2]]
        s = [-f[1], f[0]]
        for _ in range(16):
            dx, dy = rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)
            dvx, dvy = rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)
            r = copy.deepcopy(source)
            r.update(
                case=len(rows),
                source_case=source["case"],
                bank="unseen_local_probe_not_Fresh",
                not_fresh=True,
                perturbation=[dx, dy, dvx, dvy],
            )
            r["ball_world_initial"] = [ball[j] + dx * f[j] + dy * s[j] for j in range(2)] + [
                ball[2]
            ]
            r["velocity_world_initial"] = [v[j] + dvx * f[j] + dvy * s[j] for j in range(2)]
            if math.hypot(*r["velocity_world_initial"]) > 0.6:
                raise ValueError("reject entire unobserved bank, never clip or resample")
            rows.append(r)
    return rows


def prepare_probe(identity: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    from rosclaw_soccer.rsi.paid_growth_memory_data import growth_exam_candidate
    from rosclaw_soccer.rsi.paid_neighborhood_sources import load_neighborhood_plan

    bp = Path(identity["probe_base_root"])
    if (
        json.loads((bp / "complete.json").read_text()).get("status")
        != "PAID_CONTACT_NEIGHBORHOOD32_PREREGISTERED_NO_ROLLOUT"
    ):
        raise ValueError("probe must derive from original fixture, no nested probe recursion")
    br, base, pins = load_neighborhood_plan(bp, identity["probe_base_hash"])
    if br["status"] != "PAID_CONTACT_NEIGHBORHOOD32_PREREGISTERED_NO_ROLLOUT":
        raise ValueError("probe must derive from original consumed fixture, no nested probes")
    locked = {}
    data_hash = None
    for key, model, control in (
        ("probe_learned", "learned", False),
        ("probe_zero", "zero-control", True),
    ):
        root = Path(identity[key + "_root"])
        h = identity[key + "_hash"]
        spec, cfg, extra = growth_exam_candidate(root, h, base)
        if cfg["candidate_zero_control"] is not control:
            raise ValueError(
                "one fixed learned model and one actual zero-coefficient control required"
            )
        r = json.loads((root / "complete.json").read_text())
        if data_hash is not None and data_hash != r["data_hash"]:
            raise ValueError("same training data required for frozen probe pair")
        data_hash = r["data_hash"]
        locked[model] = spec
        for path, sha in extra.items():
            if path in pins and pins[path] != sha:
                raise ValueError("conflicting probe provenance")
            pins[path] = sha
    # Read actual original sources, never invert rounded perturbation arithmetic.
    edp = Path(base["pipeline_root"]) / "learned-train20-exam/declaration.json"
    if pins.get(str(edp)) != digest(edp):
        raise ValueError("bound original source bank required")
    ed = json.loads(edp.read_text())
    pd = Path(ed["plan_root"]) / "declaration.json"
    if pins.get(str(pd)) != digest(pd):
        raise ValueError("bound original source geometries required")
    original = json.loads(pd.read_text())["cases"]
    cases = probe_cases([original[i] for i in (4, 5)])

    def condition_key(c: dict[str, Any]) -> tuple[Any, ...]:
        return (
            c["source"],
            tuple(c["ball_world_initial"]),
            tuple(c["velocity_world_initial"]),
            tuple(c["goal_world"]),
        )

    unique = {condition_key(c) for c in cases}
    if len(unique) != 32 or unique.intersection(condition_key(c) for c in base["cases"]):
        raise ValueError("unseen probe must have32distinct non-training initial conditions")
    result = dict(
        **identity,
        plan_kind=KIND,
        locked_models=locked,
        candidate_training_data_hash=data_hash,
        config=probe_config({k: v["hash"] for k, v in locked.items()}),
        cases=cases,
        specifications=base["specifications"],
        body_scene_hash=base["body_scene_hash"],
        data_root=base["data_root"],
        data_hash=base["data_hash"],
        pipeline_root=base["pipeline_root"],
        pipeline_hash=base["pipeline_hash"],
    )
    return result, pins


def load_probe_plan(
    root: Path, expected: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != expected
        or (root / "failed.json").exists()
    ):
        raise ValueError("exact closed unseen probe declaration required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    if (
        r["status"] != STATUS
        or r["plan_kind"] != KIND
        or r["declaration_hash"] != digest(dp)
        or r["own_native_steps"] != 0
        or r["own_neural_rows"] != 0
        or r["optimizer_updates"] != 0
        or r["cases_preflighted"] != 32
        or any(r.get(k) != v for k, v in d.items())
        or any(
            v.get(k) is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete no-rollout frozen probe contract required")
    identity = {k: d[k] for n in IDENTITIES for k in (n + "_root", n + "_hash")}
    if Path(identity["probe_base_root"]).resolve() == root.resolve():
        raise ValueError("no recursive probe declaration")
    actual, source_pins = prepare_probe(identity)
    if any(d.get(k) != v for k, v in actual.items()):
        raise ValueError("reconstructed unseen probe differs")
    pins = dict(r["input_file_hashes"])
    if any(pins.get(k) != v for k, v in source_pins.items()):
        raise ValueError("frozen sources changed before unseen evaluation")
    if len(r["entries"]) != 32 or [v["case"] for v in r["entries"]] != list(range(32)):
        raise ValueError("complete compiled entry bank required")
    for entry in r["entries"]:
        path = root / f"case{entry['case']}-initial.npz"
        if path.is_symlink() or entry["raw_path"] != path.name or digest(path) != entry["raw_hash"]:
            raise ValueError("exact compiled unseen entry required")
        pins[str(path)] = entry["raw_hash"]
    pins[str(rp)], pins[str(dp)] = expected, digest(dp)
    return r, d, pins
