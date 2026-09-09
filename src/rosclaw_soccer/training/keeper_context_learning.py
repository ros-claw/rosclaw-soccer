"""Learn a causal expert selector from paired, physically qualified saves.

Core audits the single-focal parameter update; that is not policy promotion or
any form of robot motion permission. Unsupported contexts remain documented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.keeper_context_gate import KeeperContextGate, gate_config_hash
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture
from rosclaw_soccer.training.near_ball_plasticity import begin_update, finish_update


def train(root: Path, exams: Path, output: Path) -> dict[str, Any]:
    fixture = build_four_vs_four_fixture(root)
    groups: dict[tuple[str, float, float], dict[bool, tuple[dict[str, Any], float]]] = {}
    source_hashes = {}
    parents, configs = set(), set()
    for path in sorted(exams.glob("*/result.json")):
        raw = path.read_bytes()
        r = json.loads(raw)
        if (
            r.get("schema_version") != "s229.shared_keeper_reach_exam.v3"
            or r.get("activation_ceiling") != "SIM_ONLY"
            or r.get("candidate_promoted") is not False
            or r.get("source_integrity_verified") is not True
            or r.get("fixture_hash") != fixture.fixture_hash
            or r.get("gate_policy_hash") is not None
        ):
            raise ValueError("gate training requires bound ungated physical pair evidence")
        trace_path = path.parent / "trajectory.npz"
        if str(hash_bytes(trace_path.read_bytes())) != r["trajectory_hash"]:
            raise ValueError("gate training trajectory changed")
        with np.load(trace_path, allow_pickle=False) as a:
            ready = np.flatnonzero(a["context_ready"])
            if not len(ready):
                raise ValueError("no causal selection context was observed")
            context = float(a["context_height_m"][ready[0]])
        if not np.isfinite(context) or not 0.5 <= context <= 2:
            raise ValueError("invalid gate context")
        key = (r["team"], r["lateral"], r["height"])
        profile = r["config"]["muscle_reach_correction"]
        if type(profile) is not bool or profile in groups.setdefault(key, {}):
            raise ValueError("duplicate or invalid expert profile")
        groups[key][profile] = (r, context)
        parents.add(r["muscle_policy_hash"])
        configs.add(gate_config_hash(r["config"]))
        source_hashes[str(path.relative_to(exams))] = str(hash_bytes(raw))
    if len(parents) != 1 or None in parents or len(configs) != 1:
        raise ValueError("expert pairs changed parent or common controller contract")
    features, labels, unsupported = [], [], []
    for key, pair in groups.items():
        if set(pair) != {False, True}:
            raise ValueError("every context needs both physical expert trials")
        if abs(pair[False][1] - pair[True][1]) > 1e-8:
            raise ValueError("expert trials disagree before causal selection")
        valid = [
            p
            for p in (False, True)
            if pair[p][0]["stable_save"] is True and pair[p][0]["physical_safe"] is True
        ]
        if not valid:
            unsupported.append(key)
            continue
        features.append(pair[False][1])
        labels.append(float(valid[0]))
    if set(labels) != {0.0, 1.0}:
        raise ValueError("both expert choices require positive physical evidence")
    output.mkdir(parents=True, exist_ok=False)
    parent, config_hash = next(iter(parents)), next(iter(configs))
    x, y = np.asarray(features), np.asarray(labels)
    center, scale = float(np.mean(x)), max(0.01, float(np.std(x)))
    design = np.column_stack(((x - center) / scale, np.ones(len(x))))
    weights = np.zeros(2)
    before = {c.agent_id: fixture.foundation_policy_hash for c in fixture.cells}
    before["red.goalkeeper"] = str(hash_json({"parent": parent, "gate": None}))
    lease = begin_update(
        before=before,
        focal="red.goalkeeper",
        generation=230,
        dataset_hash=str(hash_json(source_hashes)),
        context_hash=str(hash_json({"config": config_hash, "fixture": fixture.fixture_hash})),
        maximum_steps=1000,
    )
    for _ in range(1000):
        probability = 1 / (1 + np.exp(-np.clip(design @ weights, -40, 40)))
        weights -= 0.1 * (design.T @ (probability - y) / len(x) + 0.001 * weights)
    numeric = dict(center=center, scale=scale, weight=float(weights[0]), bias=float(weights[1]))
    after = dict(before)
    after["red.goalkeeper"] = str(hash_json({"parent": parent, "gate": numeric}))
    core = finish_update(lease=lease, before=before, after=after, steps=1000)
    payload = dict(
        schema="keeper-context-gate.v1",
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        parent_policy_hash=parent,
        config_hash=config_hash,
        source_hashes=source_hashes,
        core_learning_boundary=core,
        **numeric,
    )
    path = output / "gate.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    gate = KeeperContextGate(path, parent_policy_hash=parent, config_hash=config_hash)
    report = dict(
        activation_ceiling="SIM_ONLY",
        candidate_promoted=False,
        physical_replay_passed=False,
        training_contexts=len(x),
        unsupported_contexts=unsupported,
        source_hashes=source_hashes,
        context_heights=features,
        labels=labels,
        predictions=[gate.select_reach(float(v)) for v in x],
        gate_hash=gate.policy_hash,
        core_learning_boundary=core,
    )
    (output / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--exams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.asset_root, args.exams, args.output), indent=2))


if __name__ == "__main__":
    main()
