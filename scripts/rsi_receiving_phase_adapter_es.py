"""SIM_ONLY bounded evolutionary search for one-G1 live receiving feedback."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rsi_receiving_single_live_sonic_proxy import evaluate

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _sealed(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    commitment = value.pop("report_hash")
    if commitment != hash_json(value) or value.get("schema") != schema:
        raise ValueError(f"unsealed evidence: {path}")
    return value, str(commitment)


def _rollout(
    paths: dict[str, Path], output_dir: Path, parameters: NDArray[np.float64]
) -> dict[str, Any]:
    report = evaluate(
        asset_root=paths["asset_root"],
        sonic_model_root=paths["sonic_model_root"],
        captured=paths["captured"],
        warm_start=paths["warm_start"],
        training=paths["training"],
        fresh=paths["fresh"],
        output_dir=output_dir,
        adapter_parameters=parameters,
    )
    safe = bool(
        report["minimum_pelvis_height_m"] >= 0.65
        and report["maximum_tilt_rad"] < 0.30
        and not report["proxy_nonfoot_frames"]
        and report["proxy_foot_frames"]
        and report["proxy_first_foot_frame"] <= 80
    )
    speed = float(report["tail_maximum_ball_speed_mps"])
    distance = float(report["tail_maximum_foot_distance_m"])
    # Hard safety first, then maximize physically measured retention margin.
    objective = (
        -(speed + 0.8 * distance)
        + 0.01 * min(len(report["proxy_foot_frames"]), 5)
        - 0.005 * float(np.sum(parameters**2))
        if safe
        else -1_000_000.0
    )
    return {
        "parameters": parameters.tolist(),
        "adapter_hash": report["adapter_hash"],
        "rollout_report_hash": report["report_hash"],
        "safe": safe,
        "task_gate_passed": safe and speed <= 0.35 and distance <= 0.35,
        "objective": objective,
        "tail_maximum_ball_speed_mps": speed,
        "tail_maximum_foot_distance_m": distance,
        "foot_frames": report["proxy_foot_frames"],
        "nonfoot_frames": report["proxy_nonfoot_frames"],
        "minimum_pelvis_height_m": report["minimum_pelvis_height_m"],
        "maximum_tilt_rad": report["maximum_tilt_rad"],
        "output_dir": str(output_dir),
    }


def train(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    captured: Path,
    warm_start: Path,
    training: Path,
    fresh: Path,
    proxy_gate: Path,
    zero: Path,
    output_dir: Path,
    generations: int = 5,
    population: int = 12,
    workers: int = 4,
) -> dict[str, Any]:
    source = Path(__file__)
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.parents[1]):
        raise ValueError("new external SIM_ONLY search directory required")
    if not (1 <= generations <= 20 and 4 <= population <= 32 and 1 <= workers <= 4):
        raise ValueError("bounded search budget required")
    gate, gate_hash = _sealed(
        proxy_gate / "report.json", "rosclaw_soccer.rsi.receiving_single_live_proxy_gate.v1"
    )
    baseline, baseline_hash = _sealed(
        zero / "report.json", "rosclaw_soccer.rsi.receiving_single_live_sonic_proxy.v1"
    )
    proxy_source = source.with_name("rsi_receiving_single_live_sonic_proxy.py")
    adapter_source = (
        source.parents[1] / "src/rosclaw_soccer/providers/g1/receiving_phase_adapter.py"
    )
    if (
        gate["local_training_proxy_authorized"] is not True
        or gate["zero_report_hash"] != baseline_hash
        or baseline["source_hashes"]["source"] != hash_bytes(proxy_source.read_bytes())
        or baseline["source_hashes"]["adapter"] != hash_bytes(adapter_source.read_bytes())
        or baseline["capture_report_hash"] != gate["capture_report_hash"]
    ):
        raise ValueError("current source must match sealed effect-fidelity proxy gate")
    source_hashes = {
        "search": hash_bytes(source.read_bytes()),
        "proxy": hash_bytes(proxy_source.read_bytes()),
        "adapter": hash_bytes(adapter_source.read_bytes()),
    }
    paths = dict(
        asset_root=asset_root,
        sonic_model_root=sonic_model_root,
        captured=captured,
        warm_start=warm_start,
        training=training,
        fresh=fresh,
    )
    rng = np.random.default_rng(20260928)
    mean = np.asarray((-0.75, 0, 0, 0, 0, 0, 0, 0), dtype=np.float64)
    std: NDArray[np.float64] = np.full(8, 0.35, dtype=np.float64)
    incumbent: NDArray[np.float64] = np.zeros(8, dtype=np.float64)
    records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True)
    for generation in range(generations):
        samples = np.clip(rng.normal(mean, std, size=(population, 8)), -1.0, 1.0)
        samples[0] = incumbent
        if generation == 0:
            samples[1] = np.zeros(8)
            samples[2] = mean
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _rollout,
                    paths,
                    output_dir / f"generation-{generation:02d}" / f"candidate-{index:02d}",
                    np.asarray(parameters, dtype=np.float64),
                )
                for index, parameters in enumerate(samples)
            ]
            results = [future.result() for future in futures]
        for index, result in enumerate(results):
            result["generation"] = generation
            result["index"] = index
        records.extend(results)
        elite = sorted(results, key=lambda row: row["objective"], reverse=True)[
            : max(2, population // 4)
        ]
        elite_parameters = np.asarray([row["parameters"] for row in elite], dtype=np.float64)
        mean = np.mean(elite_parameters, axis=0)
        std = np.maximum(0.08, np.std(elite_parameters, axis=0) * 1.3)
        winner = max(records, key=lambda row: row["objective"])
        incumbent = np.asarray(winner["parameters"], dtype=np.float64)
        print(
            json.dumps(
                {
                    "generation": generation,
                    "best_objective": winner["objective"],
                    "best_tail_ball_speed_mps": winner["tail_maximum_ball_speed_mps"],
                    "best_tail_foot_distance_m": winner["tail_maximum_foot_distance_m"],
                    "any_proxy_task_gate": any(row["task_gate_passed"] for row in records),
                }
            ),
            flush=True,
        )
        if winner["task_gate_passed"]:
            break
    if {
        key: hash_bytes(path.read_bytes())
        for key, path in (("search", source), ("proxy", proxy_source), ("adapter", adapter_source))
    } != source_hashes:
        raise RuntimeError("source changed during bounded simulation search")
    winner = max(records, key=lambda row: row["objective"])
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_phase_adapter_es.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": source_hashes,
        "proxy_gate_report_hash": gate_hash,
        "zero_report_hash": baseline_hash,
        "seed": 20260928,
        "generation_budget": generations,
        "population": population,
        "actual_rollouts": len(records),
        "winner": winner,
        "candidates": records,
        "proxy_task_gate_passed": winner["task_gate_passed"],
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "asset-root",
        "sonic-model-root",
        "captured",
        "warm-start",
        "training",
        "fresh",
        "proxy-gate",
        "zero",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    report = train(**vars(parser.parse_args()))
    print(
        json.dumps(
            {"report_hash": report["report_hash"], "winner": report["winner"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
