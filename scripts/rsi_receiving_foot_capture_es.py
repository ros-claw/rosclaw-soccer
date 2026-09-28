"""SIM_ONLY bounded foot-geometry search on an effect-validated live proxy."""

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

LOW = np.asarray((0.10, 0.0, -0.30, -0.10), dtype=np.float64)
HIGH = np.asarray((0.80, 0.80, -0.08, 0.10), dtype=np.float64)


def _sealed(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    commitment = value.pop("report_hash")
    if commitment != hash_json(value) or value.get("schema") != schema:
        raise ValueError(f"sealed {schema} report required")
    return value, str(commitment)


def _rollout(
    paths: dict[str, Path], output_dir: Path, parameters: NDArray[np.float64]
) -> dict[str, Any]:
    pre, post, offset_x, offset_y = (float(value) for value in parameters)
    report = evaluate(
        asset_root=paths["asset_root"],
        sonic_model_root=paths["sonic_model_root"],
        captured=paths["captured"],
        warm_start=paths["warm_start"],
        training=paths["training"],
        fresh=paths["fresh"],
        output_dir=output_dir,
        precontact_foot_gain=pre,
        foot_capture_gain=post,
        foot_offset_x_m=offset_x,
        foot_offset_y_m=offset_y,
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
    objective = -(speed + distance) if safe else -1_000_000.0
    return {
        "parameters": parameters.tolist(),
        "teacher_hash": report["foot_teacher_hash"],
        "rollout_report_hash": report["report_hash"],
        "safe": safe,
        "proxy_task_gate_passed": safe and speed <= 0.35 and distance <= 0.35,
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
    baseline: Path,
    output_dir: Path,
    generations: int = 4,
    population: int = 16,
    workers: int = 4,
) -> dict[str, Any]:
    source = Path(__file__)
    root = source.parents[1]
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY teacher search directory required")
    if not (1 <= generations <= 12 and 4 <= population <= 32 and 1 <= workers <= 4):
        raise ValueError("bounded teacher search budget required")
    gate, gate_hash = _sealed(
        proxy_gate / "report.json", "rosclaw_soccer.rsi.receiving_single_live_proxy_gate.v1"
    )
    parent, parent_hash = _sealed(
        baseline / "report.json", "rosclaw_soccer.rsi.receiving_single_live_sonic_proxy.v1"
    )
    proxy_source = source.with_name("rsi_receiving_single_live_sonic_proxy.py")
    teacher_source = root / "src/rosclaw_soccer/providers/g1/receiving_foot_capture.py"
    source_hashes = {
        "search": hash_bytes(source.read_bytes()),
        "proxy": hash_bytes(proxy_source.read_bytes()),
        "teacher": hash_bytes(teacher_source.read_bytes()),
    }
    if (
        gate["local_training_proxy_authorized"] is not True
        or gate["zero_report_hash"] != parent_hash
        or parent["source_hashes"]["source"] != source_hashes["proxy"]
        or parent["source_hashes"]["foot_teacher"] != source_hashes["teacher"]
    ):
        raise ValueError("current source must match sealed live-proxy effect gate")
    paths = dict(
        asset_root=asset_root,
        sonic_model_root=sonic_model_root,
        captured=captured,
        warm_start=warm_start,
        training=training,
        fresh=fresh,
    )
    rng = np.random.default_rng(20260929)
    mean = np.asarray((0.3, 0.3, -0.18, 0.03), dtype=np.float64)
    std = np.asarray((0.14, 0.18, 0.055, 0.05), dtype=np.float64)
    incumbent = mean.copy()
    records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True)
    for generation in range(generations):
        samples = np.clip(rng.normal(mean, std, size=(population, 4)), LOW, HIGH)
        samples[0] = incumbent
        if generation == 0:
            samples[1] = np.asarray((0.3, 0.0, -0.18, 0.03), dtype=np.float64)
            samples[2] = np.asarray((0.3, 0.3, -0.18, 0.03), dtype=np.float64)
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
        std = np.maximum(
            np.asarray((0.025, 0.025, 0.008, 0.008)),
            np.std(elite_parameters, axis=0) * 1.35,
        )
        winner = max(records, key=lambda row: row["objective"])
        incumbent = np.asarray(winner["parameters"], dtype=np.float64)
        print(
            json.dumps(
                {
                    "generation": generation,
                    "best_tail_ball_speed_mps": winner["tail_maximum_ball_speed_mps"],
                    "best_tail_foot_distance_m": winner["tail_maximum_foot_distance_m"],
                    "any_proxy_task_gate": any(row["proxy_task_gate_passed"] for row in records),
                }
            ),
            flush=True,
        )
        if winner["proxy_task_gate_passed"]:
            break
    if {
        key: hash_bytes(path.read_bytes())
        for key, path in (("search", source), ("proxy", proxy_source), ("teacher", teacher_source))
    } != source_hashes:
        raise RuntimeError("source changed during bounded teacher search")
    winner = max(records, key=lambda row: row["objective"])
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_foot_capture_es.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": source_hashes,
        "proxy_gate_report_hash": gate_hash,
        "zero_report_hash": parent_hash,
        "seed": 20260929,
        "generation_budget": generations,
        "population": population,
        "actual_rollouts": len(records),
        "winner": winner,
        "candidates": records,
        "proxy_task_gate_passed": winner["proxy_task_gate_passed"],
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
        "baseline",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    report = train(**vars(parser.parse_args()))
    print(
        json.dumps(
            {"report_hash": report["report_hash"], "winner": report["winner"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
