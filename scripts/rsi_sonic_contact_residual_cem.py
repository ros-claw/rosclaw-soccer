"""Bounded episodic CEM for SIM_ONLY SONIC contact residual or timing.

The frozen SONIC body and left-leg parent are unchanged.  Only four numbers
mapping ball lateral position to right hip/knee offsets or residual timing are plastic.
MuJoCo outcomes train the candidate; reserved courses are opened once after
selection and never feed the optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rsi_sonic_ball_contact_probe import run

from rosclaw_soccer.rsi.sonic_contact_selector import choose_lateral, load_selector
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

RESIDUAL_TRAIN = ((2.12, 0.065), (2.12, 0.105), (2.12, 0.145))
RESIDUAL_RESERVED = ((2.19, 0.075), (2.19, 0.115), (2.19, 0.155))
TIMING_TRAIN = ((2.14, 0.070), (2.14, 0.110), (2.14, 0.150))
TIMING_RESERVED = ((2.197, 0.085), (2.197, 0.125), (2.197, 0.165))
SPATIOTEMPORAL_TRAIN = tuple((x, y) for x in (2.12, 2.15, 2.18) for y in (0.07, 0.11, 0.15))
SPATIOTEMPORAL_RESERVED = ((2.195, 0.085), (2.195, 0.125), (2.195, 0.165))
RESIDUAL_PARENT = np.array((-0.25, 0.0, -0.25, 0.0), dtype=np.float64)
TIMING_PARENT = np.array((0.48, 0.0, 0.18, 0.0), dtype=np.float64)
SCHEMA = "rosclaw_soccer.rsi.sonic_contact_cem.v3"


@dataclass(frozen=True)
class Candidate:
    params: tuple[float, float, float, float]
    name: str


def _residual(params: tuple[float, float, float, float], ball_y_m: float) -> tuple[float, float]:
    offset = ball_y_m - 0.105
    return (
        float(np.clip(params[0] + params[1] * offset, -0.25, 0.25)),
        float(np.clip(params[2] + params[3] * offset, -0.25, 0.25)),
    )


def _timing(params: tuple[float, float, float, float], ball_y_m: float) -> tuple[float, float]:
    offset = ball_y_m - 0.110
    return (
        float(np.clip(params[0] + params[1] * offset, 0.35, 0.65)),
        float(np.clip(params[2] + params[3] * offset, 0.08, 0.25)),
    )


def _spatiotemporal_timing(
    params: tuple[float, float, float, float], ball_x_m: float, ball_y_m: float
) -> tuple[float, float]:
    return (
        float(np.clip(params[0] + params[1] * (ball_x_m - 2.15), 0.35, 0.65)),
        float(np.clip(params[2] + params[3] * (ball_y_m - 0.11), 0.08, 0.25)),
    )


def _params(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("finite four-parameter motor residual required")
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _score(report: dict[str, Any]) -> float:
    numbers = (
        float(report["minimum_pelvis_height_m"]),
        float(report["peak_pelvis_tilt_rad"]),
        float(report["actuator_saturation_fraction"]),
        float(report["peak_ball_speed_mps"]),
    )
    if not all(math.isfinite(value) for value in numbers):
        return -10.0
    safe = bool(
        report["minimum_pelvis_height_m"] >= 0.62
        and report["peak_pelvis_tilt_rad"] <= 0.32
        and report["actuator_saturation_fraction"] <= 0.01
    )
    if not safe:
        return -10.0
    foot = bool(report["first_robot_ball_contact_is_foot"])
    goal = bool(report["whole_ball_goal_crossed"])
    speed = min(float(report["peak_ball_speed_mps"]), 6.0)
    if foot and goal:
        return 10.0 + speed
    if foot:
        return 1.0 + 0.1 * speed
    return -2.0


def _course_run(
    *,
    candidate: Candidate,
    course: tuple[float, float],
    repeat: int,
    partition: str,
    output_root: Path,
    model_root: Path,
    stadium_assets: Path,
    selector: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    x, y = course
    if mode in ("timing", "spatiotemporal"):
        center, sigma = (
            _spatiotemporal_timing(candidate.params, x, y)
            if mode == "spatiotemporal"
            else _timing(candidate.params, y)
        )
        hip, knee = -0.25, -0.25
    else:
        hip, knee = _residual(candidate.params, y)
        center, sigma = 0.48, 0.18
    lateral = choose_lateral(selector, ball_x_m=x, ball_y_m=y)
    report = run(
        model_root=model_root,
        stadium_assets=stadium_assets,
        output_dir=output_root
        / f"{partition.lower()}-{candidate.name}-x{round(x * 1000)}-y{round(y * 1000)}-r{repeat}",
        ball_x_m=x,
        ball_y_m=y,
        frames=300,
        run_speed_mps=1.4,
        run_lateral_mps=lateral,
        stop_frame=120,
        left_hip_residual_rad=-0.15,
        left_knee_residual_rad=0.15,
        right_hip_residual_rad=hip,
        right_knee_residual_rad=knee,
        contact_envelope_center_m=center,
        contact_envelope_sigma_m=sigma,
        partition="FRESH" if partition == "RESERVED" else "DISCOVERY",
        selector_hash=selector["model_hash"],
    )
    return {
        "candidate": candidate.name,
        "course": [x, y],
        "repeat": repeat,
        "partition": partition,
        "score": _score(report),
        "safe": report["minimum_pelvis_height_m"] >= 0.62
        and report["peak_pelvis_tilt_rad"] <= 0.32
        and report["actuator_saturation_fraction"] <= 0.01,
        "foot_first": report["first_robot_ball_contact_is_foot"],
        "goal": report["whole_ball_goal_crossed"],
        "peak_speed_mps": report["peak_ball_speed_mps"],
        "contact_center_m": center,
        "contact_sigma_m": sigma,
        "right_hip_rad": hip,
        "right_knee_rad": knee,
        "report_hash": report["report_hash"],
        "trajectory_hash": report["trajectory_hash"],
    }


def _batch(
    candidates: tuple[Candidate, ...],
    courses: tuple[tuple[float, float], ...],
    *,
    repeats: int,
    partition: str,
    output_root: Path,
    model_root: Path,
    stadium_assets: Path,
    selector: dict[str, Any],
    workers: int,
    mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _course_run,
                candidate=candidate,
                course=course,
                repeat=repeat,
                partition=partition,
                output_root=output_root,
                model_root=model_root,
                stadium_assets=stadium_assets,
                selector=selector,
                mode=mode,
            )
            for candidate in candidates
            for course in courses
            for repeat in range(repeats)
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return sorted(rows, key=lambda row: (row["candidate"], row["course"], row["repeat"]))


def _fitness(rows: list[dict[str, Any]], name: str, course_count: int) -> float:
    values = [float(row["score"]) for row in rows if row["candidate"] == name]
    if len(values) != course_count:
        raise ValueError("candidate lacks a complete train course set")
    return float(np.mean(values) + 0.5 * min(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--mode", choices=("residual", "timing", "spatiotemporal"), default="residual"
    )
    args = parser.parse_args()
    if (
        args.output_root.exists()
        or not args.model_root.is_dir()
        or not args.stadium_assets.is_dir()
        or not 1 <= args.generations <= 4
        or not 2 <= args.population <= 8
        or not 1 <= args.workers <= 4
    ):
        raise ValueError("bounded run with new external output and qualified inputs required")
    selector = load_selector(args.selector)
    train_courses = (
        SPATIOTEMPORAL_TRAIN
        if args.mode == "spatiotemporal"
        else TIMING_TRAIN
        if args.mode == "timing"
        else RESIDUAL_TRAIN
    )
    reserved_courses = (
        SPATIOTEMPORAL_RESERVED
        if args.mode == "spatiotemporal"
        else TIMING_RESERVED
        if args.mode == "timing"
        else RESIDUAL_RESERVED
    )
    parent_params = TIMING_PARENT if args.mode != "residual" else RESIDUAL_PARENT
    source = Path(__file__)
    probe = source.with_name("rsi_sonic_ball_contact_probe.py")
    source_hash = hash_bytes(source.read_bytes())
    probe_hash = hash_bytes(probe.read_bytes())
    selector_hash = hash_bytes(args.selector.read_bytes())
    model_hashes = {
        name: hash_bytes((args.model_root / "low_latency" / name).read_bytes())
        for name in ("model_encoder.onnx", "model_decoder.onnx", "config.yaml")
    }
    protocol = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "source_hash": source_hash,
        "probe_hash": probe_hash,
        "selector_file_hash": selector_hash,
        "selector_model_hash": selector["model_hash"],
        "model_hashes": model_hashes,
        "train_courses": train_courses,
        "reserved_courses": reserved_courses,
        "parent": parent_params.tolist(),
        "mode": args.mode,
        "plastic_boundary": (
            "contact_envelope_center_conditional_on_ball_x_and_width_on_ball_y"
            if args.mode == "spatiotemporal"
            else "contact_envelope_center_and_width_conditional_on_ball_y"
            if args.mode == "timing"
            else "right_hip_and_knee_contact_residual_conditional_on_ball_y"
        ),
        "frozen": ["SONIC", "left_contact_residual", "approach_selector", "MuJoCo_physics"],
        "seed": 92801,
        "generations": args.generations,
        "population": args.population,
        "workers": args.workers,
        "score": "foot_first_goal_and_ball_speed_with_body_safety",
        "acceptance": (
            "all_reserved_safe_and_foot_first_goal; strict_replay; mean_speed_gain>=0.20mps"
        ),
    }
    args.output_root.mkdir(parents=True)
    protocol["protocol_hash"] = hash_json(protocol)
    (args.output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    parent = Candidate(_params(parent_params), "parent")
    all_train = _batch(
        (parent,),
        train_courses,
        repeats=1,
        partition="TRAIN",
        output_root=args.output_root,
        model_root=args.model_root,
        stadium_assets=args.stadium_assets,
        selector=selector,
        workers=args.workers,
        mode=args.mode,
    )
    rng = np.random.default_rng(92801)
    mean = parent_params.copy()
    std = (
        np.array((0.035, 0.50, 0.025, 0.20))
        if args.mode == "spatiotemporal"
        else np.array((0.035, 0.25, 0.025, 0.20))
        if args.mode == "timing"
        else np.array((0.05, 0.30, 0.05, 0.30))
    )
    history: list[dict[str, Any]] = []
    best = parent
    parent_train_fitness = _fitness(all_train, parent.name, len(train_courses))
    best_fitness = float("-inf")
    for generation in range(args.generations):
        population = tuple(
            Candidate(
                _params(
                    np.clip(
                        mean + rng.normal(size=4) * std,
                        (0.35, -2.0, 0.08, -1.0)
                        if args.mode == "spatiotemporal"
                        else (0.35, -1.0, 0.08, -1.0)
                        if args.mode == "timing"
                        else (-0.25, -1.0, -0.25, -1.0),
                        (0.65, 2.0, 0.25, 1.0)
                        if args.mode == "spatiotemporal"
                        else (0.65, 1.0, 0.25, 1.0)
                        if args.mode == "timing"
                        else (0.25, 1.0, 0.25, 1.0),
                    )
                ),
                f"g{generation}-c{index}",
            )
            for index in range(args.population)
        )
        rows = _batch(
            population,
            train_courses,
            repeats=1,
            partition="TRAIN",
            output_root=args.output_root,
            model_root=args.model_root,
            stadium_assets=args.stadium_assets,
            selector=selector,
            workers=args.workers,
            mode=args.mode,
        )
        all_train.extend(rows)
        ranking = sorted(
            (
                (_fitness(rows, candidate.name, len(train_courses)), candidate)
                for candidate in population
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        elites = [candidate for _, candidate in ranking[: max(2, args.population // 3)]]
        mean = np.mean(np.asarray([elite.params for elite in elites]), axis=0)
        floor = (
            np.array((0.005, 0.06, 0.004, 0.03))
            if args.mode == "spatiotemporal"
            else np.array((0.005, 0.03, 0.004, 0.03))
            if args.mode == "timing"
            else np.array((0.008, 0.05, 0.008, 0.05))
        )
        std = np.maximum(0.65 * std, floor)
        if ranking[0][0] > best_fitness:
            best_fitness, best = ranking[0]
        history.append(
            {
                "generation": generation,
                "ranked": [
                    {"fitness": fitness, **asdict(candidate)} for fitness, candidate in ranking
                ],
                "mean": mean.tolist(),
                "std": std.tolist(),
                "best_name": best.name,
            }
        )
        (args.output_root / "progress.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    # The holdout is opened once, only after all training decisions are frozen.
    selected = best
    selected_path = args.output_root / "selected.json"
    selected_record = {
        "candidate": asdict(selected),
        "train_fitness": best_fitness,
        "protocol_hash": protocol["protocol_hash"],
    }
    selected_record["selected_hash"] = hash_json(selected_record)
    selected_path.write_text(
        json.dumps(selected_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reserved = _batch(
        (parent, selected),
        reserved_courses,
        repeats=2,
        partition="RESERVED",
        output_root=args.output_root,
        model_root=args.model_root,
        stadium_assets=args.stadium_assets,
        selector=selector,
        workers=args.workers,
        mode=args.mode,
    )

    def rows_for(name: str) -> list[dict[str, Any]]:
        return [row for row in reserved if row["candidate"] == name]

    parent_rows = rows_for(parent.name)
    selected_rows = rows_for(selected.name)

    def strict(rows: list[dict[str, Any]]) -> bool:
        return all(
            next(row for row in rows if row["course"] == [x, y] and row["repeat"] == 0)[
                "trajectory_hash"
            ]
            == next(row for row in rows if row["course"] == [x, y] and row["repeat"] == 1)[
                "trajectory_hash"
            ]
            for x, y in reserved_courses
        )

    selected_passed = all(
        row["safe"] and row["foot_first"] and row["goal"] for row in selected_rows
    )
    parent_speed = float(np.mean([row["peak_speed_mps"] for row in parent_rows]))
    selected_speed = float(np.mean([row["peak_speed_mps"] for row in selected_rows]))
    source_stable = bool(
        hash_bytes(source.read_bytes()) == source_hash
        and hash_bytes(probe.read_bytes()) == probe_hash
        and hash_bytes(args.selector.read_bytes()) == selector_hash
        and all(
            hash_bytes((args.model_root / "low_latency" / name).read_bytes()) == digest
            for name, digest in model_hashes.items()
        )
    )
    passed = bool(
        selected.name != parent.name
        and selected_passed
        and strict(parent_rows)
        and strict(selected_rows)
        and selected_speed - parent_speed >= 0.20
        and source_stable
    )
    summary = {
        "schema": SCHEMA,
        "protocol_hash": protocol["protocol_hash"],
        "selected_hash": selected_record["selected_hash"],
        "selected": asdict(selected),
        "mode": args.mode,
        "train_fitness": best_fitness,
        "parent_train_fitness": parent_train_fitness,
        "train_episodes": len(all_train),
        "reserved_episodes": len(reserved),
        "reserved_parent_foot_first_goals": sum(
            row["safe"] and row["foot_first"] and row["goal"] for row in parent_rows[::2]
        ),
        "reserved_selected_foot_first_goals": sum(
            row["safe"] and row["foot_first"] and row["goal"] for row in selected_rows[::2]
        ),
        "parent_speed_mps": parent_speed,
        "selected_speed_mps": selected_speed,
        "parent_strict_replay": strict(parent_rows),
        "selected_strict_replay": strict(selected_rows),
        "source_stable_during_run": source_stable,
        "passed": passed,
        "promotion_status": "FROZEN_LOCAL_CONTACT_CANDIDATE" if passed else "REJECTED_DEVELOPMENT",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "no_team_claim": True,
    }
    summary["report_hash"] = hash_json(summary)
    (args.output_root / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
