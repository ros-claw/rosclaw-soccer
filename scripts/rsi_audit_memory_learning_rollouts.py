"""Bounded parallel reconstruction; every worker performs the full physics audit.

Only compact, measured arrays leave workers. Ordering is declared course/sample
order, never completion order. Worker exceptions abort the bank, not skip rows.
This helper grants no execution or promotion authority.
"""

import json
import multiprocessing
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from scripts.rsi_collect_failed_step_courses import sampling_seed
from scripts.rsi_fit_protected_phase_step_motor import gpu_observations


def ordered_audits(
    worker: Callable[[dict[str, Any]], Any], jobs: list[dict[str, Any]], workers: int
) -> Iterator[Any]:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("one to four audit workers required")
    if not jobs:
        raise ValueError("a nonempty declared audit bank is required")
    if workers == 1:
        yield from map(worker, jobs)
        return
    # Spawn avoids inheriting Torch/GPU contexts or open simulator state.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        yield from pool.map(worker, jobs, chunksize=1)


def audit_course(job: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kind = job["kind"]
    if kind == "smooth-memory":
        from rosclaw_soccer.rsi.smooth_memory_motor import SAMPLING_SCHEMA, validate_model
    elif kind == "output-memory":
        from rosclaw_soccer.rsi.output_memory_step_motor import SAMPLING_SCHEMA, validate_model
    else:
        raise ValueError("declared memory policy required")
    model = json.loads(Path(job["model_path"]).read_text())
    validate_model(model)
    if model["model_hash"] != job["model_hash"]:
        raise ValueError("model changed before worker reconstruction")
    i, row, commitment = job["index"], job["row"], job["commitment"]
    seed, lane = job["course"]
    root = Path(job["exploration_root"])
    if (
        row["index"] != i
        or [row["seed"], row["lane"]] != [seed, lane]
        or len(row["samples"]) != commitment["samples_per_course"]
    ):
        raise ValueError("declared course/sample identity changed")
    parent_folder = root / f"seed{seed}-lane{lane}-reproduction-parent"
    parent = _sealed(parent_folder / "report.json")
    audit_lateral_approach(parent_folder)
    if (
        parent["report_hash"] != row["parent_report_hash"]
        or parent["training_course_seed"] != seed
        or parent["single_course_lane"] != lane
        or parent["source_hash"] != commitment["runner_hash"]
        or parent["asset_hash"] != commitment["asset_hash"]
    ):
        raise ValueError("physical parent changed")
    records: list[dict[str, Any]] = []
    chunks: dict[str, list[Any]] = {}
    for s, sample in enumerate(row["samples"]):
        folder = root / f"seed{seed}-lane{lane}-sample-{s}-actor"
        decoded: list[Any] = []
        raw = _sealed(folder / "report.json")
        checked = _outcome(
            folder, raw["contact_motor_policy_hash"], commitment, decoder_sink=decoded
        )
        if checked["report"] != raw:
            raise ValueError("physical report changed during reconstruction")
        x, phase, raw = gpu_observations(folder, raw)
        view = raw["contact_motor_policy"]["step_motor_proof"]["model"]
        if (
            len(decoded) != 1
            or view["schema"] != SAMPLING_SCHEMA
            or view["mean_model"] != model
            or view["seed"] != sampling_seed(i, s, generation=model.get("generation", 0))
            or view["model_hash"] != sample["view_hash"]
            or view["model_hash"]
            != commitment["sampling_view_hashes"][i * commitment["samples_per_course"] + s]
            or sample["sample"] != s
            or raw["report_hash"] != sample["report_hash"]
            or raw["training_course_seed"] != seed
            or raw["single_course_lane"] != lane
            or raw["parent_report_hash"] != parent["report_hash"]
            or raw["environments"][0]["course"] != parent["environments"][0]["course"]
            or raw["source_hash"] != commitment["runner_hash"]
            or raw["asset_hash"] != commitment["asset_hash"]
            or (kind == "smooth-memory" and (view["rho"] != 0.9 or view["std_raw"] != 0.1))
        ):
            raise ValueError("sample is not the declared current physical policy")
        outcome = checked["outcome"]
        if any(sample[k] != v for k, v in outcome.items()):
            raise ValueError("sample outcome differs from independently measured physics")
        draws = [
            decoded[0].latent_sample(v, frame, int(p))
            for frame, (v, p) in enumerate(zip(x, phase, strict=True), start=30)
        ]
        group = i * commitment["samples_per_course"] + s
        values = dict(
            observation=x,
            phase_index=phase,
            latent_action=np.stack([v[0] for v in draws]),
            old_log_probability=np.asarray([v[1] for v in draws]),
            terminal_return=np.full(270, terminal_return(outcome)),
            std_raw=np.full(270, view["std_raw"]),
            trajectory_index=np.full(270, group, dtype=np.int64),
        )
        for key, value in values.items():
            chunks.setdefault(key, []).append(value)
        records.append(
            dict(
                group=group,
                seed=seed,
                lane=lane,
                sample=s,
                backend="IsaacLab",
                folder=str(folder),
                report_hash=raw["report_hash"],
                outcome=outcome,
            )
        )
        print(f"AUDITED_MEMORY_ROLLOUT course={i} sample={s} group={group}", flush=True)
    return records, {k: np.concatenate(v) for k, v in chunks.items()}
