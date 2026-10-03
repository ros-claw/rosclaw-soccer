"""Bounded scheduling for four already declared SIM_ONLY GPU shards.

No driver, model, acceptance rule, retry, evidence selection, or authority is
provided here. The caller's existing simulation workflow owns those checks.
Spawn avoids inheriting native simulator/GPU state and lets Python-heavy full
model validation run in parallel. Only compact results should leave a shard.
"""

import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any


def execute_shards(
    worker: Callable[[dict[str, Any]], Any], jobs: list[dict[str, Any]], *, spawn: bool
) -> list[Any]:
    if (
        type(spawn) is not bool
        or len(jobs) != 4
        or any(type(job.get("gpu")) is not int for job in jobs)
        or [job["gpu"] for job in jobs] != list(range(4))
    ):
        raise ValueError("exactly four ordered distinct GPU shards and explicit mode required")
    if spawn:
        with ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            return list(pool.map(worker, jobs, chunksize=1))
    with ThreadPoolExecutor(max_workers=4) as threads:
        return list(threads.map(worker, jobs))
