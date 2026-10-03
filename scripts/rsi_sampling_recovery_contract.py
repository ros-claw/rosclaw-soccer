"""Read-only preflight for explicitly recovering an interrupted sampling shard.

No simulator launch, log movement, resealing, policy update or retry occurs in
this module. A caller must preserve the original failed run and separately pin
the actual old worker source before executing a declared recovery.
"""

from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def check_recovery_job(job: dict[str, Any], commitment: dict[str, Any]) -> list[int]:
    args = job["args"]
    gpu = job.get("gpu")
    courses = job.get("courses", [])
    if (
        type(gpu) is not int
        or not 0 <= gpu < 4
        or job.get("arm") != "candidate"
        or args.get("resume") is not True
        or args.get("samples_per_course") != 16
        or type(args.get("samples_per_course")) is not int
        or args.get("shared_sampling_models") is not True
        or args.get("compressed_sampling_models") is not True
        or commitment.get("schema") != "soccer.rsi.failure_step_exploration_commitment.v1"
        or commitment.get("partition") != "TRAIN_CONSUMED"
        or commitment.get("exploration_stream") != 2
        or commitment.get("sampling_rho") != 0.9
        or commitment.get("std_raw") != 0.1
        or commitment.get("execution_timeout_s") != 600
        or commitment.get("samples_per_course") != 16
        or len(courses) != 13
        or any(type(c[k]) is not int for c in courses for k in ("seed", "lane"))
        or any(c["lane"] not in (0, 2, 4, 6) for c in courses)
        or [[c["seed"], c["lane"]] for c in courses] != commitment.get("courses")
        or len({(c["seed"], c["lane"]) for c in courses}) != 13
        or job.get("view_hashes") != commitment.get("sampling_view_hashes")
        or len(job.get("view_hashes", [])) != 208
        or len(set(job.get("view_hashes", []))) != 208
        or any(
            commitment.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("same complete declared 13x16 stream and explicit resume required")
    return list(range(gpu, 13, 4))


def inspect_failed_attempt(root: Path, *, seed: int, lane: int, sample: int) -> dict[str, Any]:
    if (
        type(seed) is not int
        or type(lane) is not int
        or lane not in (0, 2, 4, 6)
        or type(sample) is not int
        or not 0 <= sample < 16
        or root.is_symlink()
        or (root / "logs").is_symlink()
    ):
        raise ValueError("exact local declared failed native attempt required")
    stem = f"seed{seed}-lane{lane}-sample-{sample}-actor"
    log, folder = root / "logs" / f"{stem}.log", root / stem
    if (
        log.is_symlink()
        or not log.is_file()
        or folder.is_symlink()
        or any((folder / name).exists() for name in ("report.json", "report.json.gz"))
    ):
        raise ValueError("missing physical report and preserved regular failed log required")
    return dict(
        stem=stem,
        log_path=str(log),
        log_sha256=hash_bytes(log.read_bytes()),
        log_bytes=log.stat().st_size,
        report_present=False,
        native_failure_is_not_physical_training_sample=True,
    )


def describe_recovery(
    job: dict[str, Any], commitment: dict[str, Any], failed: dict[str, Any]
) -> dict[str, Any]:
    indices = check_recovery_job(job, commitment)
    allowed_stems = {
        f"seed{job['courses'][i]['seed']}-lane{job['courses'][i]['lane']}-sample-{s}-actor"
        for i in indices
        for s in range(16)
    }
    if (
        failed.get("report_present") is not False
        or failed.get("native_failure_is_not_physical_training_sample") is not True
        or failed.get("stem") not in allowed_stems
        or Path(failed["log_path"])
        != Path(job["args"]["output_root"]) / "logs" / f"{failed['stem']}.log"
    ):
        raise ValueError("original native failure must remain distinct from training outcomes")
    result = dict(
        schema="soccer.rsi.explicit_sampling_shard_recovery.v1",
        original_commitment_hash=hash_json(commitment),
        gpu=job["gpu"],
        course_indices=indices,
        original_failed_attempt=failed,
        original_failed_run_must_be_preserved=True,
        completed_reports_require_full_reaudit=True,
        reused_reports_are_not_new_physics=True,
        automatic_retry=False,
        fresh_exam_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    return result
