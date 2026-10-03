"""Describe every outcome in a sealed, complete consumed learning batch.

This is a read-only diagnostic over already audited data, not a new physical
audit, a causal explanation, a curriculum rewrite, or a promotion decision.
Overlapping failure labels are intentionally not a partition of trajectories.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality


def outcome_labels(outcome: dict[str, Any]) -> list[str]:
    bodies = outcome["contact_body_indices"]
    first = outcome["first_contact_frame"]
    if (
        not isinstance(bodies, list)
        or any(type(v) is not int or not 0 <= v < 6 for v in bodies)
        or len(set(bodies)) != len(bodies)
        or (first is not None and (type(first) is not int or not 0 <= first < 300))
        or bool(bodies) != (first is not None)
        or type(outcome["clean_foot_only"]) is not bool
        or outcome["clean_foot_only"] != bool(bodies and set(bodies) <= {0, 1})
        or type(outcome["high_quality"]) is not bool
    ):
        raise ValueError("consistent actual contact identity and boolean outcomes required")
    for key in ("minimum_pelvis_z_m", "maximum_lateral_excursion_m"):
        if type(outcome[key]) not in (int, float) or not math.isfinite(outcome[key]):
            raise ValueError("finite measured safety outcomes required")
    if outcome["maximum_lateral_excursion_m"] < 0:
        raise ValueError("nonnegative measured ball excursion required")
    forward, lateral = outcome["forward_60_m"], outcome["lateral_60_m"]
    horizon_available = first is not None and first + 60 < 300
    if any(
        (type(v) not in (int, float) or not math.isfinite(v))
        if horizon_available
        else v is not None
        for v in (forward, lateral)
    ):
        raise ValueError("actual fixed-horizon displacement availability required")
    if outcome["high_quality"] != high_quality(outcome):
        raise ValueError("reported quality differs from existing physical acceptance rule")
    labels = []
    if not bodies:
        labels.append("NO_CONTACT")
    elif not outcome["clean_foot_only"]:
        labels.append("NONFOOT_CONTACT")
    if not horizon_available:
        labels.append("NO_COMPLETE_POSTCONTACT_HORIZON")
    else:
        if forward < 1:
            labels.append("INSUFFICIENT_FORWARD_DISPLACEMENT")
        if abs(lateral) / max(forward, 0.01) > 0.3:
            labels.append("EXCESS_POSTCONTACT_DIRECTION_RATIO")
    if outcome["maximum_lateral_excursion_m"] > 4:
        labels.append("OUT_OF_PLAY")
    if outcome["minimum_pelvis_z_m"] < 0.65:
        labels.append("LOW_PELVIS")
    return labels


def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
    records = manifest["records"]
    if (
        manifest.get("schema") != "soccer.rsi.smooth_memory_on_policy_bank.v1"
        or manifest.get("partition") != "TRAIN_CONSUMED"
        or manifest.get("sampling_rho") != 0.9
        or manifest.get("candidate_previous_mean_required") is not True
        or any(
            manifest.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
        )
        or not isinstance(records, list)
        or not records
        or type(manifest.get("physical_rollout_count")) is not int
        or manifest["physical_rollout_count"] != len(records)
        or type(manifest.get("frame_sample_count")) is not int
        or manifest["frame_sample_count"] != len(records) * 270
        or type(manifest.get("independent_contexts")) is not int
        or manifest.get("report_hash")
        != hash_json({k: v for k, v in manifest.items() if k != "report_hash"})
    ):
        raise ValueError("sealed complete consumed smooth-memory manifest required")
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for index, row in enumerate(records):
        if (
            type(row.get("group")) is not int
            or row["group"] != index
            or any(type(row.get(k)) is not int or row[k] < 0 for k in ("seed", "lane", "sample"))
        ):
            raise ValueError("ordered unique real trajectory identities required")
        grouped.setdefault((row["seed"], row["lane"]), []).append(row)
    if len(grouped) != manifest["independent_contexts"]:
        raise ValueError("reported independent course count differs from trajectories")
    count = len(records) // len(grouped)
    rows = []
    for (seed, lane), course in grouped.items():
        if [r["sample"] for r in course] != list(range(count)):
            raise ValueError("all declared samples of every course required; no selection")
        labels = [outcome_labels(r["outcome"]) for r in course]
        rows.append(
            dict(
                seed=seed,
                lane=lane,
                trajectories=len(course),
                high_quality=sum(r["outcome"]["high_quality"] for r in course),
                high_quality_and_safe=sum(
                    r["outcome"]["high_quality"] and "LOW_PELVIS" not in failure
                    for r, failure in zip(course, labels, strict=True)
                ),
                overlapping_failure_counts={
                    label: sum(label in failure for failure in labels)
                    for label in sorted({v for failure in labels for v in failure})
                },
            )
        )
    result = dict(
        schema="soccer.rsi.audited_curriculum_outcome_diagnostic.v1",
        manifest_hash=manifest["report_hash"],
        data_hash=manifest["data_hash"],
        rows=rows,
        physical_rollout_count=len(records),
        high_quality=sum(r["high_quality"] for r in rows),
        zero_success_courses=[[r["seed"], r["lane"]] for r in rows if r["high_quality"] == 0],
        physical_executions_added=0,
        qualification="CONSUMED_OUTCOME_COUNTS_NOT_CAUSAL_NOT_NEW_PHYSICAL_AUDIT",
        labels_are_overlapping=True,
        evidence_is_not_fresh=True,
        curriculum_changed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    return result


def check_dataset_shape(
    manifest: dict[str, Any], groups: Any, observation_shape: tuple[int, ...]
) -> None:
    """Do not accept resealed partial metadata pointing at a complete old NPZ."""
    groups = np.asarray(groups)
    count = manifest["physical_rollout_count"]
    frames = count * 270
    if (
        observation_shape != (frames, 134)
        or groups.dtype.kind not in "iu"
        or groups.shape != (frames,)
        or not np.array_equal(groups, np.repeat(np.arange(count), 270))
    ):
        raise ValueError("manifest must describe the entire ordered original numerical dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = _sealed(args.learning_root / "rollout_manifest.json")
    data = args.learning_root / "rollouts.npz"
    if hash_bytes(data.read_bytes()) != manifest["data_hash"]:
        raise ValueError("complete original audited numerical dataset hash required")
    result = summarize(manifest)
    with np.load(data, allow_pickle=False) as arrays:
        check_dataset_shape(manifest, arrays["trajectory_index"], arrays["observation"].shape)
    if hash_bytes(data.read_bytes()) != manifest["data_hash"]:
        raise ValueError("numerical dataset changed during diagnostic")
    if _sealed(args.learning_root / "rollout_manifest.json") != manifest:
        raise ValueError("audited manifest changed during diagnostic")
    result.pop("report_hash")
    result["diagnostic_source_hash"] = hash_bytes(Path(__file__).read_bytes())
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
