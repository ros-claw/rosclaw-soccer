"""SIM-only complete consumed CPU + inherited/GPU anchor protection.

The added anchors affect only the new plastic residual gate. They do not
rewrite the legacy memory, parent output, encoder, reward or physical world.
Real replay is performed by the evidence builder, never inferred from hashes.
"""

from pathlib import Path
from typing import Any

import numpy as np
from rosclaw.growth.domain_anchor_bank import validate_domain_anchor_bank

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.multi_domain_memory_protection.v1"
FLAGS = ("physical_policy_execution_qualified", "promotion_authorized", "hardware_authorized")


def _sealed(value: Any) -> bool:
    return type(value) is dict and value.get("report_hash") == hash_json(
        {k: v for k, v in value.items() if k != "report_hash"}
    )


def validate_protection(value: dict[str, Any], initial: dict[str, Any]) -> None:
    if (
        type(value) is not dict
        or value.get("schema") != SCHEMA
        or value.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or not _sealed(value)
        or any(value.get(k) is not False for k in FLAGS)
        or value.get("selection") != "ALL_SEALED_CPU_SUCCESSES_PLUS_INTACT_INHERITED_GPU_MEMORY"
    ):
        raise ValueError("complete sealed multi-domain SIM-only protection required")
    bank = value["bank"]
    validate_domain_anchor_bank(bank)
    legacy = initial["baseline"]
    memory, manifest = legacy["memory"], legacy["consolidation_manifest"]
    if (
        bank["parent_hash"] != initial["parent_model_hash"]
        or bank["encoder_hash"] != memory["encoder_hash"]
        or len(bank["domains"]) != 2
    ):
        raise ValueError("same frozen parent/encoder and both declared domains required")
    inherited, cpu = bank["domains"]
    if (
        inherited["domain_id"] != "inherited-gpu-memory"
        or inherited["source_evidence_hash"] != manifest["report_hash"]
        or inherited["observations"] != memory["observations"]
        or inherited["context_ids"] != ["intact-legacy-memory"]
        or inherited["context_rows"] != [len(memory["observations"])]
        or cpu["domain_id"] != "cpu-retained-parent"
    ):
        raise ValueError("all original memory rows and domain provenance must remain intact")
    summary, commitment = value["cpu_summary"], value["cpu_commitment"]
    if (
        not _sealed(summary)
        or not _sealed(commitment)
        or summary.get("schema") != "soccer.rsi.cpu_retained_parent_full_coverage.v1"
        or commitment.get("schema") != "soccer.rsi.cpu_retained_parent_full_coverage_commitment.v1"
        or summary.get("commitment_hash") != commitment["report_hash"]
        or commitment.get("model_hash") != initial["parent_model_hash"]
        or commitment.get("partition") != "TRAIN_CONSUMED_CPU_DOMAIN"
        or summary.get("independent_contexts") != 52
        or len(summary["rows"]) != 52
        or summary.get("all_physical_substeps_replayed") != 156000
        or cpu["source_evidence_hash"] != summary["report_hash"]
        or any(
            item.get(k) is not False
            for item in (summary, commitment)
            for k in (
                "fresh_exam_authorized",
                "learning_authorized",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
        or commitment.get("physics_change") is not False
        or [[r["seed"], r["lane"]] for r in summary["rows"]] != commitment.get("courses")
        or [r["index"] for r in summary["rows"]] != list(range(52))
    ):
        raise ValueError("complete consumed CPU parent evidence required")
    successes = [r for r in summary["rows"] if r["outcome"]["high_quality"] is True]
    records = value["cpu_records"]
    if (
        not successes
        or len(successes) != summary["high_quality"]
        or len(records) != len(successes)
        or cpu["context_ids"] != [f"seed{r['seed']}-lane{r['lane']}" for r in successes]
        or cpu["context_rows"] != [270] * len(successes)
    ):
        raise ValueError("ALL CPU successes in sealed order required; no selected subset")
    for i, (row, record) in enumerate(zip(successes, records, strict=True)):
        outcome = row["outcome"]
        states = cpu["observations"][i * 270 : (i + 1) * 270]
        if (
            any(
                record.get(k) != row[k]
                for k in ("index", "seed", "lane", "report_hash", "review_hash")
            )
            or record.get("frames") != 270
            or record.get("state_hash") != hash_json(states)
            or outcome.get("physical_substeps") != 3000
            or outcome.get("reviewed_report_hash") != row["report_hash"]
            or outcome.get("report_hash") != row["review_hash"]
            or not _sealed(outcome)
            or any(
                outcome.get(k) is not True
                for k in (
                    "actual_mujoco_dynamics_replayed",
                    "actual_pd_torque_reconstructed",
                    "neural_target_reconstructed",
                    "clean_foot_only",
                    "safety_passed",
                )
            )
            or any(
                outcome.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
            )
            or np.asarray(states).shape != (270, 135)
        ):
            raise ValueError("complete reconstructed success state lineage required")


def protection_identity(model: dict[str, Any]) -> tuple[str, int, int]:
    value = model["protected_domain_bank"]
    initial = model["initial_actor"]
    if value is None:
        legacy = initial["baseline"]
        return (
            legacy["memory"]["memory_hash"],
            len(legacy["memory"]["observations"]),
            len(legacy["consolidation_manifest"]["records"]),
        )
    return (
        value["bank"]["bank_hash"],
        value["bank"]["row_count"],
        len(initial["baseline"]["consolidation_manifest"]["records"]) + len(value["cpu_records"]),
    )
