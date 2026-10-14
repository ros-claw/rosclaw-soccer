"""Curate every collected label and existing witness, without new physics.

This verifies receipt consistency and rederives measured labels. It does not
rerun physics, reconstruct neural sequences, or authorize learning/promotion.
"""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch, outcome_contract
from rosclaw_soccer.rsi.first_touch_recovery import FirstTouchRecoveryLedger, file_hash
from rosclaw_soccer.sim.contracts import hash_json


def sealed_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("ordinary complete receipt required")
    value = json.loads(path.read_bytes())
    if type(value) is not dict or value.get("report_hash") != hash_json(
        {k: v for k, v in value.items() if k != "report_hash"}
    ):
        raise ValueError("sealed receipt required")
    if any(value.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")):
        raise ValueError("SIM-only receipt required")
    return value


def curate_collection(ledger: FirstTouchRecoveryLedger) -> dict[str, Any]:
    """Retain all declared failures/censoring; partial collections are refused."""
    status = ledger.status()
    draws = ledger.plan.draws()
    if status["state"] != "COLLECTED" or status["completed_groups"] != list(range(len(draws))):
        raise ValueError("whole declared collection required before curation")
    journal = ledger.root / "journal.jsonl"
    pins = {
        str(path.resolve()): file_hash(path) for path in (journal, ledger.root / "declaration.json")
    }
    events = [json.loads(line) for line in journal.read_bytes().splitlines()]
    episodes = {e["episode"]["group"]: e["episode"] for e in events if e.get("episode")}
    records = []
    counts: Counter[str] = Counter()
    strata = [
        dict(episodes=0, foot_contact=0, safe_foot_contact=0, high_quality=0) for _ in range(4)
    ]
    for draw in draws:
        witnesses = episodes[draw.group]["witnesses"]
        paths = {key: Path(value["path"]) for key, value in witnesses.items()}
        pins.update({value["path"]: value["sha256"] for value in witnesses.values()})
        receipt = sealed_receipt(paths["label"])
        replay = sealed_receipt(paths["replay"])
        foundation = sealed_receipt(paths["foundation"])
        with np.load(paths["trace"], allow_pickle=False) as archive:
            label = label_first_touch({key: archive[key] for key in archive.files}).to_dict()
        if (
            hash_json(label) != hash_json(receipt.get("label"))
            or receipt.get("outcome_contract_hash") != outcome_contract()["contract_hash"]
            or replay.get("reviewed_report_hash") != receipt.get("physical_report_hash")
            or foundation.get("physical_report_hash") != receipt.get("physical_report_hash")
            or foundation.get("physical_trace_hash") != witnesses["trace"]["sha256"]
            or replay.get("physical_substeps") != 3000
            or replay.get("actual_recurrent_sampling_draws_reconstructed") != 270
            or replay.get("actual_mujoco_dynamics_replayed") is not True
            or replay.get("actual_pd_torque_reconstructed") is not True
            or foundation.get("recomputed_executed_calls") != 300
            or replay.get("high_quality") is not label["task_pass"]
        ):
            raise ValueError("raw outcome and complete independent witnesses must agree")
        counts[label["event_status"]] += 1
        bucket = strata[2 * int(draw.lane >= 8) + draw.lane % 2]
        bucket["episodes"] += 1
        bucket["foot_contact"] += int(label["first_foot_contact_frame"] is not None)
        bucket["safe_foot_contact"] += int(
            label["safe"] is True and label["first_foot_contact_frame"] is not None
        )
        bucket["high_quality"] += int(label["task_pass"] is True)
        records.append(
            dict(
                draw=asdict(draw),
                witnesses=witnesses,
                label=label,
                physical_report_hash=receipt["physical_report_hash"],
                independent_review_hash=replay["report_hash"],
                foundation_review_hash=foundation["report_hash"],
            )
        )
    if ledger.status() != status or any(
        file_hash(Path(name)) != expected for name, expected in pins.items()
    ):
        raise ValueError("collection changed during curation")
    result = dict(
        schema="soccer.rsi.curated_first_touch_labels.v1",
        stage="CURATED_LABELS_ONLY",
        public_training_plan=ledger.plan.contract(),
        outcome_contract=outcome_contract(),
        collection_status=status,
        episodes=len(draws),
        independent_contexts=len(draws) // 4,
        strata_order=["low_x_negative", "low_x_positive", "high_x_negative", "high_x_positive"],
        strata=strata,
        event_counts=dict(counts),
        records=records,
        input_file_hashes=pins,
        repeated_draws_are_independent_scenarios=False,
        all_success_and_failure_rows_retained=True,
        new_physics_executions=0,
        new_foundation_calls=0,
        neural_sequence_reconstruction_performed=False,
        complete_learning_bank=False,
        new_optimizer_updates=0,
        training_authorized=False,
        private_fresh_accessed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
