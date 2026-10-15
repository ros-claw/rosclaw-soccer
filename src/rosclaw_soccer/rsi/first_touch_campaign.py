"""Durable Soccer pilot for unattended, bounded first-touch research.

This is a SIM experiment scheduler, not a policy activation or hardware path.
Physics and teacher admission remain separate from process liveness.
"""

import hashlib
import json
import math
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json

TERMINAL_PHASES = frozenset(
    {
        "FATAL_CONTRACT_VIOLATION",
        "PLATEAU_RESEARCH_REQUIRED",
        "ATTEMPT_BUDGET_EXHAUSTED",
        "WALLTIME_BUDGET_EXHAUSTED",
        "GENERATION_BUDGET_EXHAUSTED",
        "PILOT_COMPLETE_EXPANDED_VALIDATION_REQUIRED",
    }
)


def load_verified_predecessor(specification: dict[str, Any]) -> dict[str, Any]:
    """Read a stopped, explicitly pinned predecessor without rewriting it."""
    path = Path(specification["ledger_path"])
    if (
        not path.is_absolute()
        or path.resolve().parent != Path(specification["evidence_root"]).resolve()
    ):
        raise ValueError("explicit predecessor-owned ledger required")
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("predecessor must be stopped and checkpointed before adoption")
    with path.open("rb") as stream:
        digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != specification["ledger_hash"]:
        raise ValueError("predecessor ledger bytes changed")
    previous = "GENESIS"
    state: dict[str, Any] = {}
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        for seq, link, body, digest in db.execute("SELECT * FROM events ORDER BY seq"):
            document = json.loads(body)
            if link != previous or digest != hash_json(dict(seq=seq, previous=link, body=document)):
                raise ValueError("predecessor event integrity violation")
            state = document["state"]
            previous = digest
    if (
        previous != specification["ledger_tip"]
        or state.get("declaration_hash") != specification["declaration_hash"]
        or state.get("active") is not None
        or state.get("phase") != "STOPPED_OPERATOR_REQUEST"
        or state.get("hardware_authorized") is not False
        or state.get("promotion_authorized") is not False
    ):
        raise ValueError(
            "explicit quiescent predecessor required; fatal or terminal states cannot migrate"
        )
    return state


class FirstTouchCampaign:
    """Append-only transactional event ledger with source-bound restart state."""

    def __init__(self, root: Path, declaration: dict[str, Any]) -> None:
        if (
            declaration.get("mode") != "SIM_ONLY"
            or declaration.get("hardware_authorized") is not False
            or declaration.get("promotion_authorized") is not False
            or declaration.get("private_fresh_accessed") is not False
        ):
            raise ValueError("SIM-only public research declaration required")
        for name in ("max_attempts", "max_generations", "plateau_generations"):
            value = declaration.get(name)
            if type(value) is not int or value < 1:
                raise ValueError("positive finite research budgets required")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        self.db = sqlite3.connect(root / "campaign.sqlite", isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events "
            "(seq INTEGER PRIMARY KEY, previous TEXT NOT NULL, "
            "body TEXT NOT NULL, digest TEXT NOT NULL)"
        )
        self.state: dict[str, Any] = {}
        self.tip = "GENESIS"
        for seq, previous, body, digest in self.db.execute("SELECT * FROM events ORDER BY seq"):
            document = json.loads(body)
            if previous != self.tip or digest != hash_json(
                dict(seq=seq, previous=previous, body=document)
            ):
                raise ValueError("campaign event integrity violation; no automatic repair")
            self.state = document["state"]
            self.tip = digest
        commitment = hash_json(declaration)
        if self.state:
            if self.state["declaration_hash"] != commitment:
                raise ValueError(
                    "campaign source, plan or budget changed; explicit successor required"
                )
        else:
            self.state = dict(
                declaration_hash=commitment,
                phase="PREFLIGHT",
                attempts=0,
                completed={},
                active=None,
                generations=[],
                no_gain_generations=0,
                promotion_authorized=False,
                hardware_authorized=False,
            )
            self.record("CREATED")
        self.declaration = declaration
        for receipt in self.state["completed"].values():
            self.verify_receipt(receipt)

    def verify_receipt(self, receipt: dict[str, Any]) -> None:
        witnesses = receipt.get("witnesses")
        if type(witnesses) is not dict or set(witnesses) != {
            "trace",
            "report",
            "independent_review",
            "foundation_review",
            "commitment",
            "world",
            "proof_payload",
        }:
            raise ValueError("complete immutable physical witnesses required")
        for witness in witnesses.values():
            path = Path(witness["path"])
            roots = [self.root]
            predecessor = self.declaration.get("predecessor")
            if predecessor is not None:
                roots.append(Path(predecessor["evidence_root"]).resolve())
            if not path.is_absolute() or not any(
                path.resolve().is_relative_to(root) for root in roots
            ):
                raise ValueError("campaign-owned evidence path required")
            with path.open("rb") as stream:
                actual = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != witness["sha256"]:
                raise ValueError("completed campaign witness bytes changed")

    def adopt_predecessor(self) -> None:
        specification = self.declaration.get("predecessor")
        if specification is None:
            return
        if self.state.get("adopted_predecessor_tip") == specification["ledger_tip"]:
            return
        if self.state["attempts"] or self.state["completed"] or self.state["generations"]:
            raise ValueError("predecessor can only initialize an unused successor")
        previous = load_verified_predecessor(specification)
        if previous["attempts"] > self.declaration["max_attempts"]:
            raise ValueError("inherited attempts cannot exceed the existing budget")
        for receipt in previous["completed"].values():
            self.verify_receipt(receipt)
        for key in ("attempts", "completed", "generations", "no_gain_generations", "populations"):
            if key in previous:
                self.state[key] = previous[key]
        self.state["adopted_predecessor_tip"] = specification["ledger_tip"]
        self.state["phase"] = "READY_FOR_NEXT_JOB"
        self.record("EXPLICIT_OPERATIONAL_SUCCESSOR_ADOPTED_NO_PHYSICS_RERUN")

    def record(self, event: str) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            tail = self.db.execute(
                "SELECT seq,digest FROM events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            seq = 1 if tail is None else tail[0] + 1
            previous = "GENESIS" if tail is None else tail[1]
            if previous != self.tip:
                raise ValueError("another campaign writer advanced the ledger")
            document = dict(event=event, time_unix=time.time(), state=self.state)
            digest = hash_json(dict(seq=seq, previous=previous, body=document))
            body = json.dumps(document, sort_keys=True, allow_nan=False)
            self.db.execute("INSERT INTO events VALUES (?,?,?,?)", (seq, previous, body, digest))
            self.db.execute("COMMIT")
            self.tip = digest
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def recover_interrupted_attempt(self) -> None:
        if self.state["phase"] in TERMINAL_PHASES:
            raise ValueError("terminal campaign cannot automatically resume")
        if self.state["active"] is not None:
            # No success is invented from an uncommitted directory. A new
            # attempt uses another path and consumes the original attempt too.
            self.state["active"] = None
            self.state["phase"] = "RESUMABLE_FAILURE"
            self.record("INTERRUPTED_ATTEMPT_PRESERVED")

    def begin(self, job: str) -> int:
        if self.state["phase"] in TERMINAL_PHASES:
            raise ValueError("terminal campaign cannot execute another attempt")
        if job in self.state["completed"]:
            raise ValueError("verified job cannot be executed twice")
        if self.state["active"] is not None:
            raise ValueError("one active simulation attempt required")
        if self.state["attempts"] >= self.declaration["max_attempts"]:
            raise ValueError("attempt budget exhausted")
        attempt = self.state["attempts"]
        if type(attempt) is not int:
            raise ValueError("integer attempt counter required")
        self.state["attempts"] += 1
        self.state["active"] = dict(job=job, attempt=attempt)
        self.state["phase"] = "SIMULATING_AND_INDEPENDENTLY_REPLAYING"
        self.record("ATTEMPT_STARTED")
        return attempt

    def complete(self, job: str, receipt: dict[str, Any]) -> None:
        active = self.state["active"]
        if active is None or active["job"] != job:
            raise ValueError("terminal receipt must bind the active job")
        required = {
            "physical_substeps": 3000,
            "foundation_calls": 300,
            "sampling_draws": 270,
            "promotion_authorized": False,
            "hardware_authorized": False,
        }
        if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in required.items()):
            raise ValueError("whole independent SIM verification required")
        self.verify_receipt(receipt)
        self.state["completed"][job] = receipt
        self.state["active"] = None
        self.state["phase"] = "READY_FOR_NEXT_JOB"
        self.record("VERIFIED_EPISODE_COMMITTED")

    def reuse(self, job: str, original_job: str, identity: str) -> dict[str, Any]:
        if self.state["phase"] in TERMINAL_PHASES:
            raise ValueError("terminal campaign cannot automatically reuse work")
        if job in self.state["completed"] or self.state["active"] is not None:
            raise ValueError("unused inactive job required for evidence reuse")
        receipt = self.state["completed"][original_job]
        if receipt.get("evaluation_identity") != identity:
            raise ValueError("same course, parameters and sampling identity required")
        self.verify_receipt(receipt)
        adopted = dict(receipt, reused_from_job=original_job)
        self.state["completed"][job] = adopted
        self.record("IDENTICAL_VERIFIED_EPISODE_REUSED_NOT_NEW_PHYSICS")
        return adopted

    def finish_generation(self, summary: dict[str, Any]) -> str:
        if self.state["phase"] in TERMINAL_PHASES:
            raise ValueError("terminal campaign cannot automatically iterate")
        if self.state["active"] is not None:
            raise ValueError("active work cannot finish a generation")
        if len(self.state["generations"]) >= self.declaration["max_generations"]:
            raise ValueError("generation budget exhausted")
        if type(summary.get("paired_development_gain")) is not bool:
            raise ValueError("explicit physical gain decision required")
        self.state["generations"].append(summary)
        self.state["no_gain_generations"] = (
            0 if summary["paired_development_gain"] else self.state["no_gain_generations"] + 1
        )
        if self.state["no_gain_generations"] >= self.declaration["plateau_generations"]:
            phase = "PLATEAU_RESEARCH_REQUIRED"
        elif len(self.state["generations"]) >= self.declaration["max_generations"]:
            phase = "GENERATION_BUDGET_EXHAUSTED"
        else:
            phase = "AUTOMATIC_NEXT_GENERATION"
        self.state["phase"] = phase
        self.record("GENERATION_REVIEWED")
        return phase


def summarize_first_touch_candidate(
    outcomes: list[dict[str, Any]], baselines: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(outcomes) != len(baselines) or len(outcomes) < 4:
        raise ValueError("complete paired multistate outcomes required")
    for row in outcomes + baselines:
        if (
            type(row.get("safe")) is not bool
            or type(row.get("task_pass")) is not bool
            or (row["task_pass"] and not row["safe"])
            or type(row.get("training_reward")) not in (float, int)
            or not math.isfinite(row["training_reward"])
        ):
            raise ValueError("finite primitive physical outcomes required")
    rescue = sum(
        r["task_pass"] and not b["task_pass"] for r, b in zip(outcomes, baselines, strict=True)
    )
    regression = sum(
        b["task_pass"] and not r["task_pass"] for r, b in zip(outcomes, baselines, strict=True)
    )
    safety = sum(b["safe"] and not r["safe"] for r, b in zip(outcomes, baselines, strict=True))
    return dict(
        contexts=len(outcomes),
        high_quality=sum(r["task_pass"] for r in outcomes),
        baseline_high_quality=sum(r["task_pass"] for r in baselines),
        safe_states=sum(r["safe"] for r in outcomes),
        foot_contacts=sum(r.get("first_foot_contact_frame") is not None for r in outcomes),
        observed_ball_outcomes=sum(r.get("quality_observed") is True for r in outcomes),
        event_counts=dict(Counter(str(r.get("event_status", "TEST_FIXTURE")) for r in outcomes)),
        rescue=rescue,
        regression=regression,
        safety_regression=safety,
        mean_reward=sum(r["training_reward"] for r in outcomes) / len(outcomes),
        paired_development_gain=rescue > regression and regression == 0 and safety == 0,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
