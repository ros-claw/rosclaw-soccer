"""Durable, hash-linked first-touch collection checkpoints, not physics admission.

Soccer pilot only. A caller must verify physics/model lineage before recording
an episode. This ledger binds that episode's witnesses and detects later drift;
it grants no training, fresh-exam, promotion or hardware authority.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.first_touch_outcome import outcome_contract
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.sim.contracts import hash_json

WITNESSES = frozenset({"trace", "replay", "foundation", "sampling_view", "label"})
STATES = frozenset(
    {
        "PREFLIGHT",
        "COLLECTING",
        "PARTIAL_DATA",
        "COLLECTED",
        "RESUMABLE_FAILURE",
        "FATAL_CONTRACT_VIOLATION",
    }
)


def file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("ordinary immutable witness file required")
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


class FirstTouchRecoveryLedger:
    """One exclusive writer; readers recheck every witness before skipping work."""

    def __init__(self, root: Path, plan: PublicFirstTouchTrainingPlan):
        if type(plan) is not PublicFirstTouchTrainingPlan:
            raise ValueError("exact public training declaration required")
        if root.is_symlink():
            raise ValueError("ordinary campaign root required")
        root.mkdir(parents=True, exist_ok=True)
        self.root, self.plan = root, plan
        self.declaration = dict(
            schema="soccer.rsi.first_touch_collection_recovery.v1",
            public_training_plan=plan.contract(),
            outcome_contract=outcome_contract(),
            training_authorized=False,
            private_fresh_accessed=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        self.declaration_hash = hash_json(self.declaration)
        with self._lock():
            path = root / "declaration.json"
            new = not path.exists()
            if new:
                with path.open("x") as stream:
                    json.dump(self.declaration, stream, sort_keys=True, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                self._sync_root()
            if (
                path.is_symlink()
                or hash_json(json.loads(path.read_bytes())) != self.declaration_hash
            ):
                raise ValueError("campaign declaration changed; fatal contract violation")
            journal = root / "journal.jsonl"
            if journal.is_symlink():
                raise ValueError("ordinary journal required")
            if not journal.exists():
                if not new:
                    raise ValueError("missing durable journal is fatal; never reset progress")
                journal.touch(exist_ok=False)
                self._sync_root()
            self._read()

    def _sync_root(self) -> None:
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _lock(self) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def locked() -> Any:
            path = self.root / "writer.lock"
            if path.is_symlink():
                raise ValueError("ordinary writer lock required")
            with path.open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)

        return locked()

    def _read(self) -> list[dict[str, Any]]:
        declaration = self.root / "declaration.json"
        journal = self.root / "journal.jsonl"
        if declaration.is_symlink() or journal.is_symlink():
            raise ValueError("ordinary durable state required")
        if hash_json(json.loads(declaration.read_bytes())) != self.declaration_hash:
            raise ValueError("campaign declaration mutated")
        events: list[dict[str, Any]] = []
        previous = self.declaration_hash
        raw = (self.root / "journal.jsonl").read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("torn journal is fatal; do not silently truncate evidence")
        seen = set()
        for line in raw.splitlines():
            event = json.loads(line)
            unsigned = {k: v for k, v in event.items() if k != "event_hash"}
            if (
                event.get("event_hash") != hash_json(unsigned)
                or event.get("previous_event_hash") != previous
                or event.get("declaration_hash") != self.declaration_hash
                or type(event.get("index")) is not int
                or event["index"] != len(events)
                or event.get("state") not in STATES
            ):
                raise ValueError("fatal recovery journal integrity violation")
            if events and events[-1]["state"] in {"COLLECTED", "FATAL_CONTRACT_VIOLATION"}:
                raise ValueError("terminal recovery state cannot be bypassed")
            episode = event.get("episode")
            if episode is not None:
                group = episode.get("group")
                if (
                    type(group) is not int
                    or not 0 <= group < len(self.plan.draws())
                    or group in seen
                ):
                    raise ValueError("unique declared physical episode required")
                if hash_json(episode.get("draw")) != hash_json(asdict(self.plan.draws()[group])):
                    raise ValueError("physical draw identity changed")
                witnesses = episode.get("witnesses")
                if type(witnesses) is not dict or set(witnesses) != WITNESSES:
                    raise ValueError("all episode witnesses required")
                for witness in witnesses.values():
                    if file_hash(Path(witness["path"])) != witness["sha256"]:
                        raise ValueError(
                            "fatal physical witness mutation; never resume by dropping it"
                        )
                seen.add(group)
            if event["state"] == "COLLECTED" and len(seen) != len(self.plan.draws()):
                raise ValueError("partial collection cannot be called complete")
            events.append(event)
            previous = event["event_hash"]
        return events

    def _append(self, events: list[dict[str, Any]], state: str, **payload: Any) -> None:
        event = dict(
            index=len(events),
            state=state,
            declaration_hash=self.declaration_hash,
            previous_event_hash=events[-1]["event_hash"] if events else self.declaration_hash,
            **payload,
        )
        event["event_hash"] = hash_json(event)
        with (self.root / "journal.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def status(self) -> dict[str, Any]:
        with self._lock():
            events = self._read()
            done = sorted(e["episode"]["group"] for e in events if e.get("episode") is not None)
            return dict(
                state=events[-1]["state"] if events else "PREFLIGHT",
                completed_groups=done,
                pending_groups=[d.group for d in self.plan.draws() if d.group not in done],
                declaration_hash=self.declaration_hash,
                head_hash=events[-1]["event_hash"] if events else self.declaration_hash,
                training_authorized=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )

    def record_episode(self, group: int, artifacts: dict[str, Path]) -> None:
        if type(group) is not int or not 0 <= group < len(self.plan.draws()):
            raise ValueError("declared group required")
        if type(artifacts) is not dict or set(artifacts) != WITNESSES:
            raise ValueError("all physical/model/label witnesses required")
        episode = dict(
            group=group,
            draw=asdict(self.plan.draws()[group]),
            witnesses={
                k: dict(path=str(p.resolve()), sha256=file_hash(p)) for k, p in artifacts.items()
            },
        )
        with self._lock():
            events = self._read()
            existing = [e["episode"] for e in events if e.get("episode", {}).get("group") == group]
            if existing:
                if hash_json(existing[0]) != hash_json(episode):
                    raise ValueError("cannot replace an already committed episode")
                return
            if events and events[-1]["state"] in {"FATAL_CONTRACT_VIOLATION", "COLLECTED"}:
                raise ValueError("terminal campaign cannot accept new work")
            count = sum(e.get("episode") is not None for e in events) + 1
            self._append(
                events,
                "COLLECTED" if count == len(self.plan.draws()) else "PARTIAL_DATA",
                episode=episode,
            )

    def failure(self, reason: str, *, fatal: bool = False) -> None:
        if type(reason) is not str or not 1 <= len(reason) <= 1024 or type(fatal) is not bool:
            raise ValueError("bounded explicit failure reason required")
        with self._lock():
            events = self._read()
            if events and events[-1]["state"] in {"FATAL_CONTRACT_VIOLATION", "COLLECTED"}:
                raise ValueError("terminal campaign must remain terminal")
            self._append(
                events, "FATAL_CONTRACT_VIOLATION" if fatal else "RESUMABLE_FAILURE", reason=reason
            )


def fork_verified_collection(
    source: Path, destination: Path, plan: PublicFirstTouchTrainingPlan, *, reason: str
) -> FirstTouchRecoveryLedger:
    """Explicit successor after incident review; never unfreeze the failed ledger.

    Reuses only previously committed, reverified witnesses. An uncommitted raw
    episode still needs independent physics review in the successor campaign.
    The reason is a provenance annotation, not a permission or safety bypass.
    """
    if type(reason) is not str or not 1 <= len(reason) <= 1024:
        raise ValueError("bounded incident-review reason required")
    if source.resolve() == destination.resolve():
        raise ValueError("failed campaign must remain immutable")
    old = FirstTouchRecoveryLedger(source, plan)
    status = old.status()
    if status["state"] != "FATAL_CONTRACT_VIOLATION":
        raise ValueError("explicit failed predecessor required")
    with old._lock():
        episodes = [event["episode"] for event in old._read() if event.get("episode")]
    lineage = dict(
        schema="soccer.rsi.first_touch_collection_successor.v1",
        predecessor=str(source.resolve()),
        predecessor_head=status["head_hash"],
        predecessor_journal_hash=file_hash(source / "journal.jsonl"),
        predecessor_declaration_hash=status["declaration_hash"],
        reason=reason,
        adopted_groups=status["completed_groups"],
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result = FirstTouchRecoveryLedger(destination, plan)
    with result._lock():
        path = destination / "predecessor.json"
        if path.exists():
            if path.is_symlink() or hash_json(json.loads(path.read_bytes())) != hash_json(lineage):
                raise ValueError("successor lineage cannot be replaced")
        else:
            if result._read():
                raise ValueError("existing collection cannot acquire a new predecessor")
            with path.open("x") as stream:
                json.dump(lineage, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            result._sync_root()
    for episode in episodes:
        result.record_episode(
            episode["group"],
            {key: Path(value["path"]) for key, value in episode["witnesses"].items()},
        )
    if old.status() != status:
        raise ValueError("predecessor changed during explicit recovery")
    return result
