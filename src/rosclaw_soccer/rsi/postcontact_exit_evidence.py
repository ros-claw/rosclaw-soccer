"""Distinct durable postcontact exit decisions, persisted before native action."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.postcontact_exit_controller import PostContactExitCommand
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


class PostContactExitEvidence:
    def __init__(self, root: Path) -> None:
        if (
            not isinstance(root, Path)
            or root.exists()
            or root.is_symlink()
            or root.parent.is_symlink()
        ):
            raise ValueError("unused ordinary postcontact exit evidence directory required")
        root.mkdir(mode=0o700)
        self.root, self.next_frame, self.failed = root, 0, False

    def record(self, frame: int, command: PostContactExitCommand) -> None:
        if self.failed:
            raise ValueError("failed exit evidence writer cannot retry")
        self.failed = True
        if (
            type(frame) is not int
            or frame != self.next_frame
            or not 0 <= frame < 300
            or type(command) is not PostContactExitCommand
            or command.report.get("schema") != "soccer.rsi.postcontact80ms_exit_plan.v1"
            or type(command.report.get("frame")) is not int
            or command.report["frame"] != frame
            or any(
                command.report.get(k) is not False
                for k in (
                    "future_source_frame_or_target_used",
                    "full_h300_gain_proven",
                    "neural_policy_trained",
                    "teacher_qualified",
                    "hardware_authorized",
                    "promotion_authorized",
                )
            )
            or type(command.offset_nm) is not np.ndarray
            or command.offset_nm.shape != (29,)
            or command.offset_nm.dtype != np.float64
            or not np.isfinite(command.offset_nm).all()
            or not np.array_equal(command.report.get("requested_offset_nm", []), command.offset_nm)
        ):
            raise ValueError("same-frame unqualified exact postcontact exit command required")
        witnesses = []
        for i, candidate in enumerate(command.candidates):
            f = candidate.forecast
            if (
                len(f.next_target_records) != 3
                or len(f.next_neural_records) != 3
                or candidate.score.get("schema") != "soccer.rsi.postcontact80ms_exit_proxy.v1"
            ):
                raise ValueError("three actual feedback captures and distinct exit score required")
            values: dict[str, Any] = dict(
                f.arrays,
                end_integration_state=f.end_integration_state,
                offset_nm=candidate.offset_nm,
            )
            for j, (target, neural) in enumerate(
                zip(f.next_target_records, f.next_neural_records, strict=True), start=1
            ):
                values.update({f"next_{j}_" + k: v for k, v in target.items()})
                values.update({f"neural_next_{j}_" + k: v for k, v in neural.items()})
            path = self.root / f"frame-{frame:03d}-candidate-{i:02d}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **values)
                stream.flush()
                os.fsync(stream.fileno())
            witnesses.append(
                dict(
                    name=candidate.name,
                    path=str(path),
                    sha256=hash_bytes(path.read_bytes()),
                    donor_hash=f.donor_integration_state_hash,
                    offset_nm=candidate.offset_nm.tolist(),
                    score=candidate.score,
                )
            )
        record = dict(
            command.report,
            schema="soccer.rsi.preaction_postcontact80ms_exit_decision.v1",
            raw_witnesses=witnesses,
            persisted_before_current_native_substeps=True,
        )
        record["report_hash"] = hash_json(record)
        with (self.root / f"frame-{frame:03d}.json").open("x") as stream:
            json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.next_frame += 1
        self.failed = False
