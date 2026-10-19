"""Durable pre-action SIM evidence, never an action authorization surface."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.early_impact_controller import EarlyImpactCommand
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


class EarlyImpactEvidence:
    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise ValueError("unused local evidence path required")
        if root.exists() or root.is_symlink() or root.parent.is_symlink():
            raise ValueError("unused ordinary evidence directory required")
        root.mkdir(mode=0o700)
        self.root = root
        self.next_frame = 0
        self.failed = False

    def record(self, frame: int, command: EarlyImpactCommand) -> None:
        if (
            self.failed
            or type(frame) is not int
            or frame != self.next_frame
            or not 0 <= frame < 300
        ):
            self.failed = True
            raise ValueError("one contiguous pre-action evidence record required")
        self.failed = True
        if (
            type(command) is not EarlyImpactCommand
            or type(command.report.get("frame")) is not int
            or command.report.get("frame") != frame
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
            or not isinstance(command.offset_nm, np.ndarray)
            or command.offset_nm.shape != (29,)
            or command.offset_nm.dtype != np.float64
            or not np.isfinite(command.offset_nm).all()
            or not np.array_equal(command.report.get("requested_offset_nm", []), command.offset_nm)
        ):
            raise ValueError("same-frame unqualified finite SIM command required")
        witnesses = []
        for i, candidate in enumerate(command.candidates):
            forecast = candidate.forecast
            if len(forecast.next_target_records) != 3 or len(forecast.next_neural_records) != 3:
                raise ValueError("three actual updated targets and NN captures required")
            values: dict[str, Any] = dict(
                forecast.arrays,
                end_integration_state=forecast.end_integration_state,
                offset_nm=candidate.offset_nm,
            )
            for j, (target, neural) in enumerate(
                zip(forecast.next_target_records, forecast.next_neural_records, strict=True),
                start=1,
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
                    donor_hash=forecast.donor_integration_state_hash,
                    offset_nm=candidate.offset_nm.tolist(),
                    score=candidate.score,
                )
            )
        record: dict[str, Any] = dict(
            command.report,
            schema="soccer.rsi.preaction_early80ms_decision.v1",
            raw_witnesses=witnesses,
            persisted_before_current_native_substeps=True,
        )
        record["report_hash"] = hash_json(record)
        path = self.root / f"frame-{frame:03d}.json"
        with path.open("x") as text:
            json.dump(record, text, sort_keys=True, indent=2, allow_nan=False)
            text.write("\n")
            text.flush()
            os.fsync(text.fileno())
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        # A crash before here is terminal, not permission to replay this action.
        self.next_frame += 1
        self.failed = False
