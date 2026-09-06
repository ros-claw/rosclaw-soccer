"""Train a bounded contact-handoff actor from failure-driven discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rosclaw_soccer.growth.contact_handoff_actor import (
    G1ContactHandoffActor,
    save_contact_handoff_actor,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def train_contact_handoff_actor(*, discovery_report_path: Path, output_dir: Path) -> dict[str, Any]:
    source_path = discovery_report_path.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    claimed = source.pop("report_hash", None)
    if (
        claimed != hash_json(source)
        or source.get("status") != "PASS_CONTACT_HANDOFF_DISCOVERY"
        or source.get("promotion_eligible") is not False
        or not source.get("passing_handoff_policy_frames")
    ):
        raise ValueError("contact handoff growth requires passing discovery evidence")
    source["report_hash"] = claimed
    for row in source["rows"]:
        artifact = row["trajectory"]
        path = source_path.parent / artifact["file"]
        if hash_bytes(path.read_bytes()) != artifact["file_hash"]:
            raise ValueError("contact handoff trajectory changed")
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            trajectory = {name: np.asarray(archive[name]) for name in archive.files}
        if trajectory_digest(trajectory) != artifact["trajectory_digest"]:
            raise ValueError("contact handoff trajectory digest changed")
    passing = [int(value) for value in source["passing_handoff_policy_frames"]]
    summaries = source["frame_summaries"]
    selected_frame = sorted(
        passing,
        key=lambda frame: (
            -float(summaries[str(frame)]["minimum_shooter_pelvis_height_m"]),
            float(summaries[str(frame)]["mean_shooter_tail_wobble_index"]),
            frame,
        ),
    )[0]
    contact_frames = {
        int(row["plan_decision"]["action"]["contact_policy_frame"]) for row in source["rows"]
    }
    if len(contact_frames) != 1 or selected_frame < next(iter(contact_frames)):
        raise ValueError("contact handoff discovery has inconsistent strike timing")
    contact_frame = next(iter(contact_frames))
    output = _new_external_output(output_dir)
    summary = summaries[str(selected_frame)]
    actor = G1ContactHandoffActor(
        body_hash=source["body_hash"],
        target_plan_actor_hash=source["target_plan_actor_hash"],
        target_contact_actor_hash=source["target_contact_actor_hash"],
        source_evidence_hashes=(claimed,),
        selected_offset_frames=selected_frame - contact_frame,
        evaluated_offset_frames=tuple(
            int(frame) - contact_frame for frame in sorted(summaries, key=int)
        ),
        safe_case_count=int(summary["safe_count"]),
        recovered_failure_count=int(summary["recovered_failure_count"]),
        training_case_count=len(source["rows"]) // len(summaries),
    )
    actor_path = output / "contact-handoff-actor.json"
    save_contact_handoff_actor(actor, actor_path)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.contact_handoff_training.v1",
        "status": "PASS_CONTACT_HANDOFF_TRAINING",
        "promotion_eligible": False,
        "source_discovery_report_hash": claimed,
        "source_discovery_file_hash": hash_bytes(source_path.read_bytes()),
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "selected_handoff_policy_frame": selected_frame,
        "selected_offset_frames": actor.selected_offset_frames,
        "metrics": summary,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "training-report.json", report)
    return report


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("contact handoff actor output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["train_contact_handoff_actor"]
