"""Verify passive witness against stored native physics; development only."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiving_witness_audit import audit_receiving_witness
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("root-hash", "source-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external review output required")
    paths = (a.root / "complete.json", a.source_root / "complete.json")
    if any(
        path.is_symlink() or digest(path) != expected
        for path, expected in zip(paths, (a.root_hash, a.source_hash), strict=True)
    ):
        raise ValueError("exact source-bound completed receipts required")
    new, old = (json.loads(path.read_text()) for path in paths)
    if (
        new["status"] != "PASSIVE_WITNESS_REPLAY_EXACT_DEVELOPMENT_ONLY"
        or new["source_hash"] != a.source_hash
        or len(new["records"]) != 8
        or len(old["records"]) != 16
    ):
        raise ValueError("exact eight-world replay and original source required")
    declaration = a.root / "declaration.json"
    if digest(declaration) != new["declaration_hash"]:
        raise ValueError("exact replay declaration required")
    pins = json.loads(declaration.read_text())["input_file_hashes"]
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1 or digest(scenes[0]) != pins[str(scenes[0])]:
        raise ValueError("exact source scene required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/", "b/"))
    )
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    reports = []
    for index, row in enumerate(new["records"]):
        source = old["records"][index * 2]
        if (
            row["index"] != index
            or source["arm"] != "dual"
            or any(
                row[k] != source[k]
                for k in (
                    "context",
                    "lateral_m",
                    "termination",
                    "controls",
                    "native_steps",
                    "handoff",
                )
            )
        ):
            raise ValueError("source trial mismatch")
        if set(row["artifacts"]) != set(source["artifacts"]):
            raise ValueError("source artifact set mismatch")
        for name, item in row["artifacts"].items():
            prior = source["artifacts"].get(name)
            path, prior_path = (
                a.root / item["path"],
                a.source_root / prior["path"] if prior else None,
            )
            if (
                prior_path is None
                or path.is_symlink()
                or prior_path.is_symlink()
                or not path.resolve().is_relative_to(a.root.resolve())
                or not prior_path.resolve().is_relative_to(a.source_root.resolve())
                or digest(path) != item["hash"]
                or digest(prior_path) != prior["hash"]
            ):
                raise ValueError("confined matching artifacts required")
            with (
                np.load(path, allow_pickle=False) as x,
                np.load(prior_path, allow_pickle=False) as y,
            ):
                if set(x.files) != set(y.files) or any(
                    not np.array_equal(x[k], y[k]) for k in x.files
                ):
                    raise ValueError("independent source array comparison failed")
        witness_path = a.root / row["witness"]["path"]
        if (
            witness_path.is_symlink()
            or witness_path.resolve().parent != (a.root / f"trial{index}").resolve()
            or digest(witness_path) != row["witness"]["hash"]
        ):
            raise ValueError("exact confined witness required")
        witness = json.loads(witness_path.read_text())
        with np.load(a.root / row["artifacts"]["physics"]["path"], allow_pickle=False) as physics:
            report = audit_receiving_witness(
                physics,
                witness,
                receiver_q=layout.rq,
                receiver_v=layout.rv,
                ball_q=layout.bq,
                ball_v=layout.bv,
            )
        expected_gate_controls = (
            row["handoff"]["control"]
            if row["handoff"]
            else row["controls"] - (row["termination"] == "BODY_FAILURE")
        )
        if report["gate_controls"] != expected_gate_controls or (
            row["handoff"] and report["first_ready_control"] != row["handoff"]["control"]
        ):
            raise ValueError("readiness observations disagree with handoff or failure")
        reports.append(
            dict(
                context=row["context"],
                lateral_m=row["lateral_m"],
                termination=row["termination"],
                **report,
            )
        )
    clean(a.expected_head)
    write_once(
        a.output,
        dict(
            status="INDEPENDENT_PASSIVE_WITNESS_AUDIT_DEVELOPMENT_ONLY",
            source_hash=a.source_hash,
            witness_hash=a.root_hash,
            code_head=a.expected_head,
            rows=reports,
            hardware_authorized=False,
            promotion_authorized=False,
            team_game_verified=False,
        ),
    )


if __name__ == "__main__":
    main()
