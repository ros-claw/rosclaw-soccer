"""Summarize auditable substep timing; no controller or promotion authority."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.pass_contact_timing import contact_timing
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_fresh_common import clean


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", nargs=3, action="append", required=True, metavar=("SOURCE", "AUDIT", "AUDIT_HASH")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if args.output.exists() or args.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    results = []
    provenance = []
    roots = set()
    allowed = {
        "CAPTURE_GEOMETRY_SWEEP_DEVELOPMENT32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS",
        "CAPTURE_PRIORITY_SWEEP_DEVELOPMENT32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS",
        "CAPTURE_BILATERAL_SWEEP_DEVELOPMENT32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS",
    }
    for source_name, audit_name, expected_hash in args.pair:
        source, audit = Path(source_name), Path(audit_name)
        if source.resolve() in roots or digest(audit) != expected_hash:
            raise ValueError("duplicate source or mismatched audit pin")
        roots.add(source.resolve())
        source_file = source / "complete.json"
        source_hash = digest(source_file)
        reviewed = json.loads(audit.read_text())
        report = json.loads(source_file.read_text())
        if (
            reviewed.get("status") not in allowed
            or reviewed.get("producer_hash") != source_hash
            or reviewed.get("hardware_authorized") is not False
            or len(reviewed.get("records", [])) != 32
            or len(report.get("records", [])) != 32
        ):
            raise ValueError("complete matching native replay required")
        witnesses = {r["index"]: r for r in reviewed["records"]}
        if len(witnesses) != 32 or len({r["index"] for r in report["records"]}) != 32:
            raise ValueError("duplicate replay course")
        bases = [
            Path(path)
            for path in report["input_file_hashes"]
            if path.endswith("/cached-standard/declared-scene.xml")
        ]
        if (
            len(bases) != 1
            or digest(bases[0]) != report["input_file_hashes"][str(bases[0])]
            or (source / "scene.xml").read_text()
            != shared_native_scene(bases[0].read_text(), prefixes=("a/", "b/"))
        ):
            raise ValueError("source scene differs from audited pinned model")
        model = mujoco.MjModel.from_xml_path(str(source / "scene.xml"))
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "soccer_ball_joint")
        if joint < 0:
            raise ValueError("missing ball joint")
        qadr = int(model.jnt_qposadr[joint])
        dt = float(model.opt.timestep)
        provenance.append(
            {
                "source_hash": source_hash,
                "audit_hash": expected_hash,
                "scene_hash": digest(source / "scene.xml"),
                "native_timestep_s": dt,
            }
        )
        for record in report["records"]:
            witness = witnesses.get(record["index"])
            artifact = source / record["artifact"]["path"]
            if (
                witness is None
                or witness["course"] != record["course"]
                or witness["qualified_capture"] != record["qualified_capture"]
                or artifact.resolve().parent != source.resolve()
                or digest(artifact) != record["artifact"]["hash"]
            ):
                raise ValueError("source and replay or trajectory integrity mismatch")
            with np.load(artifact, allow_pickle=False) as data:
                timing = contact_timing(data["ball_forces"], dt)
                index = timing["first_nonfoot_substep"]
                qpos = data["qpos"].reshape(-1, model.nq)
                height = float(qpos[max(0, index - 1), qadr + 2]) if index is not None else None
            results.append(
                {
                    "course": record["course"],
                    "qualified_capture": record["qualified_capture"],
                    **timing,
                    "ball_height_before_nonfoot_m": height,
                }
            )
    groups = defaultdict(list)
    for row in results:
        groups[row["course"]["arm"]].append(row)
    summary = {}
    for arm, rows in groups.items():
        delays = [r["delay_s"] for r in rows if r["delay_s"] is not None]
        heights = [
            r["ball_height_before_nonfoot_m"]
            for r in rows
            if r["ball_height_before_nonfoot_m"] is not None
        ]
        summary[arm] = {
            "worlds": len(rows),
            "qualified_captures": sum(r["qualified_capture"] for r in rows),
            "nonfoot_worlds": len(heights),
            "foot_first_then_nonfoot": len(delays),
            "delay_median_s": float(np.median(delays)) if delays else None,
            "delay_p90_s": float(np.quantile(delays, 0.9)) if delays else None,
            "ball_height_median_m": float(np.median(heights)) if heights else None,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        {
            "status": "DEVELOPMENT_DIAGNOSTIC_ONLY",
            "source_head": args.expected_head,
            "hardware_authorized": False,
            "promotion_authorized": False,
            "provenance": provenance,
            "summary": summary,
            "rows": results,
        },
    )


if __name__ == "__main__":
    main()
