"""Once-only original teacher parity plus counterfactual recorded-body IK queries.

All sixteen closed source bodies retained. Kinematics/Jacobians only: no step,
Foundation, optimizer, teacher qualification, admitted dataset or activation.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.contact_reference_labels import reference_labels
from rosclaw_soccer.rsi.recorded_body_reference_query import query_recorded_body
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def query_controls(receipts):
    """Use each original paired teacher on BOTH recorded bodies, without executing it."""
    if len(receipts) != 16:
        raise ValueError("eight whole original control/teacher pairs required")
    result = []
    for i in range(0, 16, 2):
        control, teacher = receipts[i : i + 2]
        if (
            control["role"] != "control"
            or teacher["role"] != "teacher"
            or control["candidate"] is not None
            or control["group"] != teacher["group"]
            or teacher["candidate"]["swing"] != "bounded_contact_diagnostic"
        ):
            raise ValueError("original ordered None/body-origin teacher pairs required")
        option = BoundedFirstTouchControl(**teacher["candidate"]["parameters"])
        result.extend([option, option])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        cfg.get("profile") != "recorded-body-expert-cached-kinematics-parity-v1"
        or cfg.get("mode") != "SIM_ONLY"
        or root.exists()
        or time.time() >= cfg["deadline_unix"]
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "dataset_admission_authorized",
                "neural_training_authorized",
                "private_fresh_accessed",
            )
        )
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_episodes", 16),
                ("new_physical_steps", 0),
                ("optimizer_updates", 0),
            )
        )
        or Path(__file__).resolve() != source / "scripts/rsi_verify_recorded_body_expert.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("fixed frozen once-only zero-step recorded-body diagnostic required")
    require_terminal(cfg["original_fit_unit"])
    pins = {str(args.config): digest(args.config)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path, key="report_hash"):
        path = Path(path)
        v = json.loads(path.read_text())
        if v.get(key) != hash_json({k: x for k, x in v.items() if k != key}):
            raise ValueError("unchanged complete sealed source evidence required")
        pins[str(path)] = digest(path)
        return v

    fit = sealed(cfg["original_fit_complete"])
    if fit["report_hash"] != cfg["original_fit_complete_hash"]:
        raise ValueError("original closed complete physical-source fit stage required")
    for name, h in fit["input_file_hashes"].items():
        if digest(name) != h:
            raise ValueError("original whole physical-source evidence changed")
        pins[name] = h
    pc = json.loads(Path(fit["configuration"]["positive_configuration"]).read_text())
    cc = json.loads(Path(fit["configuration"]["positive_contact_configuration"]).read_text())
    complete = sealed(Path(pc["output_root"]) / "complete.json")
    replay = sealed(Path(cc["replay_output_root"]) / "complete.json")
    contact = sealed(cc["complete_path"])
    if (
        complete["report_hash"] != fit["positive_complete_hash"]
        or contact["report_hash"] != fit["contact_complete_hash"]
        or contact["replay_complete_hash"] != replay["report_hash"]
        or len(complete["results"]) != 16
        or len(replay["rows"]) != 16
        or replay["integrated_substeps"] != 48000
        or [r["receipt_hash"] for r in complete["results"]] != fit["source_receipt_hashes"]
    ):
        raise ValueError(
            "all original sixteen closed physics and dense endpoint memberships required"
        )
    traces, dense, receipts = [], [], complete["results"]
    controls = query_controls(receipts)
    for i, (receipt, row) in enumerate(zip(receipts, replay["rows"], strict=True)):
        r = sealed(Path(pc["output_root"]) / f"receipt-{i:04d}.json", "receipt_hash")
        if r != receipt or row["source_receipt_hash"] != receipt["receipt_hash"]:
            raise ValueError("whole ordered unchanged source receipts and replay required")
        tw, dw, world = receipt["witnesses"]["trace"], row["arrays"], receipt["witnesses"]["world"]
        for w, owner in (
            (tw, pc["output_root"]),
            (world, pc["output_root"]),
            (dw, cc["replay_output_root"]),
        ):
            path = Path(w["path"])
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(Path(owner).resolve())
                or digest(path) != w["sha256"]
            ):
                raise ValueError("ordinary original owned trace, substeps and world required")
            pins[str(path)] = w["sha256"]
        if world["sha256"] != pc["mesh_task_contact_world_hash"]:
            raise ValueError("same original compiled physical world required")
        with np.load(tw["path"], allow_pickle=False) as a:
            traces.append(dict(a))
        with np.load(dw["path"], allow_pickle=False) as a:
            dense.append(dict(a))
    if args.preflight:
        print("SIXTEEN_ORIGINAL_EXPERT_QUERIES_READY_ZERO_MJDATA_OR_DYNAMICS", flush=True)
        return
    root.mkdir()
    write_once(root / "started.json", dict(configuration=cfg, automatic_retry_authorized=False))
    try:
        model = mujoco.MjModel.from_binary_path(receipts[0]["witnesses"]["world"]["path"])
        rows, arrays, teacher_count = [], {}, 0
        for i, (receipt, trace, substeps) in enumerate(zip(receipts, traces, dense, strict=True)):
            query = query_recorded_body(trace, substeps, model, control=controls[i])
            label = query["proposed_reference_increment_rad"]
            actual = reference_labels(trace)["pre_parent_reference_increment_rad"][:, 0]
            parity = label.tobytes() == actual.tobytes()
            if receipt["role"] == "teacher":
                teacher_count += 1
                if not parity:
                    raise ValueError(
                        f"original expert exact teacher labels differ on endpoint {i}: "
                        f"max={np.max(np.abs(label - actual))}"
                    )
            arrays[f"query_reference_rad_{i:02d}"] = label
            rows.append(
                dict(
                    receipt_hash=receipt["receipt_hash"],
                    group=receipt["group"],
                    role=receipt["role"],
                    original_outcome=receipt["outcome"],
                    original_fit_eligible=fit["eligibility"][i],
                    actual_teacher_labels_bit_exact=parity,
                    maximum_reference_difference_from_actual_rad=float(
                        np.max(np.abs(label - actual))
                    ),
                    query=query["diagnostic"],
                )
            )
            print("RECORDED_BODY_EXPERT_QUERY", i, receipt["role"], "parity", parity, flush=True)
        if teacher_count != 8:
            raise ValueError("all eight original teacher episodes including failures required")
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("source or original evidence changed during expert query")
        path = root / "diagnostic-proposals-not-admitted.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report = dict(
            schema="soccer.rsi.recorded_body_expert_query_complete.v1",
            configuration=cfg,
            input_file_hashes=pins,
            rows=rows,
            proposed_arrays=dict(path=str(path), sha256=digest(path)),
            original_eight_teachers_bit_exact=True,
            all_source_failures_retained=True,
            source_episodes=16,
            source_clusters=4,
            kinematic_queries=4800,
            new_physical_steps=0,
            new_foundation_inferences=0,
            optimizer_updates=0,
            own_student_body_states_queried=False,
            dagger_training_performed=False,
            teacher_qualified=False,
            dataset_admission_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        report["report_hash"] = hash_json(report)
        write_once(root / "complete.json", report)
        print("ORIGINAL_EXPERT_SAME_BODY_QUERY_PARITY_COMPLETE", report["report_hash"], flush=True)
    except BaseException as error:
        write_once(
            root / "failure.json",
            dict(type=type(error).__name__, message=str(error), automatic_retry_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
