"""Audit-bound TRAIN-only own-policy support/contact diagnosis; no fitting or new physics."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for k in ("exam-root", "audit-root", "data-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("data-hash", "producer-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen diagnostic source required")

    clean()
    if a.output.exists():
        raise ValueError("unused diagnostic output required")
    deadline = time.monotonic() + 7200
    while not all((root / "complete.json").exists() for root in (a.exam_root, a.audit_root)):
        clean()
        if any((root / "failed.json").exists() for root in (a.exam_root, a.audit_root)):
            raise ValueError("failed physics/audit contract: preserve and diagnose, never retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("specific first-enabled exam and own audit required")
        time.sleep(10)
    rp, ap, dp = (
        a.exam_root / "complete.json",
        a.audit_root / "complete.json",
        a.exam_root / "declaration.json",
    )
    r, ar, d = [json.loads(f.read_text()) for f in (rp, ap, dp)]
    if (
        r["status"] != "PAID_ROUTER_TRAIN20_WORLDS_CLOSED_PENDING_AUDIT"
        or ar["status"] != "PAID_ROUTER_TRAIN20_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS"
        or r["bank"] != "train"
        or r["declaration_hash"] != digest(dp)
        or ar["source_hash"] != digest(rp)
        or ar["student_hash"] != r["student_hash"]
        or d["student_hash"] != r["student_hash"]
        or d.get("first_enabled") is not True
        or d["schedule"] != ["student"] * 3
        or ar["summary"] != r["summary"]
        or [w["case"] for w in r["worlds"]] != list(range(20))
        or ar["counts"]["worlds"] != 40
        or ar["first_kick_replayed"] is not True
        or ar["own_native_steps"] != r["actual_native_steps"]
        or ar["own_trajectory_neural_rows"] != r["actual_neural_rows"]
        or any(v["source_head"] != a.producer_head for v in (r, ar, d))
        or any(
            v[k] is not False
            for v in (r, ar, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("full same-model TRAIN first-enabled ownNN/native audit required")
    data_path = a.data_root / "complete.json"
    if digest(data_path) != a.data_hash:
        raise ValueError("exact original TRAIN curation required")
    data = json.loads(data_path.read_text())
    sp = Path(d["student_root"]) / "complete.json"
    if digest(sp) != d["student_hash"]:
        raise ValueError("same recorded student identity required")
    student = json.loads(sp.read_text())
    if student["data_hash"] != a.data_hash or student["status"] not in (
        "PAID_COMPACT_SUPPORT_BATCH1_AUDITED_NOT_PHYSICALLY_QUALIFIED",
        "PAID_ACTIVE_COMPACT_SUPPORT_AUDITED_NOT_PHYSICALLY_QUALIFIED",
    ):
        raise ValueError("student and diagnostic TRAIN curriculum differ")
    pins = dict(ar["input_file_hashes"])
    for f in (rp, ap, dp, sp, data_path):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            exam_root=str(a.exam_root),
            exam_hash=digest(rp),
            audit_hash=digest(ap),
            data_hash=a.data_hash,
            student_hash=digest(sp),
            cases=list(range(20)),
            selected_by_outcome=False,
            Fresh_rows_read=0,
            physical_replay_performed=False,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_compact_diagnosis import (
            compact_own_trace,
            summarize_compact_trace,
        )
        from rosclaw_soccer.rsi.paid_contact_diagnosis import contact_failure_causes
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(student["checkpoint_path"]),
            checkpoint_hash=student["checkpoint_hash"],
            training_receipt=sp,
            training_receipt_hash=digest(sp),
            scene_hash=student["body_scene_hash"],
        )
        f = Path(data["feature_path"])
        if f.is_symlink() or digest(f) != data["feature_hash"]:
            raise ValueError("own-causal TRAIN archive commitment changed")
        pins[str(f)] = data["feature_hash"]
        with np.load(f, allow_pickle=False) as archive:
            z = {k: archive[k].copy() for k in archive.files}
        teachers = {t["case"]: t for t in data["trajectories"] if "teacher" in t["role"]}
        rows, cases = 0, []
        for world in r["worlds"]:
            link = next(v for v in world["artifacts"] if Path(v["raw_path"]).stem == "kick0")
            raw_path = a.exam_root / link["raw_path"]
            if (
                raw_path.is_symlink()
                or not raw_path.resolve().is_relative_to(a.exam_root.resolve())
                or digest(raw_path) != link["raw_hash"]
                or pins.get(str(raw_path)) != link["raw_hash"]
            ):
                raise ValueError("exact independently audited own first-kick trace required")
            with np.load(raw_path, allow_pickle=False) as archive:
                raw = {k: archive[k].copy() for k in archive.files}
            trace = compact_own_trace(actor, raw)
            rows += len(trace["features"])
            kwargs = {}
            if world["case"] in teachers:
                t = teachers[world["case"]]
                sl = slice(t["start"], t["stop"])
                kwargs = dict(
                    teacher_features=z["features"][sl],
                    teacher_delta=z["parent_delta"][sl] + z["target"][sl],
                )
            post = world["kicks"][0]["result"]
            summary = summarize_compact_trace(
                trace,
                **kwargs,
                first_foot_substep=post["postcontact"]["first_foot_contact_substep"],
            )
            trace_path = a.output / f"case{world['case']}-own-support.npz"
            np.savez_compressed(trace_path, **trace)
            cases.append(
                dict(
                    case=world["case"],
                    quality=world["quality_by_kick"][0],
                    failure_causes=contact_failure_causes(post),
                    **summary,
                    trace_path=str(trace_path),
                    trace_hash=digest(trace_path),
                )
            )
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_TRAIN_COMPACT_OWN_POLICY_SUPPORT_AND_CONTACT_DIAGNOSIS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                cases=cases,
                own_recomputed_neural_rows=rows,
                own_native_steps=0,
                optimizer_updates=0,
                Fresh_rows_read=0,
                teacher_comparisons_are_not_own_state_expert_labels=True,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
