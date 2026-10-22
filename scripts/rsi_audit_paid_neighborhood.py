"""Independent ownNN/IK/all-native audit of every fixed contact perturbation."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("producer-head", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen perturbation audit required")

    clean()
    if a.output.exists():
        raise ValueError("unused audit output required, no retry")
    deadline = time.monotonic() + 7200
    while not (a.root / "complete.json").exists():
        clean()
        if (a.root / "failed.json").exists():
            raise ValueError("producer failed, no automatic retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("bounded specific arm wait")
        time.sleep(10)
    from rosclaw_soccer.rsi.paid_neighborhood_sources import (
        load_current_student_prefix,
        load_neighborhood_plan,
    )
    from rosclaw_soccer.rsi.paid_neighborhood_study import current_prefix_config

    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if rp.is_symlink() or dp.is_symlink() or (a.root / "failed.json").exists():
        raise ValueError("ordinary exact closed experiment required")
    r, d = [json.loads(f.read_text()) for f in (rp, dp)]
    _, plan, pins = load_neighborhood_plan(Path(d["plan_root"]), d["plan_hash"])
    arm = r["arm"]
    current = d.get("current_student_prefix", False)
    if type(current) is not bool or current and arm != "fixed_successful_teacher":
        raise ValueError("typed explicit current-prefix teacher mode required")
    config = current_prefix_config(d["handoff_control"]) if current else plan["config"]
    prefix = None
    if current:
        prefix, extra = load_current_student_prefix(
            Path(d["prefix_root"]),
            d["prefix_hash"],
            Path(d["prefix_audit_root"]),
            d["prefix_audit_hash"],
            plan,
            d["plan_hash"],
        )
        for f, h in extra.items():
            if f in pins and pins[f] != h:
                raise ValueError("conflicting prefix identity")
            pins[f] = h
    spec = plan["specifications"]["models"]["learned_memory" if current else arm]
    candidate = "candidate_root" in d or "candidate_hash" in d
    if candidate:
        if (
            current
            or arm != "learned_memory"
            or "candidate_root" not in d
            or "candidate_hash" not in d
        ):
            raise ValueError("typed growing learned arm without teacher mode required")
        from rosclaw_soccer.rsi.paid_candidate_dispatch import contact_exam_candidate

        spec, config, extra = contact_exam_candidate(
            Path(d["candidate_root"]), d["candidate_hash"], plan
        )
        for f, h in extra.items():
            if f in pins and pins[f] != h:
                raise ValueError("conflicting candidate identity")
            pins[f] = h
    if (
        r["status"] != "PAID_CONTACT_NEIGHBORHOOD32_ARM_CLOSED_PENDING_AUDIT"
        or r["declaration_hash"] != digest(dp)
        or d.get("base_stand") is not True
        or d.get("teacher_reoptimized") is not False
        or any(
            v["source_head"] != a.producer_head
            or v["config"] != config
            or v["model_hash"] != spec["hash"]
            or v["body_scene_hash"] != plan["body_scene_hash"]
            for v in (r, d)
        )
        or any(
            r.get(k) != v for k, v in d.items() if k not in ("base_stand", "teacher_reoptimized")
        )
        or any(
            v.get(k) is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
        or r["episodes"] != 64
        or r["repeats_exact"] is not True
        or r["optimizer_updates"] != 0
        or r["Fresh_rows_read"] != 0
        or [(v["case"], v["repeat"]) for v in r["records"]]
        != [(i, j) for i in range(32) for j in range(2)]
    ):
        raise ValueError("complete committed all-context perturbation arm required")
    for path, h in r["input_file_hashes"].items():
        if path in pins and pins[path] != h:
            raise ValueError("conflicting perturbation provenance")
        pins[path] = h
    pins[str(rp)], pins[str(dp)] = digest(rp), digest(dp)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    counts = dict(episodes=0, native=0, neural=0, own_neural=0, IK=0, R2_planner=0)
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_native_phase_replay import PaidNativePhaseReplay
        from rosclaw_soccer.rsi.paid_neural_stand_audit import StandingObservationAuditor
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor
        from rosclaw_soccer.rsi.paid_successor_contact_audit import audit_successor_contact

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        auditor, motions = open_paid_auditor(pins, plan["cases"])
        observer, replay = StandingObservationAuditor(auditor), PaidNativePhaseReplay(auditor)
        srp = Path(spec["root"]) / "complete.json"
        s = json.loads(srp.read_text())
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(s["checkpoint_path"]),
            checkpoint_hash=s["checkpoint_hash"],
            training_receipt=srp,
            training_receipt_hash=spec["hash"],
            scene_hash=plan["body_scene_hash"],
        )
        write_once(
            a.output / "declaration.json",
            dict(
                source_head=a.expected_head,
                source_hash=digest(rp),
                producer_head=a.producer_head,
                arm=arm,
                plan_hash=d["plan_hash"],
                input_file_hashes=pins,
                base_stand=True,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        visited = set()

        def read(link):
            path = a.root / link["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(a.root.resolve())
                or digest(path) != link["raw_hash"]
            ):
                raise ValueError("confined exact raw phase required")
            visited.add(str(path))
            pins[str(path)] = link["raw_hash"]
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def inference():
            counts["own_neural"] += 1

        outcomes = []
        for row in r["records"]:
            case = plan["cases"][row["case"]]
            if row["source_case"] != case["source_case"]:
                raise ValueError("source context identity differs")
            target = plan["specifications"]["teachers"][str(case["source_case"])]
            teacher = arm == "fixed_successful_teacher"
            entry = read(row["initial"])
            with np.load(
                Path(d["plan_root"]) / f"case{case['case']}-initial.npz", allow_pickle=False
            ) as z:
                identical(entry, {k: z[k].copy() for k in z.files})
            result = audit_successor_contact(
                auditor,
                observer,
                replay,
                actor,
                row,
                read=read,
                entry=entry,
                previous=np.zeros(29, np.float32),
                motion=motions[case["source"]],
                center=target["center"],
                latent=np.array(target["latent"], np.float64),
                pure=not teacher,
                count_inference=inference,
                handoff_control=(d["handoff_control"] if current else target["switch_control"])
                if teacher
                else None,
                base_stand=True,
            )
            if prefix is not None:
                raw = read(row["kick"])
                link = prefix["records"][2 * row["case"] + row["repeat"]]["kick"]
                with np.load(Path(d["prefix_root"]) / link["raw_path"], allow_pickle=False) as z:
                    for k in (
                        "NN_obs",
                        "NN_action",
                        "NN_h_before",
                        "NN_c_before",
                        "NN_h_after",
                        "NN_c_after",
                    ):
                        if not np.array_equal(
                            raw[k][: d["handoff_control"]], z[k][: d["handoff_control"]]
                        ):
                            raise ValueError(
                                "independent current-student prefix/private memory differs"
                            )
            healthy = bool(result["quality"] and result["stand_healthy"] and result["stand_stable"])
            if row["quality"] != result["quality"] or row["healthy_success"] != healthy:
                raise ValueError("independently measured task outcome differs")
            for k in ("native", "neural"):
                counts[k] += result[k]
            counts["IK"] += result.get("own_IK_queries", 0)
            counts["R2_planner"] += result.get("own_R2_planner_head_rows", 0)
            counts["episodes"] += 1
            outcomes.append(
                dict(case=row["case"], repeat=row["repeat"], healthy_success=healthy, **result)
            )
            if row["repeat"] == 1:
                first = r["records"][2 * row["case"]]
                for key in ("initial", "kick", "kick_terminal", "stand"):
                    if (first[key] is None) != (row[key] is None):
                        raise ValueError("whole repeated branch differs")
                    if first[key] is not None:
                        identical(read(first[key]), read(row[key]))
                print(
                    json.dumps(dict(case=row["case"], arm=arm, status="OWN_NN_IK_NATIVE_CHECKED")),
                    flush=True,
                )
        if (
            counts["native"] != r["actual_native_steps"]
            or replay.native_steps != counts["native"]
            or counts["neural"] != r["actual_neural_rows"]
            or counts["own_neural"] != counts["neural"]
            or any(str(f) not in visited for f in a.root.rglob("*.npz"))
        ):
            raise ValueError("all declared physical and neural work must reconcile")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CONTACT_NEIGHBORHOOD32_OWN_NN_IK_ALL_NATIVE_AUDIT_PASS",
                source_head=a.expected_head,
                source_hash=digest(rp),
                arm=arm,
                plan_hash=d["plan_hash"],
                input_file_hashes=pins,
                counts=counts,
                outcomes=outcomes,
                own_native_steps=replay.native_steps,
                own_trajectory_neural_rows=counts["own_neural"],
                repeats_exact=True,
                optimizer_updates=0,
                Fresh_rows_read=0,
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
                counts=counts,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
