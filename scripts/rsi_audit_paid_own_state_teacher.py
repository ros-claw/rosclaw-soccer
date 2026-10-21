"""Independent own-student NN, own-R2 planner, IK and all-step SIM handoff audit."""

import argparse
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_probe_paid_own_state_teacher import load_sources

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen independent audit source required")

    clean()
    if a.output.exists():
        raise ValueError("unused audit output required")
    deadline = time.monotonic() + 7200
    while not (a.root / "complete.json").exists():
        clean()
        if (a.root / "failed.json").exists():
            raise ValueError("upstream failed, no automatic retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("closed teacher pilot required")
        time.sleep(10)
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    r, d = [json.loads(f.read_text()) for f in (rp, dp)]
    if (
        r["status"] != "PAID_OWN_STATE_TEACHER_HANDOFF_CLOSED_PENDING_AUDIT"
        or r["declaration_hash"] != digest(dp)
        or r["source_head"] != a.producer_head
        or d["source_head"] != a.producer_head
        or r["config"] != d["config"]
        or r["episodes"] != 20
        or r["optimizer_updates"] != 0
        or r["Fresh_rows_read"] != 0
        or any(
            o[k] is not False
            for o in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("exact complete SIM-only handoff pilot required")
    roots, sources, cases, pins = load_sources(SimpleNamespace(**d["source_arguments"]))
    for path, h in r["input_file_hashes"].items():
        if path in pins and pins[path] != h:
            raise ValueError("conflicting handoff lineage")
        pins[path] = h
    pins[str(rp)], pins[str(dp)] = digest(rp), digest(dp)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_hash=digest(rp),
            input_file_hashes=pins,
            own_student_neural_recompute=True,
            own_R2_planner_recompute=True,
            own_kinematic_IK_recompute=True,
            own_native_integration=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, own_neural=0, episodes=0, R2_planner=0, IK=0)
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_native_phase_replay import PaidNativePhaseReplay
        from rosclaw_soccer.rsi.paid_neural_stand_audit import StandingObservationAuditor
        from rosclaw_soccer.rsi.paid_own_state_teacher import handoff_controls
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor
        from rosclaw_soccer.rsi.paid_successor_contact_audit import audit_successor_contact

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        auditor, motions = open_paid_auditor(pins, [cases[i] for i in (4, 5)])
        observer, replay = StandingObservationAuditor(auditor), PaidNativePhaseReplay(auditor)
        s = sources["student"]
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(s["checkpoint_path"]),
            checkpoint_hash=s["checkpoint_hash"],
            training_receipt=roots["student"] / "complete.json",
            training_receipt_hash=d["source_arguments"]["student_hash"],
            scene_hash=s["body_scene_hash"],
        )
        fixed = sources["teacher"]["searches"][0]
        center, latent = fixed["center"], np.array(fixed["latent"], np.float64)
        switches = handoff_controls(center, len(motions[cases[4]["source"]]["q"]))
        expected_config = dict(
            protocol="paid.TRAIN.own_state_teacher_handoff.v1",
            cases=[4, 5],
            arms=switches,
            repeats=2,
            maximum_episodes=20,
            maximum_native_steps=77800,
            student_seed=0,
            teacher_seed=0,
            center=center,
            latent=latent.tolist(),
            new_optimizer_updates=0,
            Fresh_rows_read=0,
            DEV_fit_rows=0,
        )
        if r["config"] != expected_config or [x["arm"] for x in r["arms"]] != list(switches):
            raise ValueError("exact fixed phase boundary design required")
        visited = set()

        def read(link, root=a.root):
            f = root / link["raw_path"]
            if (
                f.is_symlink()
                or not f.resolve().is_relative_to(root.resolve())
                or digest(f) != link["raw_hash"]
            ):
                raise ValueError("exact confined handoff raw required")
            pins[str(f)] = link["raw_hash"]
            if root == a.root:
                visited.add(link["raw_path"])
            with np.load(f, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def inference():
            counts["own_neural"] += 1

        summaries = []
        for arm in r["arms"]:
            name = arm["arm"]
            if arm["switch_control"] != switches[name] or len(arm["repeats"]) != 2:
                raise ValueError("both fixed repeats required")
            outcomes = []
            for repeat, rows in enumerate(arm["repeats"]):
                if [row["case"] for row in rows] != [4, 5]:
                    raise ValueError("both fixed TRAIN cases required")
                results = []
                raws = {}
                for row in rows:
                    i = row["case"]
                    if row["arm"] != name or row["switch_control"] != switches[name]:
                        raise ValueError("own arm identity differs")
                    for key, suffix in (
                        ("initial", "initial"),
                        ("kick", "kick"),
                        ("kick_terminal", "kick-terminal"),
                        ("stand", "stand"),
                    ):
                        if (
                            row[key] is not None
                            and row[key]["raw_path"]
                            != f"{name}-repeat{repeat}/case{i}/{suffix}.npz"
                        ):
                            raise ValueError("own case/phase evidence required")
                    entry = read(row["initial"])
                    source = sources["student_exam"]["worlds"][i]
                    initial_link = next(
                        v
                        for v in source["artifacts"]
                        if Path(v["raw_path"]).stem == "world-initial"
                    )
                    identical(entry, read(initial_link, roots["student_exam"]))
                    result = audit_successor_contact(
                        auditor,
                        observer,
                        replay,
                        actor,
                        row,
                        read=read,
                        entry=entry,
                        previous=np.zeros(29, np.float32),
                        motion=motions[cases[i]["source"]],
                        center=center,
                        latent=latent,
                        pure=False,
                        count_inference=inference,
                        handoff_control=switches[name],
                    )
                    counts["native"] += result["native"]
                    counts["neural"] += result["neural"]
                    counts["episodes"] += 1
                    counts["R2_planner"] += result["own_R2_planner_head_rows"]
                    counts["IK"] += result["own_IK_queries"]
                    healthy = (
                        result["quality"] and result["stand_healthy"] and result["stand_stable"]
                    )
                    if row["quality"] != result["quality"] or row["healthy_success"] != healthy:
                        raise ValueError("independent contact/recovery result differs")
                    result.update(case=i, healthy_success=healthy)
                    results.append(result)
                    kick = read(row["kick"])
                    raws[f"{i}-kick"] = kick
                    if row["stand"] is not None:
                        raws[f"{i}-stand"] = read(row["stand"])
                    if name in ("student", "teacher"):
                        if name == "student":
                            link = next(
                                v
                                for v in source["artifacts"]
                                if Path(v["raw_path"]).stem == "kick0"
                            )
                            old = read(link, roots["student_exam"])
                        else:
                            link = next(v for v in fixed["selected"][0] if v["case"] == i)["kick"]
                            old = read(link, roots["teacher"])
                        own = {
                            k: v
                            for k, v in kick.items()
                            if not k.startswith("PLAN_")
                            and (name == "student" or k != "NN_nominal_action")
                        }
                        identical(own, {k: old[k] for k in own})
                    print(
                        json.dumps(
                            dict(
                                arm=name,
                                repeat=repeat,
                                case=i,
                                status="OWN_NN_PLANNER_IK_NATIVE_CHECKED",
                                healthy_success=healthy,
                            )
                        ),
                        flush=True,
                    )
                if repeat == 0:
                    first_raws = raws
                else:
                    if results != outcomes[0] or set(raws) != set(first_raws):
                        raise ValueError("repeat outcomes/phases differ")
                    for key in raws:
                        identical(raws[key], first_raws[key])
                outcomes.append(results)
            summaries.append(
                dict(
                    arm=name,
                    switch_control=switches[name],
                    cases=outcomes[0],
                    healthy_successes=sum(v["healthy_success"] for v in outcomes[0]),
                )
            )
        if (
            counts["native"] != r["actual_native_steps"]
            or counts["native"] != replay.native_steps
            or counts["neural"] != r["actual_neural_rows"]
            or counts["own_neural"] != counts["neural"]
            or counts["R2_planner"] != r["planner_R2_head_rows"]
            or counts["IK"] != r["IK_queries"]
            or counts["episodes"] != 20
            or any(str(f.relative_to(a.root)) not in visited for f in a.root.rglob("*.npz"))
        ):
            raise ValueError("all own forward/native/episode counts and raw coverage required")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_OWN_STATE_TEACHER_NN_PLANNER_IK_ALL_NATIVE_AUDIT_PASS",
                source_head=a.expected_head,
                source_hash=digest(rp),
                input_file_hashes=pins,
                config=expected_config,
                summary=summaries,
                counts=counts,
                own_native_steps=counts["native"],
                own_actor_neural_rows=counts["own_neural"],
                own_R2_planner_head_rows=counts["R2_planner"],
                unique_source_contexts=2,
                repeats_are_not_extra_contexts=True,
                promotion_authorized=False,
                hardware_authorized=False,
                optimizer_updates=0,
                Fresh_rows_read=0,
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
