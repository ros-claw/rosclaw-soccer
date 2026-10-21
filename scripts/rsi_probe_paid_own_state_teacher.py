"""Finite TRAIN teacher recovery from actual student prefixes, not a deployed policy."""

import argparse
import json
import subprocess
from pathlib import Path


def load_sources(a):
    """Authenticate the already closed teacher and student evidence before numerics."""
    from scripts.rsi_native_context_recovery_common import digest

    roots = {
        name: Path(getattr(a, name + "_root"))
        for name in ("teacher", "teacher_audit", "student", "student_exam", "student_audit")
    }
    reports, pins = {}, {}
    for name, root in roots.items():
        rp = root / "complete.json"
        if (
            rp.is_symlink()
            or (root / "failed.json").exists()
            or digest(rp) != getattr(a, name + "_hash")
        ):
            raise ValueError("exact closed teacher/student witnesses required")
        r = json.loads(rp.read_text())
        if any(r[k] is not False for k in ("hardware_authorized", "promotion_authorized")):
            raise ValueError("SIM-only source evidence required")
        for path, h in r["input_file_hashes"].items():
            if path in pins and pins[path] != h:
                raise ValueError("conflicting source lineage")
            pins[path] = h
        pins[str(rp)] = digest(rp)
        reports[name] = r
    t, ta, s, e, ea = [
        reports[k] for k in ("teacher", "teacher_audit", "student", "student_exam", "student_audit")
    ]
    td = roots["teacher"] / "declaration.json"
    ed = roots["student_exam"] / "declaration.json"
    teacher_decl, exam_decl = [json.loads(p.read_text()) for p in (td, ed)]
    interventional = getattr(a, "interventional", False)
    if type(interventional) is not bool:
        raise ValueError("explicit typed student generation required")
    student_status = (
        "PAID_INTERVENTIONAL_DAGGER_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED"
        if interventional
        else "PAID_ACTIVE_COMPACT_SUPPORT_AUDITED_NOT_PHYSICALLY_QUALIFIED"
    )
    if (
        t["status"] != "PAID_TRAIN_FIRST_CONTACT_TEACHER_CLOSED_PENDING_OWN_AUDIT"
        or ta["status"] != "PAID_FIRST_CONTACT_ALL_TRIAL_OWN_NN_IK_AND_NATIVE_AUDIT_PASS"
        or t["config"]["seed_index"] != 0
        or t["episodes"] != 108
        or ta["source_hash"] != a.teacher_hash
        or ta["counts"]["episodes"] != 108
        or ta["own_native_steps"] != t["actual_native_steps"]
        or ta["own_trajectory_neural_rows"] != t["actual_neural_rows"]
        or t["bank"] != "train"
        or t["Fresh_rows_read"] != 0
        or s["status"] != student_status
        or interventional
        and s["config"].get("protocol") != "interventional_dagger.warm_contact.v1"
        or s["config"]["seed_index"] != 0
        or e["status"] != "PAID_ROUTER_TRAIN20_WORLDS_CLOSED_PENDING_AUDIT"
        or ea["status"] != "PAID_ROUTER_TRAIN20_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS"
        or ea["source_hash"] != a.student_exam_hash
        or ea["student_hash"] != a.student_hash
        or e["student_hash"] != a.student_hash
        or ea["counts"]["worlds"] != 40
        or ea["own_native_steps"] != e["actual_native_steps"]
        or ea["own_trajectory_neural_rows"] != e["actual_neural_rows"]
        or e["bank"] != "train"
        or exam_decl.get("first_enabled") is not True
        or exam_decl["schedule"] != ["student"] * 3
        or digest(td) != t.get("declaration_hash", digest(td))
        or digest(ed) != e["declaration_hash"]
        or t["searches"][0]["pair"] != [4, 5]
        or t["searches"][0]["selected_rank"][:6] != [1, 1, 1, 1, 2, 2]
    ):
        raise ValueError("fully audited fixed seed0 TRAIN teacher and failed student required")
    # The original teacher search did not store declaration_hash in its receipt;
    # its independent audit binds that file through input_file_hashes.
    if ta["input_file_hashes"].get(str(td)) != digest(td):
        raise ValueError("original teacher declaration is not independently bound")
    pins[str(td)], pins[str(ed)] = digest(td), digest(ed)
    cases = teacher_decl["source_cases"]
    if [c["case"] for c in cases] != list(range(20)) or len(e["worlds"]) != 20:
        raise ValueError("original complete TRAIN20 case identities required")
    return roots, reports, cases, pins


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "teacher-audit", "student", "student-exam", "student-audit"):
        p.add_argument("--" + name + "-root", type=Path, required=True)
        p.add_argument("--" + name + "-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused pilot and frozen clean source required")
    roots, reports, cases, pins = load_sources(a)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    bench = None
    episodes = r2_head_queries = ik_queries = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
        from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
        from rosclaw_soccer.rsi.paid_own_state_teacher import (
            OwnStateTeacherHandoff,
            handoff_controls,
        )
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        fixed = reports["teacher"]["searches"][0]
        centre, latent = fixed["center"], np.array(fixed["latent"], np.float64)
        bench, policy, scene = open_paid_bench(pins, [cases[i] for i in (4, 5)])
        s = reports["student"]
        if s["body_scene_hash"] != scene:
            raise ValueError("unchanged body/ball/PD scene required")
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(s["checkpoint_path"]),
            checkpoint_hash=s["checkpoint_hash"],
            training_receipt=roots["student"] / "complete.json",
            training_receipt_hash=a.student_hash,
            scene_hash=scene,
        )
        bounds = [handoff_controls(centre, bench.clips[cases[i]["source"]].length) for i in (4, 5)]
        if bounds[0] != bounds[1]:
            raise ValueError("paired actual horizons must agree")
        config = dict(
            protocol="paid.TRAIN.own_state_teacher_handoff.v1",
            cases=[4, 5],
            arms=bounds[0],
            repeats=2,
            maximum_episodes=20,
            maximum_native_steps=77800,
            student_seed=0,
            teacher_seed=0,
            center=centre,
            latent=latent.tolist(),
            new_optimizer_updates=0,
            Fresh_rows_read=0,
            DEV_fit_rows=0,
        )
        write_once(
            a.output / "declaration.json",
            dict(
                source_head=a.expected_head,
                config=config,
                source_arguments={
                    k: str(v) for k, v in vars(a).items() if k not in ("output", "expected_head")
                },
                input_file_hashes=pins,
                body_scene_hash=scene,
                cases=[cases[i] for i in (4, 5)],
                policy_mode="explicit_composite_teacher_not_student_deployment",
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )

        def read(root, link):
            f = root / link["raw_path"]
            if (
                f.is_symlink()
                or not f.resolve().is_relative_to(root.resolve())
                or pins.get(str(f)) != link["raw_hash"]
                or digest(f) != link["raw_hash"]
            ):
                raise ValueError("own-audited confined reference raw required")
            with np.load(f, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def store(path, z):
            if path.exists():
                raise ValueError("raw overwrite forbidden")
            np.savez_compressed(path, **z)
            return dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))

        def inference():
            with bench.lock:
                bench.neural_rows += 1

        arms = []
        for arm, control in config["arms"].items():
            replicas = []
            for repeat in range(2):
                rows, blobs = [], {}
                for i in (4, 5):
                    folder = a.output / f"{arm}-repeat{repeat}" / f"case{i}"
                    folder.mkdir(parents=True)
                    ep = bench.make(cases[i])
                    initial = snapshot(ep.env.data)
                    source_world = reports["student_exam"]["worlds"][i]
                    initial_link = next(
                        v
                        for v in source_world["artifacts"]
                        if Path(v["raw_path"]).stem == "world-initial"
                    )
                    identical(initial, read(roots["student_exam"], initial_link))
                    intent = FootTaskSpaceIntent(
                        ep.env.model,
                        foot="left" if "left" in cases[i]["source"] else "right",
                        center=centre,
                        latent=latent,
                    )
                    teacher = OwnStateTeacherHandoff(
                        ep, actor.base, intent=intent, switch_control=control
                    )
                    raw, row = bench.collect_current(
                        ep, actor, causal_transform=teacher, stop_at_reference=True
                    )
                    r2_head_queries += teacher.r2_head_queries
                    ik_queries += teacher.ik_queries
                    record = dict(
                        case=i,
                        arm=arm,
                        switch_control=control,
                        initial=store(folder / "initial.npz", initial),
                        kick=dict(row, **store(folder / "kick.npz", raw)),
                        kick_terminal=store(folder / "kick-terminal.npz", snapshot(ep.env.data)),
                        stand=None,
                    )
                    if not row["body_failed"]:
                        task = NeuralStandingTask(
                            ep,
                            actor.base,
                            state=tuple(
                                torch.from_numpy(raw[k][-1].copy())
                                for k in ("NN_h_after", "NN_c_after")
                            ),
                            memory_mode="carry",
                            count_inference=inference,
                        )
                        sr, ss = task.collect(100)
                        record["stand"] = dict(ss, **store(folder / "stand.npz", sr))
                        blobs[f"case{i}-stand"] = sr
                    if arm in ("student", "teacher"):
                        if arm == "student":
                            reference = next(
                                v
                                for v in source_world["artifacts"]
                                if Path(v["raw_path"]).stem == "kick0"
                            )
                            old = read(roots["student_exam"], reference)
                        else:
                            reference = next(v for v in fixed["selected"][0] if v["case"] == i)[
                                "kick"
                            ]
                            old = read(roots["teacher"], reference)
                        own = {
                            k: v
                            for k, v in raw.items()
                            if not k.startswith("PLAN_")
                            and (arm == "student" or k != "NN_nominal_action")
                        }
                        identical(own, {k: old[k] for k in own})
                    record["quality"] = bool(contact_rank([row, row])[1] > 0)
                    record["healthy_success"] = bool(
                        record["quality"]
                        and record["stand"]
                        and record["stand"]["healthy"]
                        and record["stand"]["stability_observed"]
                    )
                    rows.append(record)
                    blobs[f"case{i}-kick"] = raw
                    episodes += 1
                    if episodes > 20 or bench.native_steps > 77800:
                        raise ValueError("declared pilot budget exceeded")
                    print(
                        json.dumps(
                            dict(
                                arm=arm,
                                repeat=repeat,
                                case=i,
                                quality=record["quality"],
                                healthy_success=record["healthy_success"],
                            )
                        ),
                        flush=True,
                    )
                if repeat == 0:
                    first_blobs = blobs
                else:
                    if set(blobs) != set(first_blobs):
                        raise ValueError("repeat phase coverage differs")
                    for k in blobs:
                        identical(blobs[k], first_blobs[k])
                replicas.append(rows)
            arms.append(dict(arm=arm, switch_control=control, repeats=replicas))
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_OWN_STATE_TEACHER_HANDOFF_CLOSED_PENDING_AUDIT",
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                config=config,
                input_file_hashes=pins,
                arms=arms,
                episodes=episodes,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                planner_R2_head_rows=r2_head_queries,
                IK_queries=ik_queries,
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
                episodes=episodes,
                actual_native_steps=bench.native_steps if bench else 0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
