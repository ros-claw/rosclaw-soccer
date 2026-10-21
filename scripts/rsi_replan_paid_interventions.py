"""Bounded CEM teacher recovery on CURRENT primary student prefixes; TRAIN only."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_probe_paid_own_state_teacher import load_sources

    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "teacher",
        "teacher-audit",
        "student",
        "student-exam",
        "student-audit",
        "diagnosis",
    ):
        p.add_argument("--" + name + "-root", type=Path, required=True)
        p.add_argument("--" + name + "-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen interventional replanner required")

    clean()
    if a.output.exists():
        raise ValueError("unused replan output required")
    a.interventional = True
    roots, sources, cases, pins = load_sources(a)
    diag_path = a.diagnosis_root / "complete.json"
    if diag_path.is_symlink() or digest(diag_path) != a.diagnosis_hash:
        raise ValueError("exact current own-policy diagnosis required")
    diagnosis = json.loads(diag_path.read_text())
    if (
        (a.diagnosis_root / "failed.json").exists()
        or diagnosis["status"] != "PAID_TRAIN_COMPACT_OWN_POLICY_SUPPORT_AND_CONTACT_DIAGNOSIS"
        or diagnosis["input_file_hashes"].get(str(roots["student"] / "complete.json"))
        != a.student_hash
        or [c["case"] for c in diagnosis["cases"]] != list(range(20))
        or any(sources["student_exam"]["worlds"][i]["quality_by_kick"][0] for i in (4, 5))
        or any(diagnosis[k] is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError("diagnosed current primary hard-first failures required")
    pins[str(diag_path)] = a.diagnosis_hash
    for path, h in diagnosis["input_file_hashes"].items():
        if path in pins and pins[path] != h:
            raise ValueError("conflicting own-policy diagnostic lineage")
        pins[path] = h
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    bench = None
    counts = dict(episodes=0, CEM_updates=0, R2_planner=0, IK=0)
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_first_teacher import first_teacher_rank
        from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
        from rosclaw_soccer.rsi.paid_intervention_replan import (
            elite_update,
            replan_config,
            sample_latents,
        )
        from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
        from rosclaw_soccer.rsi.paid_own_state_teacher import OwnStateTeacherHandoff
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        fixed = sources["teacher"]["searches"][0]
        center, initial_latent = fixed["center"], np.array(fixed["latent"], np.float64)
        bench, policy, scene = open_paid_bench(pins, [cases[i] for i in (4, 5)])
        s = sources["student"]
        if s["body_scene_hash"] != scene:
            raise ValueError("same body, ball, PD and authority required")
        config = replan_config(center, bench.clips[cases[4]["source"]].length)
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(s["checkpoint_path"]),
            checkpoint_hash=s["checkpoint_hash"],
            training_receipt=roots["student"] / "complete.json",
            training_receipt_hash=a.student_hash,
            scene_hash=scene,
        )
        write_once(
            a.output / "declaration.json",
            dict(
                source_head=a.expected_head,
                input_file_hashes=pins,
                config=config,
                source_arguments={
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(a).items()
                    if k not in ("output", "expected_head")
                },
                policy_mode="explicit_replanned_teacher_NOT_learned_student",
                body_scene_hash=scene,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )

        def read_source(i, stem):
            world = sources["student_exam"]["worlds"][i]
            link = next(v for v in world["artifacts"] if Path(v["raw_path"]).stem == stem)
            path = roots["student_exam"] / link["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(roots["student_exam"].resolve())
                or pins.get(str(path)) != link["raw_hash"]
                or digest(path) != link["raw_hash"]
            ):
                raise ValueError("bound independently audited source trajectory required")
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        reference_initial = {i: read_source(i, "world-initial") for i in (4, 5)}
        reference_kick = {i: read_source(i, "kick0") for i in (4, 5)}

        def store(path, z):
            if path.exists():
                raise ValueError("raw overwrite forbidden")
            np.savez_compressed(path, **z)
            return dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))

        def inference():
            with bench.lock:
                bench.neural_rows += 1

        def run_pair(folder, latent, switch):
            rows = []
            for i in (4, 5):
                sub = folder / f"case{i}"
                sub.mkdir(parents=True)
                ep = bench.make(cases[i])
                initial = snapshot(ep.env.data)
                identical(initial, reference_initial[i])
                intent = FootTaskSpaceIntent(
                    ep.env.model,
                    foot="left" if "left" in cases[i]["source"] else "right",
                    center=center,
                    latent=latent,
                )
                teacher = OwnStateTeacherHandoff(
                    ep, actor.base, intent=intent, switch_control=switch
                )
                raw, result = bench.collect_current(
                    ep, actor, causal_transform=teacher, stop_at_reference=True
                )
                for k in (
                    "NN_obs",
                    "NN_action",
                    "NN_h_before",
                    "NN_c_before",
                    "NN_h_after",
                    "NN_c_after",
                ):
                    if not np.array_equal(raw[k][:switch], reference_kick[i][k][:switch]):
                        raise ValueError(
                            "current student's actual prefix or private memory changed"
                        )
                row = dict(
                    case=i,
                    switch_control=switch,
                    latent=latent.tolist(),
                    initial=store(sub / "initial.npz", initial),
                    kick=dict(result, **store(sub / "kick.npz", raw)),
                    kick_terminal=store(sub / "kick-terminal.npz", snapshot(ep.env.data)),
                    stand=None,
                )
                if not result["body_failed"]:
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
                    row["stand"] = dict(ss, **store(sub / "stand.npz", sr))
                row["quality"] = bool(contact_rank([result, result])[1])
                row["healthy_success"] = bool(
                    row["quality"]
                    and row["stand"]
                    and row["stand"]["healthy"]
                    and row["stand"]["stability_observed"]
                )
                counts["episodes"] += 1
                counts["R2_planner"] += teacher.r2_head_queries
                counts["IK"] += teacher.ik_queries
                if (
                    counts["episodes"] > config["max_episodes"]
                    or bench.native_steps > config["max_native_steps"]
                ):
                    raise ValueError("finite own-state replan budget exceeded")
                rows.append(row)
            return rows

        rng = np.random.default_rng(config["seed"])
        searches = []
        for switch in config["switch_controls"]:
            folder = a.output / f"switch{switch}"
            baseline = run_pair(folder / "student-control", initial_latent, config["horizon"])
            original = run_pair(folder / "fixed-teacher-control", initial_latent, switch)
            best = initial_latent.copy()
            rank = first_teacher_rank(original, [False, False])
            mean = np.arctanh(np.clip(best, -0.999999, 0.999999))
            std = np.full((2, 6), 0.7)
            history = []
            for generation in range(config["generations"]):
                unbounded, proposals = sample_latents(rng, mean, std, best)
                records = []
                for slot, latent in enumerate(proposals):
                    sub = folder / f"generation{generation}-slot{slot}"
                    rows = run_pair(sub, latent, switch)
                    score = first_teacher_rank(rows, [False, False])
                    record = dict(slot=slot, latent=latent.tolist(), rank=list(score), rows=rows)
                    write_once(sub / "complete.json", record)
                    records.append(record)
                    if score > rank:
                        best, rank = latent.copy(), score
                mean, std, elite = elite_update(unbounded, [tuple(r["rank"]) for r in records])
                event = dict(
                    generation=generation,
                    candidates=records,
                    elite=elite,
                    next_mean=mean.tolist(),
                    next_std=std.tolist(),
                )
                write_once(folder / f"generation{generation}.json", event)
                history.append(event)
                counts["CEM_updates"] += 1
                print(
                    json.dumps(
                        dict(
                            switch=switch,
                            generation=generation,
                            best_rank=list(rank),
                            counts=counts,
                        )
                    ),
                    flush=True,
                )
            selected = []
            for repeat in range(config["selected_repeats"]):
                rows = run_pair(folder / f"selected-repeat{repeat}", best, switch)
                if first_teacher_rank(rows, [False, False]) != rank:
                    raise ValueError("selected own-state expert failed to reproduce")
                selected.append(rows)
            searches.append(
                dict(
                    switch_control=switch,
                    baseline=baseline,
                    fixed_teacher=original,
                    latent=best.tolist(),
                    selected_rank=list(rank),
                    selected=selected,
                    history=history,
                )
            )
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_INTERVENTIONAL_OWN_PREFIX_REPLAN_CLOSED_PENDING_AUDIT",
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                input_file_hashes=pins,
                config=config,
                counts=counts,
                searches=searches,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
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
                actual_native_steps=bench.native_steps if bench else 0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
