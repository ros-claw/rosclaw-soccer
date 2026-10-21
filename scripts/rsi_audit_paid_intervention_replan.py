"""All-trial independent own-NN/IK/native audit of on-student-prefix CEM."""

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
            raise ValueError("clean frozen independent replan auditor required")

    clean()
    if a.output.exists():
        raise ValueError("unused independent audit output required")
    deadline = time.monotonic() + 7200
    while not (a.root / "complete.json").exists():
        clean()
        if (a.root / "failed.json").exists():
            raise ValueError("producer failed; preserve and diagnose without implicit retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("closed finite own-prefix search required")
        time.sleep(10)
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    r, d = [json.loads(f.read_text()) for f in (rp, dp)]
    if (
        r["status"] != "PAID_INTERVENTIONAL_OWN_PREFIX_REPLAN_CLOSED_PENDING_AUDIT"
        or r["source_head"] != a.producer_head
        or d["source_head"] != a.producer_head
        or r["declaration_hash"] != digest(dp)
        or r["config"] != d["config"]
        or r["optimizer_updates"] != 0
        or r["Fresh_rows_read"] != 0
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("closed exact SIM-only replan evidence required")
    roots, sources, cases, pins = load_sources(SimpleNamespace(**d["source_arguments"]))
    for path, h in r["input_file_hashes"].items():
        if path in pins and pins[path] != h:
            raise ValueError("conflicting replan source lineage")
        pins[path] = h
    pins[str(rp)], pins[str(dp)] = digest(rp), digest(dp)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    counts = dict(native=0, neural=0, own_neural=0, episodes=0, R2_planner=0, IK=0)
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_first_teacher import first_teacher_rank
        from rosclaw_soccer.rsi.paid_intervention_replan import (
            elite_update,
            replan_config,
            sample_latents,
        )
        from rosclaw_soccer.rsi.paid_native_phase_replay import PaidNativePhaseReplay
        from rosclaw_soccer.rsi.paid_neural_stand_audit import StandingObservationAuditor
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor
        from rosclaw_soccer.rsi.paid_successor_contact_audit import audit_successor_contact

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        auditor, motions = open_paid_auditor(pins, [cases[i] for i in (4, 5)])
        observer, replay = StandingObservationAuditor(auditor), PaidNativePhaseReplay(auditor)
        fixed = sources["teacher"]["searches"][0]
        center, initial_latent = fixed["center"], np.array(fixed["latent"], np.float64)
        config = replan_config(center, len(motions[cases[4]["source"]]["q"]))
        if (
            r["config"] != config
            or [s["switch_control"] for s in r["searches"]] != config["switch_controls"]
        ):
            raise ValueError("predeclared complete two-boundary design required")
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
        visited = set()

        def read(link, root=a.root):
            path = root / link["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != link["raw_hash"]
            ):
                raise ValueError("confined exact raw required")
            pins[str(path)] = link["raw_hash"]
            if root == a.root:
                visited.add(link["raw_path"])
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def inference():
            counts["own_neural"] += 1

        def pair(rows, switch, latent, folder):
            if [row["case"] for row in rows] != [4, 5]:
                raise ValueError("both paired source contexts required")
            results, blobs = [], {}
            for row in rows:
                i = row["case"]
                if row["switch_control"] != switch or row["latent"] != latent.tolist():
                    raise ValueError("declared current-prefix teacher identity changed")
                for key, suffix in (
                    ("initial", "initial"),
                    ("kick", "kick"),
                    ("kick_terminal", "kick-terminal"),
                    ("stand", "stand"),
                ):
                    if (
                        row[key] is not None
                        and row[key]["raw_path"] != f"{folder}/case{i}/{suffix}.npz"
                    ):
                        raise ValueError("all-trial phase path identity differs")
                entry = read(row["initial"])
                source = sources["student_exam"]["worlds"][i]
                initial_link = next(
                    v for v in source["artifacts"] if Path(v["raw_path"]).stem == "world-initial"
                )
                kick_link = next(
                    v for v in source["artifacts"] if Path(v["raw_path"]).stem == "kick0"
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
                    handoff_control=switch,
                )
                raw, old = read(row["kick"]), read(kick_link, roots["student_exam"])
                for key in (
                    "NN_obs",
                    "NN_action",
                    "NN_h_before",
                    "NN_c_before",
                    "NN_h_after",
                    "NN_c_after",
                ):
                    if not np.array_equal(raw[key][:switch], old[key][:switch]):
                        raise ValueError("own current-student prefix differs")
                healthy = bool(
                    result["quality"] and result["stand_healthy"] and result["stand_stable"]
                )
                if row["quality"] != result["quality"] or row["healthy_success"] != healthy:
                    raise ValueError("independent expert outcome differs")
                result.update(case=i, healthy_success=healthy)
                results.append(result)
                for key in ("native", "neural"):
                    counts[key] += result[key]
                counts["R2_planner"] += result["own_R2_planner_head_rows"]
                counts["IK"] += result["own_IK_queries"]
                counts["episodes"] += 1
                blobs[f"{i}-kick"] = raw
                if row["stand"] is not None:
                    blobs[f"{i}-stand"] = read(row["stand"])
            return results, blobs

        rng = np.random.default_rng(config["seed"])
        summaries = []
        for search in r["searches"]:
            switch = search["switch_control"]
            folder = f"switch{switch}"
            pair(search["baseline"], config["horizon"], initial_latent, folder + "/student-control")
            pair(search["fixed_teacher"], switch, initial_latent, folder + "/fixed-teacher-control")
            best, rank = (
                initial_latent.copy(),
                first_teacher_rank(search["fixed_teacher"], [False, False]),
            )
            mean, std = np.arctanh(np.clip(best, -0.999999, 0.999999)), np.full((2, 6), 0.7)
            if len(search["history"]) != 2:
                raise ValueError("both fixed generations required")
            for generation, event in enumerate(search["history"]):
                unbounded, proposals = sample_latents(rng, mean, std, best)
                if event["generation"] != generation or [
                    v["slot"] for v in event["candidates"]
                ] != list(range(12)):
                    raise ValueError("all candidate slots required")
                ranks = []
                for slot, record in enumerate(event["candidates"]):
                    latent = proposals[slot]
                    if record["latent"] != latent.tolist():
                        raise ValueError("deterministic CEM proposal differs")
                    pair(
                        record["rows"],
                        switch,
                        latent,
                        f"{folder}/generation{generation}-slot{slot}",
                    )
                    score = first_teacher_rank(record["rows"], [False, False])
                    if record["rank"] != list(score):
                        raise ValueError("actual paired candidate rank differs")
                    ranks.append(score)
                    if score > rank:
                        best, rank = latent.copy(), score
                mean, std, elite = elite_update(unbounded, ranks)
                if (
                    event["elite"] != elite
                    or event["next_mean"] != mean.tolist()
                    or event["next_std"] != std.tolist()
                ):
                    raise ValueError("deterministic physical elite update differs")
                print(
                    json.dumps(
                        dict(
                            switch=switch,
                            generation=generation,
                            status="ALL_TRIALS_OWN_NN_IK_NATIVE_CHECKED",
                        )
                    ),
                    flush=True,
                )
            if (
                search["latent"] != best.tolist()
                or search["selected_rank"] != list(rank)
                or len(search["selected"]) != 2
            ):
                raise ValueError("selected actual incumbent differs")
            outcomes, first_blobs = [], None
            for repeat, rows in enumerate(search["selected"]):
                results, blobs = pair(rows, switch, best, f"{folder}/selected-repeat{repeat}")
                if first_teacher_rank(rows, [False, False]) != rank:
                    raise ValueError("selected rank does not reproduce")
                if first_blobs is not None:
                    if set(blobs) != set(first_blobs) or results != outcomes[0]:
                        raise ValueError("repeat phases or outcomes differ")
                    for k in blobs:
                        identical(blobs[k], first_blobs[k])
                first_blobs = blobs
                outcomes.append(results)
            summaries.append(
                dict(switch_control=switch, outcomes=outcomes, selected_rank=list(rank))
            )
        actual = {str(f.relative_to(a.root)) for f in a.root.rglob("*.npz")}
        if (
            visited != actual
            or counts["episodes"] != config["max_episodes"]
            or counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_neural_rows"]
            or counts["R2_planner"] != r["counts"]["R2_planner"]
            or counts["IK"] != r["counts"]["IK"]
        ):
            raise ValueError("all charged trial phases and compute must be independently accounted")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_INTERVENTIONAL_OWN_PREFIX_ALL_TRIAL_NN_IK_NATIVE_AUDIT_PASS",
                source_head=a.expected_head,
                source_hash=digest(rp),
                input_file_hashes=pins,
                counts=counts,
                summaries=summaries,
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
