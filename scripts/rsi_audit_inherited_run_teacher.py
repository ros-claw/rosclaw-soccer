"""Audit every trial, failure and selected repeat in the fixed TRAIN expert pilot."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--source-hash")
    p.add_argument("--producer-head")
    p.add_argument("--declaration-hash")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean pinned independent audit source required")

    clean()
    from rosclaw_soccer.rsi.queued_audit_commitment import bind_complete

    a.source_hash = bind_complete(
        a.root,
        complete_hash=a.source_hash,
        producer_head=a.producer_head,
        declaration_hash=a.declaration_hash,
    )
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (rp, dp))
        or digest(rp) != a.source_hash
        or (a.root / "failed.json").exists()
    ):
        raise ValueError("exact closed source and unused audit output required")
    r, d = [json.loads(x.read_text()) for x in (rp, dp)]
    response = d["protocol"] == "paid.TRAIN.inherited_run_actual_state_expert.response_v3"
    contextual = (
        response or d["protocol"] == "paid.TRAIN.inherited_run_actual_state_expert.contextual_v2"
    )
    total_worlds, rounds, candidates = (
        (28, 0, 11) if response else (134, 4, 16) if contextual else (38, 2, 8)
    )
    if (
        r["status"] != "INHERITED_RUN_TRAIN_EXPERT_SEARCH_CLOSED_PENDING_INDEPENDENT_AUDIT"
        or d["protocol"]
        not in (
            "paid.TRAIN.inherited_run_actual_state_expert.v1",
            "paid.TRAIN.inherited_run_actual_state_expert.contextual_v2",
            "paid.TRAIN.inherited_run_actual_state_expert.response_v3",
        )
        or r["declaration_hash"] != digest(dp)
        or r["input_file_hashes"] != d["input_file_hashes"]
        or r["worlds"] != total_worlds
        or len(r["records"]) != total_worlds
        or not r["selected_whole_repeats_exact"]
        or r["source_head"] != d["source_head"]
        or d["center"] != 105
        or d["seed"] != 41022
        or d["rounds"] != rounds
        or d["candidates_per_round"] != candidates
        or d["elite"] != (0 if response else 4 if contextual else 2)
        or d["initial_std"] != (None if response else 0.2 if contextual else 0.6)
        or d["minimum_std"] != (None if response else 0.03 if contextual else 0.1)
        or any(
            v.get(k) is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized", "learned_policy_claim")
        )
    ):
        raise ValueError("fixed full38world TRAIN expert declaration required")
    pins = dict(r["input_file_hashes"])
    for x in (rp, dp):
        pins[str(x)] = digest(x)
    paths = [
        Path(x)
        for x, h in pins.items()
        if h == d["motor_lineage_hash"] and x.endswith("/complete.json")
    ]
    if len(paths) != 1:
        raise ValueError("exact inherited motor lineage required")
    training = json.loads(paths[0].read_text())
    source_paths = [
        Path(x) for x, h in pins.items() if h == d["source_hash"] and x.endswith("/complete.json")
    ]
    if len(source_paths) != 1:
        raise ValueError("exact source-bound original two-course witness required")
    original = json.loads(source_paths[0].read_text())
    response_report = None
    if response:
        response_paths = [
            Path(x)
            for x, h in pins.items()
            if h == d["response_hash"] and x.endswith("/complete.json")
        ]
        if len(response_paths) != 1:
            raise ValueError("exact response diagnostic commitment required")
        response_report = json.loads(response_paths[0].read_text())
        if (
            response_report["status"] != "TRAIN_RUN_CONTACT_RESPONSE_DIAGNOSED_NOT_MOTOR_POLICY"
            or response_report["predicted_success_is_evidence"] is not False
        ):
            raise ValueError("physical trials cannot inherit predicted success labels")
    centers = [105, 105]
    if contextual:
        for ci in range(2):
            witness = next(
                x
                for x in original["records"]
                if x["course"] == ci and x["arm"] == "reference_from_start" and x["repeat"] == 0
            )
            centers[ci] = witness["result"]["postcontact"]["first_foot_contact_substep"] // 10
        if d["centers"] != centers or d["independent_context_search"] is not True:
            raise ValueError("source contact phase and separate context experts required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    counts = dict(native=0, neural=0, kinematic_intent=0)
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_run_pilot_recovery import contact_quality, stand_from_raw
        from rosclaw_soccer.rsi.paid_run_teacher_audit import equal, replay_teacher_world

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, [d["source_case"]])
        wrapper, _ = load_paid_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=paths[0],
            training_receipt_hash=d["motor_lineage_hash"],
            scene_hash=scene,
        )
        actor = wrapper.base
        candidate_tags = (
            {f"r-n{n}-c{c}" for n in range(11) for c in range(2)}
            if response
            else {
                f"g{g}-n{n}-c{c}"
                for g in range(rounds)
                for n in range(candidates)
                for c in range(2)
            }
        )
        expected = (
            {f"baseline-c{c}" for c in range(2)}
            | candidate_tags
            | {f"selected-c{c}-repeat{t}" for c in range(2) for t in range(2)}
        )
        if {x["tag"] for x in r["records"]} != expected:
            raise ValueError("all failure/candidate/repeat trials required")
        outcomes = {}
        all_blobs = {}
        for rec in r["records"]:
            ci = rec["course"]
            if ci not in (0, 1) or rec["tag"] not in expected:
                raise ValueError("declared world identity required")
            expected.remove(rec["tag"])
            blobs = {}
            for name, art in rec["artifacts"].items():
                x = a.root / art["path"]
                if (
                    x.is_symlink()
                    or not x.resolve().is_relative_to(a.root.resolve())
                    or digest(x) != art["hash"]
                ):
                    raise ValueError("confined source-bound raw required")
                pins[str(x)] = art["hash"]
                with np.load(x, allow_pickle=False) as z:
                    blobs[name] = {k: z[k].copy() for k in z.files}
            kick, stand = blobs["kick"], blobs.get("stand")
            ep = bench.make(d["source_case"])
            q, _ = ep.layout.canonical(ep.env.data)
            w, x, y, z = q[3:7]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            ep.env.data.qpos[ep.layout.rq : ep.layout.rq + 2] -= d["courses"][ci][
                "backoff_m"
            ] * np.array([np.cos(yaw), np.sin(yaw)])
            mujoco.mj_forward(ep.env.model, ep.env.data)
            before = dict(counts)
            replay_teacher_world(
                ep,
                actor,
                kick,
                stand,
                latent=np.array(rec["latent"], np.float64),
                center=centers[ci],
                foot="left" if "left" in d["source_case"]["source"] else "right",
                counts=counts,
            )
            if (
                counts["native"] - before["native"] != rec["native_steps"]
                or counts["neural"] - before["neural"] != rec["neural_rows"]
            ):
                raise ValueError("actual per-world work differs")
            arrays = {
                k: np.concatenate([kick[k], stand[k]]) if stand is not None else kick[k]
                for k in ("qpos", "qvel", "forces", "body_failed")
            }
            f = arrays["forces"].reshape(-1, 3)
            post = native_contact_window(
                force=f[:, :2],
                canonical_qpos=arrays["qpos"].reshape(-1, 43),
                canonical_qvel=arrays["qvel"].reshape(-1, 41),
                goal=ep.goal,
                intended_foot=0 if "left" in d["source_case"]["source"] else 1,
            )
            result = dict(
                body_failed=bool(arrays["body_failed"].any()),
                forbidden=bool((f[:, 2] > 1).any()),
                nonfoot_impulse_Ns=float(f[:, 2].sum() * 0.002),
                postcontact=post,
            )
            ss = stand_from_raw(stand) if stand is not None else None
            quality = bool(
                contact_quality(result) and ss and ss["healthy"] and ss["stability_observed"]
            )
            if result != rec["result"] or ss != rec["stand"] or quality != rec["healthy_quality"]:
                raise ValueError("own physical scoring differs")
            outcomes[rec["tag"]] = rec
            all_blobs[rec["tag"]] = blobs
            print(
                json.dumps(dict(tag=rec["tag"], native=counts["native"], healthy_quality=quality)),
                flush=True,
            )

        # Verify CEM proposals, generation updates and deterministic selected repeats.
        def rank(rr):
            ps = [x["result"]["postcontact"] for x in rr]
            return [
                sum(x["healthy_quality"] for x in rr),
                sum(x["correct_foot_first"] for x in ps),
                sum(not x["result"]["forbidden"] for x in rr),
                sum(x["post_contact_quality_observed"] for x in ps),
                -sum(
                    x["goal_lateral_1p2s_m"] if x["goal_lateral_1p2s_m"] is not None else 100
                    for x in ps
                ),
                sum(
                    x["goal_progress_1p2s_m"] if x["goal_progress_1p2s_m"] is not None else -100
                    for x in ps
                ),
            ]

        base = [outcomes[f"baseline-c{c}"] for c in range(2)]
        best_rank, best, best_tags = rank(base), np.zeros((2, 6)), [x["tag"] for x in base]
        if contextual:
            from rosclaw_soccer.rsi.run_teacher_objective import run_teacher_rank

            context_latents, context_tags, context_ranks = [], [], []
            for ci in range(2):
                best_row = base[ci]
                best_latent = np.zeros((2, 6))
                context_rng = np.random.default_rng(41022 + ci)
                context_mean, context_std = np.zeros((2, 6)), np.full((2, 6), 0.2)
                if response:
                    for n, proposed in enumerate(
                        response_report["contexts"][ci]["proposed_latents"]
                    ):
                        row = outcomes[f"r-n{n}-c{ci}"]
                        latent = np.array(proposed, np.float64)
                        equal(
                            latent, np.array(row["latent"]), "preregistered response intervention"
                        )
                        if run_teacher_rank(row) > run_teacher_rank(best_row):
                            best_row, best_latent = row, latent.copy()
                for g in range(0 if response else 4):
                    trials = []
                    for n in range(16):
                        latent = np.clip(context_rng.normal(context_mean, context_std), -1, 1)
                        row = outcomes[f"g{g}-n{n}-c{ci}"]
                        equal(latent, np.array(row["latent"]), "contextual preregistered CEM")
                        score = run_teacher_rank(row)
                        trials.append((score, n, latent))
                        if score > run_teacher_rank(best_row):
                            best_row, best_latent = row, latent.copy()
                    elites = sorted(trials, key=lambda x: (x[0], -x[1]), reverse=True)[:4]
                    values = np.stack([x[2] for x in elites])
                    context_mean = values.mean(axis=0)
                    context_std = np.maximum(values.std(axis=0), 0.03)
                context_latents.append(best_latent)
                context_tags.append(best_row["tag"])
                context_ranks.append(list(run_teacher_rank(best_row)))
            best, best_tags = np.stack(context_latents), context_tags
            best_rank = rank([outcomes[tag] for tag in best_tags])
            if r["selected_context_ranks"] != context_ranks:
                raise ValueError("context-specific expert selection differs")
        rng = np.random.default_rng(41022)
        mean, std = np.zeros((2, 6)), np.full((2, 6), 0.6)
        for g in range(0 if contextual else 2):
            trials = []
            for n in range(8):
                latent = np.clip(rng.normal(mean, std), -1, 1)
                pair = [outcomes[f"g{g}-n{n}-c{c}"] for c in range(2)]
                for x in pair:
                    equal(latent, np.array(x["latent"]), "preregistered CEM latent")
                score = rank(pair)
                trials.append((score, n, latent))
                if score > best_rank:
                    best_rank, best, best_tags = score, latent.copy(), [x["tag"] for x in pair]
            elite = sorted(trials, key=lambda x: (x[0], -x[1]), reverse=True)[:2]
            values = np.stack([x[2] for x in elite])
            mean, std = values.mean(axis=0), np.maximum(values.std(axis=0), 0.1)
        equal(best, np.array(r["selected_latent"]), "selected latent")
        if best_rank != r["selected_rank"] or rank(base) != r["baseline_rank"]:
            raise ValueError("selection ranks differ")
        for ci in range(2):
            for t in range(2):
                tag = f"selected-c{ci}-repeat{t}"
                equal(
                    np.array(outcomes[tag]["latent"]),
                    best[ci] if contextual else best,
                    "repeated selected latent",
                )
                for name, blob in all_blobs[tag].items():
                    ref = all_blobs[best_tags[ci]][name]
                    if set(blob) != set(ref):
                        raise ValueError("whole selected repeat schema differs")
                    for k, v in blob.items():
                        equal(v, ref[k], "whole selected repeat " + k)
        if counts != dict(
            native=r["actual_native_steps"],
            neural=r["actual_neural_calls"],
            kinematic_intent=r["actual_kinematic_intent_calls"],
        ):
            raise ValueError("complete all-trial actual work counts differ")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="INHERITED_RUN_EXPERT_ALL_TRIAL_WHOLE_NATIVE_OWN_NN_IK_AUDIT_PASS",
                source_hash=a.source_hash,
                source_head=a.expected_head,
                input_file_hashes=pins,
                counts=counts,
                worlds=total_worlds,
                protocol=d["protocol"],
                independent_whole_world_replay=True,
                shared_observation_codec=True,
                shared_IK_solver_reexecuted=True,
                selected_rank=best_rank,
                baseline_rank=rank(base),
                new_qualified_policy=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error), counts=counts, automatic_retry=False, hardware_authorized=False
            ),
        )
        raise


if __name__ == "__main__":
    main()
