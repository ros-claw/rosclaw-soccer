"""Audit every successor CEM candidate, own network/IK and strict 2ms physics."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source-root", "source-audit-root", "source-replay-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused audit output and immutable clean source required")
    deadline = time.monotonic() + 7200
    while not (a.root / "complete.json").exists():
        if (a.root / "failed.json").exists():
            raise ValueError("teacher failed; retain data and diagnose, not automatic retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("closed teacher required; wait bounded without physics retry")
        time.sleep(10)
    if (a.root / "failed.json").exists():
        raise ValueError("failed teacher cannot be admitted by a complete marker")
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    r, d = [json.loads(f.read_text()) for f in (rp, dp)]
    pins = dict(r["input_file_hashes"])
    if (
        r["status"] != "PAID_SUCCESSOR_CONTACT_TEACHER_CLOSED_PENDING_AUDIT"
        or r["declaration_hash"] != digest(dp)
        or r["source_head"] != d["source_head"]
        or d["pairs"] != [[2, 3], [10, 11]]
        or d["generations"] != 2
        or d["population"] != 12
        or d["elite_count"] != 4
        or d["random_seed"] not in (20261022, 20261023, 20261024)
        or r["cem_distribution_updates"] != 4
        or r["optimizer_updates"] != 0
        or d["optimizer_updates"] != 0
        or any(
            obj[key] is not False
            for obj in (r, d)
            for key in ("hardware_authorized", "promotion_authorized", "uninterrupted_game")
        )
    ):
        raise ValueError("complete frozen two-pair common-intent teacher search required")
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate

    standing_gate(a.source_root, a.source_audit_root, variant=1, bank="train")
    for root in (a.source_root, a.source_audit_root, a.source_replay_root):
        if pins.get(str(root / "complete.json")) != digest(root / "complete.json"):
            raise ValueError("teacher lost its exact admitted source proof")
    original, sd = [
        json.loads((a.source_root / name).read_text())
        for name in ("complete.json", "declaration.json")
    ]
    sr = json.loads((a.source_replay_root / "complete.json").read_text())
    if (
        sr["status"] != "PAID_POST_FIRST_ALL_PHASES_STRICT_NATIVE_REPLAY_PASS"
        or sr["input_file_hashes"].get(str(a.source_root / "complete.json"))
        != digest(a.source_root / "complete.json")
        or d["source_cases"] != sd["cases"]
    ):
        raise ValueError("qualified exact TRAIN successor source required")
    pins.update({str(rp): digest(rp), str(dp): digest(dp)})
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            all_candidates_own_network_and_IK=True,
            all_candidates_strict_native_replay=True,
            hardware_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, own_neural=0, branches=0)
    replay = observer = None
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_continuity_audit import identical, snapshot_fields
        from rosclaw_soccer.rsi.paid_native_phase_replay import PaidNativePhaseReplay
        from rosclaw_soccer.rsi.paid_neural_stand_audit import StandingObservationAuditor
        from rosclaw_soccer.rsi.paid_recovery_proof import unique_proof_path
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor
        from rosclaw_soccer.rsi.paid_successor_contact_audit import audit_successor_contact
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        auditor, motions = open_paid_auditor(pins, d["source_cases"])
        observer, replay = StandingObservationAuditor(auditor), PaidNativePhaseReplay(auditor)
        policy = next(Path(path) for path in pins if path.endswith("/ckp/policy_30000.onnx"))
        scene = next(
            h for path, h in pins.items() if path.endswith("/cached-standard/declared-scene.xml")
        )
        actor, _ = load_contact_residual_actor(
            policy,
            checkpoint=unique_proof_path(pins, sd["actor_checkpoint_hash"]),
            checkpoint_hash=sd["actor_checkpoint_hash"],
            training_receipt=unique_proof_path(pins, sd["actor_training_receipt_hash"]),
            training_receipt_hash=sd["actor_training_receipt_hash"],
            scene_hash=scene,
        )
        visited = set()

        def read(row, root=a.root):
            path = root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("confined exact teacher raw required")
            pins[str(path)] = row["raw_hash"]
            if root == a.root:
                visited.add(row["raw_path"])
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def inference():
            counts["own_neural"] += 1

        def score(rows):
            return (
                int(all(x["stand"] and x["stand"]["healthy"] for x in rows)),
                int(all(x["stand"] and x["stand"]["stability_observed"] for x in rows)),
                *contact_rank([x["kick"] for x in rows]),
            )

        def checked_rows(rows, pair, latent, center, folder, *, pure=False):
            if len(rows) != 2 or [x["case"] for x in rows] != pair:
                raise ValueError("both ordered predeclared cases required")
            checked = []
            for row, i in zip(rows, pair, strict=True):
                for key, suffix in (
                    ("kick", "kick"),
                    ("kick_terminal", "kick-terminal"),
                    ("stand", "stand"),
                ):
                    if row[key] is not None and row[key]["raw_path"] != (
                        folder + f"/case{i}-{suffix}.npz"
                    ):
                        raise ValueError("each candidate must bind its own case/phase raw")
                world = original["worlds"][i]
                boundary = read(world["boundaries"][1]["boundary"], a.source_root)
                motion = motions[d["source_cases"][i]["source"]]
                if pure or not latent.any():
                    raw = read(row["kick"])
                    identical(
                        {k: v for k, v in raw.items() if not k.startswith("PLAN_")},
                        read(world["kicks"][2]["result"], a.source_root),
                    )
                    if row["stand"] is not None:
                        identical(
                            read(row["stand"]), read(world["stands"][2]["result"], a.source_root)
                        )
                result = audit_successor_contact(
                    auditor,
                    observer,
                    replay,
                    actor.base,
                    row,
                    read=read,
                    entry=snapshot_fields(boundary, "after_"),
                    previous=boundary["previous_actual_action"],
                    motion=motion,
                    center=center,
                    latent=latent,
                    pure=pure,
                    count_inference=inference,
                )
                counts["native"] += result["native"]
                counts["neural"] += result["neural"]
                counts["branches"] += 1
                checked.append(result)
            return checked

        if len(r["searches"]) != 2:
            raise ValueError("both searches retained")
        rng = np.random.default_rng(d["random_seed"])
        summaries = []
        for search, pair in zip(r["searches"], d["pairs"], strict=True):
            if search["pair"] != pair:
                raise ValueError("ordered pair identity differs")
            center = (
                sum(
                    original["worlds"][i]["kicks"][2]["result"]["postcontact"][
                        "first_foot_contact_substep"
                    ]
                    // 10
                    for i in pair
                )
                // 2
            )
            if search["center"] != center or len(search["history"]) != 2:
                raise ValueError("actual TRAIN-only contact phase differs")
            folder = f"pair{pair[0]}"
            zero = np.zeros((2, 6))
            baseline = checked_rows(
                search["baseline"], pair, zero, center, folder + "/baseline", pure=True
            )
            best, rank = zero.copy(), score(search["baseline"])
            if list(rank) != search["baseline_rank"]:
                raise ValueError("baseline rank differs")
            mean, std = zero.copy(), np.ones((2, 6)) * 0.7
            for generation, history in enumerate(search["history"]):
                record_path = a.root / folder / f"generation{generation}.json"
                if json.loads(record_path.read_text()) != history:
                    raise ValueError("sealed generation record changed")
                pins[str(record_path)] = digest(record_path)
                sampled = rng.normal(mean, std, size=(12, 2, 6))
                sampled[0] = np.arctanh(np.clip(best, -0.999999, 0.999999))
                candidates = np.tanh(sampled)
                candidates[0] = best
                if len(history["candidates"]) != 12 or history["generation"] != generation:
                    raise ValueError("all12 predeclared candidates required")
                for slot, (record, latent) in enumerate(
                    zip(history["candidates"], candidates, strict=True)
                ):
                    subfolder = folder + f"/generation{generation}-slot{slot}"
                    cp = a.root / subfolder / "complete.json"
                    if (
                        json.loads(cp.read_text()) != record
                        or record["slot"] != slot
                        or not np.array_equal(record["latent"], latent)
                    ):
                        raise ValueError("seeded CEM proposal record differs")
                    pins[str(cp)] = digest(cp)
                    checked_rows(record["rows"], pair, latent, center, subfolder)
                    actual_rank = score(record["rows"])
                    if list(actual_rank) != record["rank"]:
                        raise ValueError("CEM rank differs from actual physics")
                    if actual_rank > rank:
                        best, rank = latent.copy(), actual_rank
                elite = sorted(
                    range(12), key=lambda k: tuple(history["candidates"][k]["rank"]), reverse=True
                )[:4]
                mean, std = sampled[elite].mean(axis=0), np.maximum(sampled[elite].std(axis=0), 0.1)
                if (
                    history["elite"] != elite
                    or not np.array_equal(history["next_mean"], mean)
                    or not np.array_equal(history["next_std"], std)
                ):
                    raise ValueError("CEM elite/distribution update differs")
                print(json.dumps(dict(pair=pair, generation=generation)), flush=True)
            selected = checked_rows(search["selected"], pair, best, center, folder + "/selected")
            if (
                not np.array_equal(search["latent"], best)
                or search["selected_rank"] != list(rank)
                or score(search["selected"]) != rank
            ):
                raise ValueError("selected common-intent teacher differs")
            summaries.append(
                dict(
                    pair=pair,
                    baseline=baseline,
                    selected=selected,
                    baseline_rank=search["baseline_rank"],
                    selected_rank=list(rank),
                )
            )
        if (
            counts["native"] != r["actual_native_steps"]
            or counts["native"] != replay.native_steps
            or counts["neural"] != r["actual_neural_rows"]
            or counts["own_neural"] != counts["neural"]
            or counts["branches"] != 104
            or visited != {str(f.relative_to(a.root)) for f in a.root.rglob("*.npz")}
        ):
            raise ValueError("complete teacher raw/work/inference coverage differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_SUCCESSOR_CONTACT_ALL_CANDIDATES_NATIVE_NEURAL_IK_CEM_AUDIT_PASS",
                input_file_hashes=pins,
                source_head=a.expected_head,
                summaries=summaries,
                counts=counts,
                own_native_steps=replay.native_steps,
                own_kinematic_observations=observer.evaluations,
                optimizer_updates=0,
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
                own_native_steps=replay.native_steps if replay is not None else 0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
