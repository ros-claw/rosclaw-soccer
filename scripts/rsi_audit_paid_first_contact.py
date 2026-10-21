"""Independently verify every first-contact CEM trial's own NN, IK, native physics and search."""

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
            raise ValueError("frozen clean independent audit source required")

    clean()
    if a.output.exists():
        raise ValueError("unused audit output required")
    deadline = time.monotonic() + 7200
    while not (a.root / "complete.json").exists():
        clean()
        if (a.root / "failed.json").exists():
            raise ValueError("search failed, no automatic retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("closed complete first-contact search required")
        time.sleep(10)
    from rosclaw_soccer.rsi.paid_first_teacher import (
        FIRST_PAIRS,
        first_teacher_config,
        first_teacher_rank,
    )

    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    r, d = [json.loads(f.read_text()) for f in (rp, dp)]
    config = first_teacher_config(d["config"]["seed_index"])
    if (
        (a.root / "failed.json").exists()
        or r["status"] != "PAID_TRAIN_FIRST_CONTACT_TEACHER_CLOSED_PENDING_OWN_AUDIT"
        or d["config"] != config
        or r["config"] != config
        or any(o["source_head"] != a.producer_head for o in (r, d))
        or r["declaration_hash"] != digest(dp)
        or r["episodes"] != 108
        or r["cem_distribution_updates"] != 4
        or r["optimizer_updates"] != 0
        or any(o["bank"] != "train" or o["Fresh_rows_read"] != 0 for o in (r, d))
        or any(
            o[k] is not False
            for o in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("entire fixed first-contact TRAIN-only pilot required")
    qr = Path(d["qualification_root"])
    qp = qr / "complete.json"
    if digest(qp) != d["qualification_hash"]:
        raise ValueError("original qualification changed")
    qualified = json.loads(qp.read_text())
    qd = json.loads((qr / "declaration.json").read_text())
    references = []
    for name in qd["reference_shards"]:
        root = Path(name)
        f = root / "complete.json"
        if qualified["input_file_hashes"].get(str(f)) != digest(f):
            raise ValueError("qualified TRAIN reference changed")
        shard = json.loads(f.read_text())
        sd = json.loads((root / "declaration.json").read_text())
        start = shard["shard"] * 5
        if sd["cases"] != d["source_cases"][start : start + 5]:
            raise ValueError("complete exact source cases required")
        references.extend((root / "grown", g["arms"]["grown"]) for g in shard["groups"])
    if len(references) != 20 or [w["case"] for _, w in references] != list(range(20)):
        raise ValueError("complete ordered TRAIN source bank required")
    pins = dict(r["input_file_hashes"])
    for f in (rp, dp):
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
            source_hash=digest(rp),
            input_file_hashes=pins,
            own_neural_recompute=True,
            own_kinematic_IK_recompute=True,
            own_all_trial_native_integration=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, own_neural=0, episodes=0)
    replay = None
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
        auditor, motions = open_paid_auditor(pins, d["source_cases"])
        observer, replay = StandingObservationAuditor(auditor), PaidNativePhaseReplay(auditor)
        sr = Path(d["student_root"])
        training = json.loads((sr / "complete.json").read_text())
        if digest(sr / "complete.json") != d["student_hash"]:
            raise ValueError("exact frozen base lineage required")
        policy = next(Path(p) for p in pins if p.endswith("/ckp/policy_30000.onnx"))
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=sr / "complete.json",
            training_receipt_hash=d["student_hash"],
            scene_hash=training["body_scene_hash"],
        )
        visited = set()

        def read(row, root=a.root):
            f = root / row["raw_path"]
            if (
                f.is_symlink()
                or not f.resolve().is_relative_to(root.resolve())
                or digest(f) != row["raw_hash"]
            ):
                raise ValueError("exact confined own trial raw required")
            pins[str(f)] = row["raw_hash"]
            if root == a.root:
                visited.add(row["raw_path"])
            with np.load(f, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def inference():
            counts["own_neural"] += 1

        def checked_rows(rows, pair, latent, center, folder, *, pure=False):
            if len(rows) != 2 or [x["case"] for x in rows] != list(pair):
                raise ValueError("both fixed paired cases required")
            for row, i in zip(rows, pair, strict=True):
                for key, suffix in (
                    ("initial", "initial"),
                    ("kick", "kick"),
                    ("kick_terminal", "kick-terminal"),
                    ("stand", "stand"),
                ):
                    if (
                        row[key] is not None
                        and row[key]["raw_path"] != folder + f"/case{i}-{suffix}.npz"
                    ):
                        raise ValueError("trial must bind own case/phase raw")
                entry = read(row["initial"])
                root, world = references[i]
                identical(entry, read(world["initial"], root))
                if pure or not latent.any():
                    z = read(row["kick"])
                    identical(
                        {k: v for k, v in z.items() if not k.startswith("PLAN_")},
                        read(world["kicks"][0]["result"], root),
                    )
                    if row["stand"] is not None:
                        identical(read(row["stand"]), read(world["stands"][0]["result"], root))
                result = audit_successor_contact(
                    auditor,
                    observer,
                    replay,
                    actor.base,
                    row,
                    read=read,
                    entry=entry,
                    previous=np.zeros(29, np.float32),
                    motion=motions[d["source_cases"][i]["source"]],
                    center=center,
                    latent=latent,
                    pure=pure,
                    count_inference=inference,
                )
                counts["native"] += result["native"]
                counts["neural"] += result["neural"]
                counts["episodes"] += 1

        rng = np.random.default_rng(config["random_seed"])
        summaries = []
        if len(r["searches"]) != 2:
            raise ValueError("both declared searches required")
        for search, pair in zip(r["searches"], FIRST_PAIRS, strict=True):
            center = (
                sum(
                    references[i][1]["kicks"][0]["result"]["postcontact"][
                        "first_foot_contact_substep"
                    ]
                    // 10
                    for i in pair
                )
                // 2
            )
            protected = [references[i][1]["quality_by_kick"][0] for i in pair]
            if (
                search["pair"] != list(pair)
                or search["center"] != center
                or search["protected"] != protected
                or d["centers"][str(pair[0])] != center
                or len(search["history"]) != 2
            ):
                raise ValueError("paired cases/contact phase/protection changed")
            folder = f"pair{pair[0]}"
            zero = np.zeros((2, 6))
            checked_rows(search["baseline"], pair, zero, center, folder + "/baseline", pure=True)
            best, rank = zero.copy(), first_teacher_rank(search["baseline"], protected)
            if search["baseline_rank"] != list(rank):
                raise ValueError("actual baseline rank differs")
            mean, std = zero.copy(), np.ones((2, 6)) * 0.7
            for generation, event in enumerate(search["history"]):
                f = a.root / folder / f"generation{generation}.json"
                if json.loads(f.read_text()) != event:
                    raise ValueError("sealed generation differs")
                pins[str(f)] = digest(f)
                sampled = rng.normal(mean, std, size=(12, 2, 6))
                sampled[0] = np.arctanh(np.clip(best, -0.999999, 0.999999))
                proposals = np.tanh(sampled)
                proposals[0] = best
                records = event["candidates"]
                if event["generation"] != generation or len(records) != 12:
                    raise ValueError("all12 declared trials required")
                for slot, (record, latent) in enumerate(zip(records, proposals, strict=True)):
                    subfolder = folder + f"/generation{generation}-slot{slot}"
                    f = a.root / subfolder / "complete.json"
                    if (
                        json.loads(f.read_text()) != record
                        or record["slot"] != slot
                        or not np.array_equal(record["latent"], latent)
                    ):
                        raise ValueError("seeded unmodified CEM proposals required")
                    pins[str(f)] = digest(f)
                    checked_rows(record["rows"], pair, latent, center, subfolder)
                    score = first_teacher_rank(record["rows"], protected)
                    if list(score) != record["rank"]:
                        raise ValueError("actual contact/recovery/protection ranking differs")
                    if score > rank:
                        best, rank = latent.copy(), score
                elite = sorted(range(12), key=lambda k: tuple(records[k]["rank"]), reverse=True)[:4]
                mean, std = sampled[elite].mean(axis=0), np.maximum(sampled[elite].std(axis=0), 0.1)
                if (
                    event["elite"] != elite
                    or not np.array_equal(event["next_mean"], mean)
                    or not np.array_equal(event["next_std"], std)
                ):
                    raise ValueError("actual elite distribution update differs")
                print(
                    json.dumps(
                        dict(
                            pair=pair,
                            generation=generation,
                            status="ALL_TRIALS_OWN_NN_IK_PHYSICS_CHECKED",
                        )
                    ),
                    flush=True,
                )
            if (
                not np.array_equal(best, search["latent"])
                or list(rank) != search["selected_rank"]
                or len(search["selected"]) != 2
            ):
                raise ValueError("best teacher selection differs")
            for repeat, rows in enumerate(search["selected"]):
                checked_rows(rows, pair, best, center, folder + f"/selected-repeat{repeat}")
                if first_teacher_rank(rows, protected) != rank:
                    raise ValueError("selected teacher physical rank not reproducible")
            for one, two in zip(*search["selected"], strict=True):
                for key in ("initial", "kick", "kick_terminal", "stand"):
                    if (one[key] is None) != (two[key] is None):
                        raise ValueError("selected repeated branch coverage differs")
                    if one[key] is not None:
                        identical(read(one[key]), read(two[key]))
            summaries.append(
                dict(
                    pair=list(pair),
                    baseline_rank=search["baseline_rank"],
                    selected_rank=list(rank),
                    protected=protected,
                    own_all_trials_verified=True,
                )
            )
        if (
            counts["native"] != r["actual_native_steps"]
            or replay.native_steps != counts["native"]
            or counts["neural"] != r["actual_neural_rows"]
            or counts["own_neural"] != counts["neural"]
            or counts["episodes"] != 108
            or len(visited) != len(list(a.root.rglob("*.npz")))
        ):
            raise ValueError("complete native/own-NN/raw coverage accounting differs")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_FIRST_CONTACT_ALL_TRIAL_OWN_NN_IK_AND_NATIVE_AUDIT_PASS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                source_hash=digest(rp),
                config=config,
                counts=counts,
                summaries=summaries,
                own_native_steps=replay.native_steps,
                own_trajectory_neural_rows=counts["own_neural"],
                selected_whole_repeats_exact=True,
                first_kick_replayed=True,
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
                own_native_steps=replay.native_steps if replay else 0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
