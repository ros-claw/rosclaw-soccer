"""Independently verify frozen teacher transfer, all32 entries and both repeats."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and clean immutable audit source required")
    deadline = time.monotonic() + 7200
    while not (a.root / "complete.json").exists():
        if (a.root / "failed.json").exists():
            raise ValueError("transfer failed; preserve branch evidence and diagnose")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("closed transfer required, no automatic retry")
        time.sleep(10)
    if (a.root / "failed.json").exists():
        raise ValueError("failed transfer cannot be admitted")
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    r, d = [json.loads(f.read_text()) for f in (rp, dp)]
    if (
        r["status"] != "PAID_FROZEN_SUCCESSOR_TEACHER_TRAIN_DEV_TRANSFER_PENDING_AUDIT"
        or r["source_head"] != a.producer_head
        or d["source_head"] != a.producer_head
        or r["declaration_hash"] != digest(dp)
        or set(r["results"]) != {"train", "dev"}
        or d["teacher_seed"] != 0
        or d["repeats"] != 2
        or d["all16_per_bank"] is not True
        or d["no_DEV_search"] is not True
        or r["all_branch_repeat_arrays_exact"] is not True
        or any(
            obj[key] is not False
            for obj in (r, d)
            for key in ("hardware_authorized", "promotion_authorized", "uninterrupted_game")
        )
        or r["optimizer_updates"] != 0
        or d["optimizer_updates"] != 0
    ):
        raise ValueError("exact predeclared full-bank seed0 restored-branch transfer required")
    pins = dict(r["input_file_hashes"])
    teacher_root, teacher_audit = Path(d["teacher_root"]), Path(d["teacher_audit_root"])
    if (
        pins.get(str(teacher_root / "complete.json")) != d["teacher_hash"]
        or digest(teacher_root / "complete.json") != d["teacher_hash"]
        or pins.get(str(teacher_audit / "complete.json")) != digest(teacher_audit / "complete.json")
    ):
        raise ValueError("exact admitted teacher proof required")
    tr, td, ta = [
        json.loads((root / name).read_text())
        for root, name in (
            (teacher_root, "complete.json"),
            (teacher_root, "declaration.json"),
            (teacher_audit, "complete.json"),
        )
    ]
    if (
        ta["status"] != "PAID_SUCCESSOR_CONTACT_ALL_CANDIDATES_NATIVE_NEURAL_IK_CEM_AUDIT_PASS"
        or ta["input_file_hashes"].get(str(teacher_root / "complete.json")) != d["teacher_hash"]
        or td["random_seed"] != 20261022
        or td["pairs"] != [[2, 3], [10, 11]]
    ):
        raise ValueError("predeclared seed0 teacher proof differs")
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate

    banks = {}
    for bank in ("train", "dev"):
        root, audit = Path(d["bank_roots"][bank]), Path(d["bank_audit_roots"][bank])
        if standing_gate(root, audit, variant=1, bank=bank) != d["baseline_gates"][bank]:
            raise ValueError("frozen baseline evidence changed")
        source, sd = [
            json.loads((root / name).read_text()) for name in ("complete.json", "declaration.json")
        ]
        banks[bank] = (root, source, sd)
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
            all_branches_required=64,
            own_NN_and_IK_and_strict_native_replay=True,
            hardware_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, own_neural=0, branches=0)
    replay = None
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

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cases = [case for _, _, sd in banks.values() for case in sd["cases"]]
        auditor, motions = open_paid_auditor(pins, cases)
        observer, replay = StandingObservationAuditor(auditor), PaidNativePhaseReplay(auditor)
        policy = next(Path(path) for path in pins if path.endswith("/ckp/policy_30000.onnx"))
        scene = next(
            h for path, h in pins.items() if path.endswith("/cached-standard/declared-scene.xml")
        )
        sd = banks["train"][2]
        actor, _ = load_contact_residual_actor(
            policy,
            checkpoint=unique_proof_path(pins, sd["actor_checkpoint_hash"]),
            checkpoint_hash=sd["actor_checkpoint_hash"],
            training_receipt=unique_proof_path(pins, sd["actor_training_receipt_hash"]),
            training_receipt_hash=sd["actor_training_receipt_hash"],
            scene_hash=scene,
        )
        intents = {
            td["source_cases"][s["pair"][0]]["source"]: (s["center"], np.array(s["latent"]))
            for s in tr["searches"]
        }
        visited = set()

        def read(row, root=a.root):
            path = root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("exact confined transfer artifact required")
            pins[str(path)] = row["raw_hash"]
            if root == a.root:
                visited.add(row["raw_path"])
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def inference():
            counts["own_neural"] += 1

        summaries = {}
        for bank, (root, source, sd) in banks.items():
            worlds = r["results"][bank]["rows"]
            if len(worlds) != 16 or [w["case"] for w in worlds] != list(range(16)):
                raise ValueError("all16 ordered contexts retained required")
            outcomes = []
            for i, (world, case) in enumerate(zip(worlds, sd["cases"], strict=True)):
                prior = source["worlds"][i]
                boundary = read(prior["boundaries"][1]["boundary"], root)
                applied = case["source"] in intents
                center, latent = intents.get(case["source"], (100, np.zeros((2, 6))))
                repetitions, raw_repetitions = [], []
                for repeat, row in enumerate((world, world["repeat"])):
                    parent = a.root / bank / f"case{i}-repeat{repeat}"
                    wp = parent / "complete.json"
                    if json.loads(wp.read_text()) != {
                        k: v for k, v in row.items() if k != "repeat"
                    }:
                        raise ValueError("sealed branch record differs")
                    pins[str(wp)] = digest(wp)
                    data = {}
                    for key, filename in (
                        ("kick", "kick"),
                        ("kick_terminal", "kick-terminal"),
                        ("stand", "stand"),
                    ):
                        if row[key] is None:
                            continue
                        if (a.root / row[key]["raw_path"]) != parent / (filename + ".npz"):
                            raise ValueError("branch phase must bind its own raw")
                        data[key] = read(row[key])
                    if not applied:
                        identical(data["kick"], read(prior["kicks"][2]["result"], root))
                        identical(data["kick_terminal"], read(prior["kicks"][2]["terminal"], root))
                        identical(data["stand"], read(prior["stands"][2]["result"], root))
                    result = audit_successor_contact(
                        auditor,
                        observer,
                        replay,
                        actor.base,
                        row,
                        read=read,
                        entry=snapshot_fields(boundary, "after_"),
                        previous=boundary["previous_actual_action"],
                        motion=motions[case["source"]],
                        center=center,
                        latent=latent,
                        pure=not applied,
                        count_inference=inference,
                    )
                    if (
                        row["teacher_applied"] != applied
                        or row["bank"] != bank
                        or row["case"] != i
                        or row["quality"] != result["quality"]
                    ):
                        raise ValueError("branch applicability or physical outcome differs")
                    counts["native"] += result["native"]
                    counts["neural"] += result["neural"]
                    counts["branches"] += 1
                    repetitions.append(result)
                    raw_repetitions.append(data)
                if repetitions[0] != repetitions[1] or set(raw_repetitions[0]) != set(
                    raw_repetitions[1]
                ):
                    raise ValueError("complete repeated branch outcomes differ")
                for key in raw_repetitions[0]:
                    identical(raw_repetitions[0][key], raw_repetitions[1][key])
                result = dict(
                    case=i, **repetitions[0], baseline_quality=prior["quality_by_kick"][2]
                )
                outcomes.append(result)
                write_once(a.output / f"{bank}-case{i}.json", result)
            summary = dict(
                quality=sum(x["quality"] for x in outcomes),
                body_healthy=sum(x["stand_healthy"] for x in outcomes),
                stand_stable=sum(x["stand_stable"] for x in outcomes),
                baseline_quality=source["summary"]["quality_by_kick"][2],
            )
            if any(r["results"][bank][key] != value for key, value in summary.items()):
                raise ValueError("bank aggregate outcomes differ")
            summaries[bank] = dict(
                **summary,
                rescue=[x["case"] for x in outcomes if x["quality"] and not x["baseline_quality"]],
                regression=[
                    x["case"] for x in outcomes if not x["quality"] and x["baseline_quality"]
                ],
            )
        if (
            counts["branches"] != 64
            or counts["native"] != r["actual_native_steps"]
            or replay.native_steps != counts["native"]
            or counts["neural"] != r["actual_neural_rows"]
            or counts["own_neural"] != counts["neural"]
            or visited != {str(f.relative_to(a.root)) for f in a.root.rglob("*.npz")}
        ):
            raise ValueError("complete transfer raw/physics/neural work coverage differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_SUCCESSOR_TRANSFER_ALL_NATIVE_NEURAL_IK_AUDIT_PASS",
                input_file_hashes=pins,
                source_head=a.expected_head,
                summaries=summaries,
                counts=counts,
                own_native_steps=replay.native_steps,
                optimizer_updates=0,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
                uninterrupted_game=False,
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
