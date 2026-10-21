"""Independently read all16 kick/standing lifecycle native and neural evidence."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if (
        a.output.exists()
        or digest(rp) != a.hash
        or (a.root / "failed.json").exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("closed exact lifecycle and unused frozen audit output required")
    r, d = [json.loads(f.read_text()) for f in (rp, dp)]
    if (
        r["status"] != "PAID_NEURAL_STANDING_LIFECYCLE_CLOSED_PENDING_AUDIT"
        or r["source_head"] != d["source_head"]
        or r["declaration_hash"] != digest(dp)
        or r["course_bank"] != d["course_bank"]
        or d["course_bank"] not in ("train", "dev")
        or r["variant"] != d["variant"]
        or d["variant"] not in (0, 1, 2)
        or r["skills"] != d["skills"]
        or d["skills"] not in (1, 3)
        or len(d["cases"]) != 16
        or d["independent_world_repeats"] != 2
        or d["standing_controls"] != 100
        or d["settle_max_controls"] != 50
        or d["no_motion_padding"] is not True
        or d["body_resets"] != 0
        or r["all_first_prefix_and_full_repeat_arrays_exact"] is not True
        or any(
            d[key] is not False or r[key] is not False
            for key in (
                "old_exam_result",
                "fresh_exam",
                "team_game",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or d["optimizer_updates"] != 0
        or r["optimizer_updates"] != 0
    ):
        raise ValueError("complete declared all16 new standing lifecycle required")
    from rosclaw_soccer.rsi.paid_recovery_proof import recovery_proof, unique_proof_path

    source_root = Path(d["source_root"])
    sd, sp = recovery_proof(
        source_root,
        d["source_hash"],
        Path(d["source_audit_root"]),
        d["source_audit_hash"],
        cases=d["cases"],
        course_bank=d["course_bank"],
    )
    if (
        d["actor_checkpoint_hash"] != sd["checkpoint_hash"]
        or d["actor_training_receipt_hash"] != sd["training_receipt_hash"]
    ):
        raise ValueError("frozen typed motor lineage differs")
    source = json.loads((source_root / "complete.json").read_text())
    pins = dict(r["input_file_hashes"])
    if any(pins.get(path) != h for path, h in sp.items()):
        raise ValueError("source proof was not committed by producer")
    pins.update({str(rp): a.hash, str(dp): digest(dp)})
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            own_native_steps=0,
            hardware_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, own_neural=0, worlds=0)
    observer = None
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_balance_recovery import transfer_reference_prefix
        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_neural_stand_audit import StandingObservationAuditor
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor
        from rosclaw_soccer.rsi.paid_standing_lifecycle_audit import audit_standing_lifecycle

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        auditor, motions = open_paid_auditor(pins, d["cases"])
        observer = StandingObservationAuditor(auditor)
        policy = next(Path(path) for path in pins if path.endswith("/ckp/policy_30000.onnx"))
        scene = next(
            h for path, h in pins.items() if path.endswith("/cached-standard/declared-scene.xml")
        )
        actor, _ = load_contact_residual_actor(
            policy,
            checkpoint=unique_proof_path(pins, sd["checkpoint_hash"]),
            checkpoint_hash=sd["checkpoint_hash"],
            training_receipt=unique_proof_path(pins, sd["training_receipt_hash"]),
            training_receipt_hash=sd["training_receipt_hash"],
            scene_hash=scene,
        )
        visited = set()

        def inference():
            counts["own_neural"] += 1

        def read(row, root=a.root):
            path = root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("confined exact raw trajectory required")
            pins[str(path)] = row["raw_hash"]
            if root == a.root:
                visited.add(row["raw_path"])
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        worlds = r["worlds"]
        if len(worlds) != 16 or [w["case"] for w in worlds] != list(range(16)):
            raise ValueError("all16 ordered cases must be retained")
        results = []
        for i, (world, case) in enumerate(zip(worlds, d["cases"], strict=True)):
            motion = motions[case["source"]]
            first = source["results"]["learned"]["worlds"][i]["skills"][0]["result"]
            prefix = transfer_reference_prefix(read(first, source_root), horizon=len(motion["q"]))
            replicas, raws = [], []
            for repeat, artifacts in enumerate((world["artifacts"], world["repeat_artifacts"])):
                parent = a.root / f"case{i}-repeat{repeat}"
                wp = parent / "complete.json"
                actual = json.loads(wp.read_text())
                pins[str(wp)] = digest(wp)
                if actual["artifacts"] != artifacts or (
                    repeat == 0
                    and actual != {k: v for k, v in world.items() if k != "repeat_artifacts"}
                ):
                    raise ValueError("sealed world differs")
                links = {Path(x["raw_path"]).stem: x for x in artifacts}
                if len(links) != len(artifacts) or any(
                    (a.root / x["raw_path"]).parent != parent for x in artifacts
                ):
                    raise ValueError("unique raw artifacts confined to own world required")
                rows = [x[key] for x in actual["kicks"] for key in ("result", "terminal")]
                rows += [x["result"] for x in actual["stands"]]
                rows += [row for b in actual["boundaries"] for row in b.values() if row is not None]
                for row in rows:
                    if any(
                        row[key] != links[Path(row["raw_path"]).stem][key]
                        for key in ("raw_path", "raw_hash")
                    ):
                        raise ValueError("phase raw link lost its hash binding")
                data = {key: read(row) for key, row in links.items()}
                result = audit_standing_lifecycle(
                    auditor,
                    observer,
                    actor,
                    actual,
                    data,
                    motion,
                    prefix,
                    variant=d["variant"],
                    skills=d["skills"],
                    count_inference=inference,
                )
                for key in ("native", "neural"):
                    counts[key] += result[key]
                counts["worlds"] += 1
                replicas.append(result)
                raws.append(data)
            if replicas[0] != replicas[1] or set(raws[0]) != set(raws[1]):
                raise ValueError("independent full world outcomes differ")
            for key in raws[0]:
                identical(raws[0][key], raws[1][key])
            results.append(replicas[0])
            print(json.dumps(dict(case=i, status=replicas[0]["status"])), flush=True)
        summary = dict(stable=sum(w["stable"] for w in results))
        for key in ("quality_by_kick", "stand_healthy", "stand_certificates"):
            summary[key] = [
                sum(len(w[key]) > k and w[key][k] for w in results) for k in range(d["skills"])
            ]
        if (
            summary != r["summary"]
            or counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_neural_rows"]
            or counts["own_neural"] != counts["neural"]
            or visited != {str(f.relative_to(a.root)) for f in a.root.rglob("*.npz")}
        ):
            raise ValueError("aggregate physics/neural accounting or complete raw coverage differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_NEURAL_STANDING_LIFECYCLE_NATIVE_AND_NEURAL_AUDIT_PASS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                summary=summary,
                counts=counts,
                own_kinematic_forward_calls=observer.evaluations,
                own_native_steps=0,
                optimizer_updates=0,
                hardware_authorized=False,
                promotion_authorized=False,
                old_exam_result=False,
                team_game=False,
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
                own_kinematic_forward_calls=observer.evaluations if observer is not None else 0,
                own_native_steps=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
