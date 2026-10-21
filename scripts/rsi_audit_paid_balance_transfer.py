"""Independently verify complete TRAIN/DEV balance-transfer worlds."""

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
        raise ValueError("closed exact transfer and unused frozen audit output required")
    r, d = [json.loads(f.read_text()) for f in (rp, dp)]
    if (
        r["status"] != "PAID_BALANCE_FULL16_TRANSFER_CLOSED_PENDING_AUDIT"
        or d["source_head"] != r["source_head"]
        or r["course_bank"] != d["course_bank"]
        or d["course_bank"] not in ("train", "dev")
        or len(d["cases"]) != 16
        or d["arms"] != ["zero", "feedback"]
        or set(r["results"]) != {"zero", "feedback"}
        or d["independent_world_repeats"] != 2
        or d["body_resets"] != 0
        or d["second_is_reference_prefix"] is not True
        or d["all_cases_retained"] is not True
        or r["all_prefix_and_independent_world_arrays_exact"] is not True
    ):
        raise ValueError("complete declared all16 body-continuing transfer protocol required")
    from rosclaw_soccer.rsi.paid_balance_teacher_proof import balance_teacher_proof
    from rosclaw_soccer.rsi.paid_recovery_proof import recovery_proof, unique_proof_path

    teacher, td, tp, latent = balance_teacher_proof(
        Path(d["teacher_root"]),
        d["teacher_hash"],
        Path(d["teacher_audit_root"]),
        d["teacher_audit_hash"],
    )
    train_root = Path(d["training_source_root"])
    train_cases = json.loads((train_root / "declaration.json").read_text())["cases"]
    train_sd, train_pins = recovery_proof(
        train_root,
        d["training_source_hash"],
        Path(d["training_source_audit_root"]),
        d["training_source_audit_hash"],
        cases=train_cases,
    )
    if (
        train_cases != td["source_cases"]
        or teacher["input_file_hashes"].get(str(train_root / "complete.json"))
        != d["training_source_hash"]
        or teacher["input_file_hashes"].get(
            str(Path(d["training_source_audit_root"]) / "complete.json")
        )
        != d["training_source_audit_hash"]
    ):
        raise ValueError("exact teacher/source lineage required")
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
        sd["checkpoint_hash"] != train_sd["checkpoint_hash"]
        or d["actor_checkpoint_hash"] != sd["checkpoint_hash"]
        or d["actor_training_receipt_hash"] != sd["training_receipt_hash"]
        or (d["course_bank"] == "train" and d["cases"] != train_cases)
    ):
        raise ValueError("one exact same frozen kick network required")
    source = json.loads((source_root / "complete.json").read_text())
    pins = dict(r["input_file_hashes"])
    for required in (tp, train_pins, sp):
        if any(pins.get(path) != h for path, h in required.items()):
            raise ValueError("producer did not bind every required source proof")
    if r.get("declaration_hash", digest(dp)) != digest(dp):
        raise ValueError("declaration commitment changed")
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
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_balance_recovery import transfer_reference_prefix
        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_recovery_trial_audit import audit_recovery_trial
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if not np.array_equal(latent, d["common_gain"]):
            raise ValueError("one frozen common gain required, not a course winner lookup")
        auditor, motions = open_paid_auditor(pins, d["cases"])
        policy = next(Path(p) for p in pins if p.endswith("/ckp/policy_30000.onnx"))
        scene = next(
            h for p, h in pins.items() if p.endswith("/cached-standard/declared-scene.xml")
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

        def read(row, root=a.root):
            path = root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("confined exact raw artifact required")
            pins[str(path)] = row["raw_hash"]
            if root == a.root:
                visited.add(row["raw_path"])
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        summaries = {}
        for arm, gain in (("zero", np.zeros((9, 4))), ("feedback", latent)):
            worlds = r["results"][arm]["worlds"]
            if len(worlds) != 16 or [w["case"] for w in worlds] != list(range(16)):
                raise ValueError("all16 declared conditions retained in order required")
            checked_worlds = []
            for i, (world, case) in enumerate(zip(worlds, d["cases"], strict=True)):
                ref = source["results"]["learned"]["worlds"][i]
                blocks = {
                    Path(x["raw_path"]).stem: read(x, source_root)
                    for x in ref["artifacts"]
                    if Path(x["raw_path"]).stem in ("skill0", "terminal0")
                    or Path(x["raw_path"]).stem.startswith("between0-1-")
                }
                motion = motions[case["source"]]
                if len(ref["skills"]) > 1:
                    blocks["skill1"] = transfer_reference_prefix(
                        read(ref["skills"][1]["result"], source_root), horizon=len(motion["q"])
                    )
                replicas = []
                raw_replicas = []
                for repeat, artifacts in enumerate((world["artifacts"], world["repeat_artifacts"])):
                    parent = a.root / arm / f"case{i}-repeat{repeat}"
                    path = parent / "complete.json"
                    actual_world = json.loads(path.read_text())
                    pins[str(path)] = digest(path)
                    if actual_world["artifacts"] != artifacts or (
                        repeat == 0
                        and actual_world
                        != {k: v for k, v in world.items() if k != "repeat_artifacts"}
                    ):
                        raise ValueError("sealed per-world record differs")
                    links = {Path(x["raw_path"]).stem: x for x in artifacts}
                    if len(links) != len(artifacts) or any(
                        (a.root / x["raw_path"]).parent != parent for x in artifacts
                    ):
                        raise ValueError("unique artifacts confined to own world required")
                    for skill in actual_world["skills"]:
                        for key, row in (
                            (f"skill{skill['index']}", skill["result"]),
                            (f"terminal{skill['index']}", skill["terminal"]),
                        ):
                            if any(row[k] != links[key][k] for k in ("raw_path", "raw_hash")):
                                raise ValueError("motor record is not bound to its raw artifact")
                    if actual_world["recovery"] is not None and any(
                        actual_world["recovery"][k] != links["recovery"][k]
                        for k in ("raw_path", "raw_hash")
                    ):
                        raise ValueError("recovery record is not bound to its raw artifact")
                    for j, boundary in enumerate(actual_world["boundaries"]):
                        prefix = "between0-1-" if j == 0 else "post-recovery-"
                        if boundary["artifacts"] != [
                            x for x in artifacts if Path(x["raw_path"]).stem.startswith(prefix)
                        ]:
                            raise ValueError("boundary record lost physical artifacts")
                    data = {key: read(row) for key, row in links.items()}
                    result = audit_recovery_trial(
                        auditor,
                        actual_world,
                        data=data,
                        source_blocks=blocks,
                        first_boundary_status=ref["boundaries"][0]["status"]
                        if ref["boundaries"]
                        else "NOT_REACHED",
                        motion=motion,
                        actor=actor,
                        latent=gain,
                    )
                    for key in ("native", "neural", "own_neural"):
                        counts[key] += result[key]
                    counts["worlds"] += 1
                    replicas.append(result)
                    raw_replicas.append(data)
                if replicas[0] != replicas[1] or set(raw_replicas[0]) != set(raw_replicas[1]):
                    raise ValueError("complete physical world outcomes or artifact sets differ")
                for key in raw_replicas[0]:
                    identical(raw_replicas[0][key], raw_replicas[1][key])
                checked_worlds.append(replicas[0])
            summary = dict(
                stable=sum(w["stable"] for w in checked_worlds),
                recovery_healthy=sum(w["recovery_healthy"] for w in checked_worlds),
                recovery_certificates=sum(w["recovery_certificate"] for w in checked_worlds),
                quality_by_skill=[
                    sum(
                        len(w["quality_by_skill"]) > k and w["quality_by_skill"][k]
                        for w in checked_worlds
                    )
                    for k in range(3)
                ],
            )
            if any(r["results"][arm][k] != v for k, v in summary.items()):
                raise ValueError("full-course aggregate metrics differ")
            summaries[arm] = summary
        if (
            counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_neural_rows"]
            or visited != {str(f.relative_to(a.root)) for f in a.root.rglob("*.npz")}
        ):
            raise ValueError("complete native/neural accounting or raw coverage differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_BALANCE_FULL16_TRANSFER_NATIVE_AND_NEURAL_AUDIT_PASS",
                input_file_hashes=pins,
                summaries=summaries,
                counts=counts,
                own_native_steps=0,
                own_neural_rows=counts["own_neural"],
                course_bank=d["course_bank"],
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
                team_game=False,
                fresh_exam=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), counts=counts, own_native_steps=0, hardware_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
