"""Verify252 recorded SIM bodies:238 new integrations,14 physical proofs reused.

Recorded targets, not NN/search reconstruction or teacher admission. Reused
physics is explicitly excluded from the714000 new-step budget.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_read_postcontact_full_bank import check_complete
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_full_physical_review(review):
    errors = review.get("maximum_absolute_errors", {})
    if (
        type(review.get("actual_physical_steps")) is not int
        or review["actual_physical_steps"] != 3000
        or not errors
        or any(
            type(v) not in (float, int) or not np.isfinite(v) or v < 0 or v > 1e-8
            for v in errors.values()
        )
    ):
        raise ValueError("whole3000-step finite matching physical review required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--readback", required=True, type=Path)
    parser.add_argument("--readback-unit", required=True)
    parser.add_argument("--readback-file-hash", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if (
        args.output.exists()
        or (args.root / "failure.json").exists()
        or digest(args.readback) != args.readback_file_hash
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError(
            "whole closed raw bytes, unused output and clean frozen physical verifier required"
        )
    require_terminal(args.unit)
    require_terminal(args.readback_unit)
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts")
        for p in folder.rglob("*.py")
    }

    def pin(name, expected=None):
        file = Path(name)
        h = digest(file)
        if (
            file.is_symlink()
            or (expected is not None and h != expected)
            or (str(file) in pins and pins[str(file)] != h)
        ):
            raise ValueError("whole original ordinary physical bank inputs required")
        pins[str(file)] = h

    def sealed(name, expected=None):
        pin(name, expected)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole bank physics input seal required")
        return d

    rb = sealed(args.readback)
    c, d = sealed(args.root / "complete.json"), sealed(args.root / "declaration.json")
    check_complete(c, d, args.root)
    if (
        rb["schema"] != "soccer.rsi.postcontact_full_bank_readback.v1"
        or rb["complete_hash"] != c["report_hash"]
        or len(rb["contexts"]) != 18
        or rb["comparison_rollouts"] != 252
        or rb["reused_execution_count"] != 14
        or rb["actual_source_native_steps"] != 714000
    ):
        raise ValueError("same whole252 source-authenticated raw matrix required")
    for name, h in rb["input_file_hashes"].items():
        pin(name, h)
    cfg = d["config"]
    require_terminal(cfg["release_physics_unit"])
    oldroot = Path(cfg["release_physics_root"])
    old = sealed(oldroot / "complete.json", cfg["release_physics_complete_file_hash"])
    od = sealed(oldroot / "declaration.json")
    if (
        old["schema"] != "soccer.rsi.recorded_postcontact14_complete.v1"
        or old["declaration_hash"] != od["report_hash"]
        or od["producer_complete_hash"] != d["reused_authority_hash"]
        or len(old["rows"]) != 14
        or old["actual_physical_steps"] != 42000
    ):
        raise ValueError("all14 exact independently replayed original authority bodies required")
    for name, h in od["input_file_hashes"].items():
        pin(name, h)
    stage = sealed(cfg["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    commitments = []
    for parent in native["results"]:
        w = parent["witnesses"]["commitment"]
        pin(w["path"], w["sha256"])
        commitments.append(json.loads(Path(w["path"]).read_text()))
    jobs = []
    for ci, (context, checked, job) in enumerate(
        zip(c["contexts"], rb["contexts"], d["jobs"], strict=True)
    ):
        if (
            context != sealed(args.root / f"context-{ci}-complete.json")
            or checked["context_index"] != ci
            or checked["source_context_hash"] != context["report_hash"]
            or job["index"] != ci
        ):
            raise ValueError("all18 original ordered checked bank contexts required")
        parent = native["results"][job["source_index"]]
        world = parent["witnesses"]["world"]
        pin(world["path"], world["sha256"])
        for ai, row in enumerate(context["rows"]):
            if row != sealed(args.root / f"context-{ci}-action-{ai}-complete.json"):
                raise ValueError("all252 complete ordered source rows required")
            pin(row["arrays"]["path"], row["arrays"]["sha256"])
            reused = ci == cfg["reused_authority_context_index"]
            proof = None
            if reused:
                proof = old["rows"][ai]
                if (
                    proof != sealed(oldroot / f"trial-{ai}-complete.json")
                    or proof["source_trial_hash"] != row["reused_case_hash"]
                ):
                    raise ValueError("exact original physical proof for each reused body required")
                require_full_physical_review(proof["physical_review"])
            if (
                row["execution_reused"] is not reused
                or row["source_receipt_hash"] != parent["receipt_hash"]
            ):
                raise ValueError("honest new/reused execution and native body binding required")
            jobs.append((ci, ai, row, world, commitments[job["source_index"]], proof))
    if len(jobs) != 252:
        raise ValueError("whole252 physical comparisons required")
    if args.preflight:
        print("FULL_BANK252_PHYSICS_WITH14_REUSED_READY_NO_NEW_INTEGRATION", flush=True)
        return
    args.output.mkdir()
    declaration = dict(
        schema="soccer.rsi.recorded_postcontact_full_bank_declaration.v1",
        source_head=args.source_head,
        producer_complete_hash=c["report_hash"],
        raw_readback_hash=rb["report_hash"],
        input_file_hashes=pins,
        reused_physics_hash=old["report_hash"],
        comparison_bodies=252,
        new_physical_body_budget=238,
        new_physical_step_budget=714000,
        reused_physical_bodies=14,
        reused_physical_steps=42000,
        maximum_runtime_seconds=3600,
        recorded_targets_consumed=True,
        neural_policy_qualification_claimed=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(args.output / "declaration.json", declaration)
    rows = []
    new = reused = steps = 0
    audit = current = None
    deadline = time.monotonic() + 3600
    try:
        import mujoco

        from rosclaw_soccer.physics.recorded_early_episode import RecordedEarlyEpisode
        from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController

        for ci, ai, row, world, cm, proof in jobs:
            current = dict(context_index=ci, action_index=ai)
            audit = None
            if time.monotonic() >= deadline:
                raise RuntimeError("fixed bank physical replay deadline exhausted")
            if proof is None:
                if steps + 3000 > 714000:
                    raise RuntimeError("fixed new physical step budget exhausted")
                model = mujoco.MjModel.from_binary_path(world["path"])
                ph = cm["physics"]
                audit = RecordedEarlyEpisode(
                    model,
                    world_hash=world["sha256"],
                    kp=np.asarray(ph["canonical_pd_kp"], dtype=np.float64),
                    kd=np.asarray(ph["canonical_pd_kd"], dtype=np.float64),
                    course=tuple(row["course"]),
                    defaults=np.asarray(G1SonicRunupController.default_angles, dtype=np.float64),
                )
                with np.load(row["arrays"]["path"], allow_pickle=False) as a:
                    review = audit.verify_postcontact_release(dict(a))
            else:
                review = proof["physical_review"]
            require_full_physical_review(review)
            result = dict(
                current,
                source_bank_case_hash=row["report_hash"],
                physical_review=review,
                physical_execution_reused=proof is not None,
                reused_physical_case_hash=None if proof is None else proof["report_hash"],
            )
            result["report_hash"] = hash_json(result)
            write_once(args.output / f"context-{ci}-action-{ai}-complete.json", result)
            rows.append(result)
            if proof is None:
                new += 1
                steps += review["actual_physical_steps"]
            else:
                reused += 1
            print(
                "FULL_BANK_PHYSICAL_BODY_COMPLETE",
                current,
                "physical_reused",
                proof is not None,
                flush=True,
            )
            audit = None
        if (
            new != 238
            or reused != 14
            or steps != 714000
            or any(digest(n) != h for n, h in pins.items())
        ):
            raise ValueError("whole immutable252 physical coverage and238-new-body budget required")
        result = dict(
            schema="soccer.rsi.recorded_postcontact_full_bank_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            comparison_bodies_verified=252,
            new_physical_bodies=new,
            new_physical_steps=steps,
            reused_physical_bodies=reused,
            reused_physical_steps=42000,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
            independent_neural_or_search_qualified=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(args.output / "complete.json", result)
    except BaseException as error:
        if audit is not None:
            with (args.output / "failure-state.npz").open("xb") as stream:
                np.savez_compressed(
                    stream, qpos=audit.data.qpos, qvel=audit.data.qvel, ctrl=audit.data.ctrl
                )
        write_once(
            args.output / "failure.json",
            dict(
                current=current,
                completed_new_physical_steps=steps,
                completed_new_bodies=new,
                reused_physical_bodies=reused,
                completed_comparison_bodies=len(rows),
                current_attempted_steps=0 if audit is None else audit.attempted_steps,
                current_integrated_steps=0 if audit is None else audit.integrated_steps,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
