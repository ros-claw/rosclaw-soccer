"""All18 recorded postcontact bodies, not independent NN/search qualification."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_read_postcontact_transfer import check_complete
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


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
            "same complete closed inputs, unused output and frozen exact source required"
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
            raise ValueError("whole ordinary immutable physical transfer input required")
        pins[str(file)] = h

    def sealed(name):
        pin(name)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole physical transfer input seal required")
        return d

    rb = sealed(args.readback)
    c, d = sealed(args.root / "complete.json"), sealed(args.root / "declaration.json")
    adaptive, _ = check_complete(c, d, args.root)
    if (
        rb["schema"] != "soccer.rsi.postcontact_transfer_readback.v1"
        or rb["complete_hash"] != c["report_hash"]
        or rb["actual_native_steps"] != 54000
        or len(rb["rows"]) != 18
    ):
        raise ValueError("whole same closed independently read-back eighteen bodies required")
    for name, h in rb["input_file_hashes"].items():
        pin(name, h)
    stage = sealed(d["config"]["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    cases = []
    for i, (row, checked) in enumerate(zip(c["rows"], rb["rows"], strict=True)):
        if (
            row != sealed(args.root / f"case-{i}-complete.json")
            or checked["index"] != i
            or checked["source_case_hash"] != row["report_hash"]
            or checked["outcome"] != row["outcome"]
        ):
            raise ValueError("all eighteen ordered complete checked physical traces required")
        parent = native["results"][row["source_index"]]
        if parent["receipt_hash"] != row["source_receipt_hash"]:
            raise ValueError("same original native source binding required")
        w, cw = parent["witnesses"]["world"], parent["witnesses"]["commitment"]
        pin(w["path"], w["sha256"])
        pin(cw["path"], cw["sha256"])
        commitment = json.loads(Path(cw["path"]).read_text())
        if commitment["motor_policy_hash"] != row["original_policy_hash"]:
            raise ValueError("same original physical motor identity required")
        pin(row["arrays"]["path"], row["arrays"]["sha256"])
        cases.append((row, w, commitment))
    if args.preflight:
        print("WHOLE18_POSTCONTACT_PHYSICS_READY_NO_NEW_INTEGRATION", flush=True)
        return
    args.output.mkdir()
    declaration = dict(
        schema="soccer.rsi.recorded_postcontact18_declaration.v1",
        source_head=args.source_head,
        producer_complete_hash=c["report_hash"],
        readback_hash=rb["report_hash"],
        input_file_hashes=pins,
        postcontact_exit_profile=adaptive,
        physical_step_budget=54000,
        maximum_runtime_seconds=1800,
        recorded_targets_consumed=True,
        neural_policy_qualification_claimed=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(args.output / "declaration.json", declaration)
    rows, audit, index = [], None, None
    deadline = time.monotonic() + 1800
    try:
        import mujoco

        from rosclaw_soccer.physics.recorded_early_episode import RecordedEarlyEpisode
        from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController

        for index, (row, world, commitment) in enumerate(cases):
            audit = None
            if time.monotonic() >= deadline:
                raise RuntimeError("whole18 physical diagnostic deadline exhausted")
            model = mujoco.MjModel.from_binary_path(world["path"])
            ph = commitment["physics"]
            audit = RecordedEarlyEpisode(
                model,
                world_hash=world["sha256"],
                kp=np.asarray(ph["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(ph["canonical_pd_kd"], dtype=np.float64),
                course=tuple(row["course"]),
                defaults=np.asarray(G1SonicRunupController.default_angles, dtype=np.float64),
            )
            with np.load(row["arrays"]["path"], allow_pickle=False) as a:
                trace = dict(a)
            review = (
                audit.verify_postcontact_exit(trace)
                if adaptive
                else audit.verify_postcontact_release(trace)
            )
            result = dict(index=index, source_case_hash=row["report_hash"], physical_review=review)
            result["report_hash"] = hash_json(result)
            write_once(args.output / f"case-{index}-complete.json", result)
            rows.append(result)
            print("WHOLE18_POSTCONTACT_RECORDED_PHYSICS_COMPLETE", index, flush=True)
        if any(digest(n) != h for n, h in pins.items()):
            raise ValueError("physical source/input changed")
        result = dict(
            schema="soccer.rsi.recorded_postcontact18_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            actual_physical_steps=sum(r["physical_review"]["actual_physical_steps"] for r in rows),
            new_neural_inferences=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            training_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
            independent_neural_or_search_qualified=False,
        )
        if result["actual_physical_steps"] != 54000:
            raise ValueError("all eighteen complete physical bodies required")
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
                current_case=index,
                completed_cases=len(rows),
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
