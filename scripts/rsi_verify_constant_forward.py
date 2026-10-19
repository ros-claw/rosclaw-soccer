"""Whole18 raw/current-law readback followed by independent recorded physics.

Recorded targets, not reexecuted NN/navigation. No promotion or learning.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_parity import original_arrays
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_constant_forward_approach import approach_review, require_scope
from scripts.rsi_postcontact_release_transfer import paired_summary
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if (
        args.output.exists()
        or (args.root / "failure.json").exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("complete predecessor, unused output and exact frozen reviewer required")
    require_terminal(args.unit)
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts")
        for p in folder.rglob("*.py")
    }

    def pin(name, expected=None):
        path = Path(name)
        h = digest(path)
        if (
            path.is_symlink()
            or (expected is not None and h != expected)
            or (str(path) in pins and pins[str(path)] != h)
        ):
            raise ValueError("same whole immutable approach input required")
        pins[str(path)] = h

    def sealed(name):
        pin(name)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole approach input seal required")
        return d

    def arrays(witness):
        pin(witness["path"], witness["sha256"])
        with np.load(witness["path"], allow_pickle=False) as archive:
            return dict(archive)

    c, d = sealed(args.root / "complete.json"), sealed(args.root / "declaration.json")
    require_scope(d["config"])
    if (
        c["schema"] != "soccer.rsi.constant_forward18_complete.v1"
        or d["schema"] != "soccer.rsi.constant_forward18_declaration.v1"
        or c["declaration_hash"] != d["report_hash"]
        or len(c["rows"]) != 18
        or len(d["jobs"]) != 18
        or Path(d["config"]["output_root"]) != args.root
        or any(
            type(c.get(k)) is not int or c[k] != v
            for k, v in dict(
                new_native_steps=54000,
                forecast_steps=0,
                new_optimizer_updates=0,
                executed_control_foundation_pairs=5400,
                baseline_rollouts_reused=18,
            ).items()
        )
        or any(
            c.get(k) is not False
            for k in (
                "teacher_qualified",
                "training_authorized",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("whole eighteen original approach bodies required")
    for n, h in d["input_file_hashes"].items():
        pin(n, h)
    stage = sealed(d["config"]["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    local = Path(d["config"]["local_root"])
    checks, cases = [], []
    for i, (row, job) in enumerate(zip(c["rows"], d["jobs"], strict=True)):
        if (
            row != sealed(args.root / f"case-{i}-complete.json")
            or row["index"] != i
            or any(row[k] != v for k, v in job.items())
            or row["partition"]
            != ("CONSUMED_LOCAL_DEVELOPMENT" if i < 12 else "CONSUMED_ORIGINAL_RETENTION")
            or type(row["source_index"]) is not int
            or not 0 <= row["source_index"] < 6
        ):
            raise ValueError("all ordered original approach jobs and failures required")
        parent = native["results"][row["source_index"]]
        cw, world = parent["witnesses"]["commitment"], parent["witnesses"]["world"]
        pin(cw["path"], cw["sha256"])
        pin(world["path"], world["sha256"])
        commitment = json.loads(Path(cw["path"]).read_text())
        if (
            parent["receipt_hash"] != row["source_receipt_hash"]
            or commitment["motor_policy_hash"] != row["original_policy_hash"]
        ):
            raise ValueError("same original full model/receipt required")
        if i < 12:
            b = sealed(local / f"pair-{i}-original_control-complete.json")
            if b["report_hash"] != row["baseline_hash"]:
                raise ValueError("same original local control required")
            baseline = arrays(b["arrays"])
        else:
            if parent["receipt_hash"] != row["baseline_hash"]:
                raise ValueError("same original consumed retention required")
            baseline = arrays(parent["witnesses"]["trace"])
        trace = arrays(row["arrays"])
        original_arrays(trace)
        control = approach_review(trace, baseline)
        outcome = label_first_touch(trace).to_dict()
        if (
            control != row["control_review"]
            or outcome != row["outcome"]
            or label_first_touch(baseline).to_dict() != row["baseline_outcome"]
            or row["causal_review"].get("constant_forward_approach") is not True
            or row["causal_review"].get("original_precontact_forward_law_unchanged") is not False
        ):
            raise ValueError("independent raw navigation/M0 differs")
        checks.append(
            dict(
                index=i,
                source_case_hash=row["report_hash"],
                outcome=outcome,
                control_review=control,
            )
        )
        cases.append((row, world, commitment))
    summary = paired_summary(c["rows"])
    summary["approach_mechanism_locally_promising"] = summary.pop("fixed_action_locally_promising")
    if any(c.get(k) != v for k, v in summary.items()):
        raise ValueError("whole independently rederived paired summary differs")
    if args.preflight:
        print("WHOLE18_APPROACH_RAW_READY_NO_NEW_PHYSICS", flush=True)
        return
    args.output.mkdir()
    declaration = dict(
        schema="soccer.rsi.constant_forward18_verification_declaration.v1",
        source_head=args.source_head,
        input_file_hashes=pins,
        producer_complete_hash=c["report_hash"],
        whole_raw_rows=checks,
        physical_step_budget=54000,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
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
                raise RuntimeError("fixed whole18 physics budget expired")
            ph = commitment["physics"]
            model = mujoco.MjModel.from_binary_path(world["path"])
            audit = RecordedEarlyEpisode(
                model,
                world_hash=world["sha256"],
                kp=np.asarray(ph["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(ph["canonical_pd_kd"], dtype=np.float64),
                course=tuple(row["course"]),
                defaults=np.asarray(G1SonicRunupController.default_angles, dtype=np.float64),
            )
            review = audit.verify_unforced_control(arrays(row["arrays"]))
            if review["actual_physical_steps"] != 3000 or any(
                not np.isfinite(e) or e > 1e-8 for e in review["maximum_absolute_errors"].values()
            ):
                raise ValueError("whole finite strict physical alignment required")
            result = dict(index=index, source_case_hash=row["report_hash"], physical_review=review)
            result["report_hash"] = hash_json(result)
            write_once(args.output / f"case-{index}-complete.json", result)
            rows.append(result)
            print("CONSTANT_FORWARD18_PHYSICS_CASE_COMPLETE", index, flush=True)
        if any(digest(n) != h for n, h in pins.items()):
            raise ValueError("whole original input/source chain changed during verification")
        result = dict(
            schema="soccer.rsi.constant_forward18_verification_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            actual_physical_steps=sum(r["physical_review"]["actual_physical_steps"] for r in rows),
            new_neural_inferences=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            training_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
            independent_neural_policy_qualified=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(args.output / "complete.json", result)
    except BaseException as error:
        write_once(
            args.output / "failure.json",
            dict(
                current_case=index,
                completed_cases=len(rows),
                current_attempted_steps=0 if audit is None else audit.attempted_steps,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
