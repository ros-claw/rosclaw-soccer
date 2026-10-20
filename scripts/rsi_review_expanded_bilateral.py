"""Fixed dispatch or independent native review of a completed expanded40 shard.

All complete failures are reviewed too. A dispatch is not physical verification
or permission to train/promote. No encoder/decoder/planner math replay.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dispatch", "native"), required=True)
    parser.add_argument(
        "--producer", nargs=2, action="append", required=True, metavar=("UNIT", "ROOT")
    )
    for key in ("world", "motion-csv", "core-source", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("expected-head", "expected-producer-head"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--expected-producer-hash")
    parser.add_argument("--native-unit-prefix")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists():
        raise ValueError("sealed source and unused output required")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("dirty review source")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "src", "scripts"], text=True
    ).splitlines()
    if any(p.endswith(".py") for p in untracked):
        raise ValueError("unsealed review Python source")
    if (
        args.mode == "native"
        and (len(args.producer) != 1 or args.expected_producer_hash is None)
        or args.mode == "dispatch"
        and (
            len(args.producer) != 4
            or args.native_unit_prefix is None
            or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,90}", args.native_unit_prefix) is None
        )
    ):
        raise ValueError("exact mode-specific closed producer inputs required")
    from scripts.rsi_expanded_bilateral_review_common import admit_closed
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    pins = {}
    sources = []
    shards = set()
    units = set()
    for unit, path in args.producer:
        root = Path(path)
        if args.mode == "native" and digest(root / "complete.json") != args.expected_producer_hash:
            raise ValueError("closed producer complete identity differs")
        declaration, complete, original_pins = admit_closed(unit, root, args.expected_producer_head)
        if complete["shard"] in shards or unit in units:
            raise ValueError("unique producer shard/unit required")
        pins.update(original_pins)
        sources.append((unit, root, declaration, complete))
        shards.add(complete["shard"])
        units.add(unit)
    if args.mode == "dispatch" and shards != set(range(4)):
        raise ValueError("whole40 required before native dispatch")
    check_pins(pins)
    for p in (args.world, args.motion_csv):
        if pins.get(str(p)) != digest(p):
            raise ValueError("original compiled world and source motion required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.expanded_bilateral_" + args.mode + ".v1",
            source_head=head,
            input_file_hashes=pins,
            producer_complete_hashes={
                str(root / "complete.json"): digest(root / "complete.json")
                for _, root, _, _ in sources
            },
            new_neural_calls=0,
            neural_math_replayed=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    results = []
    launches = []
    completed_native = 0
    progress = {}
    try:
        if args.mode == "dispatch":
            for unit, root, _, complete in sorted(sources, key=lambda x: x[3]["shard"]):
                shard = complete["shard"]
                name = f"{args.native_unit_prefix}-c{shard}"
                output = args.output / f"native-c{shard}"
                command = [
                    "systemd-run",
                    "--user",
                    "--unit=" + name,
                    "--no-block",
                    "--property=Type=oneshot",
                    "--property=RemainAfterExit=yes",
                    "--property=MemoryMax=6G",
                    "--property=NoNewPrivileges=yes",
                    "--working-directory=" + str(Path.cwd()),
                    "--setenv=PYTHONPATH=src:.:" + str(args.core_source),
                    "--setenv=OMP_NUM_THREADS=1",
                    "--setenv=OPENBLAS_NUM_THREADS=1",
                    "--setenv=MKL_NUM_THREADS=1",
                    "--setenv=CUDA_VISIBLE_DEVICES=",
                    sys.executable,
                    "scripts/rsi_review_expanded_bilateral.py",
                    "--mode",
                    "native",
                    "--expected-head",
                    head,
                    "--expected-producer-head",
                    args.expected_producer_head,
                    "--producer",
                    unit,
                    str(root),
                    "--expected-producer-hash",
                    digest(root / "complete.json"),
                    "--world",
                    str(args.world),
                    "--motion-csv",
                    str(args.motion_csv),
                    "--core-source",
                    str(args.core_source),
                    "--output",
                    str(output),
                ]
                subprocess.run(command, check=True)
                launch = dict(
                    unit=name + ".service",
                    output=str(output),
                    producer_complete_hash=digest(root / "complete.json"),
                )
                launches.append(launch)
                write_once(args.output / f"launch-c{shard}.json", launch)
            check_pins(pins)
            write_once(
                args.output / "complete.json",
                dict(
                    status="ALL40_NATIVE_REVIEW_DISPATCHED_NOT_VERIFIED",
                    launched_reviews=launches,
                    new_native_steps=0,
                    new_neural_calls=0,
                    goal_complete=False,
                    promotion_authorized=False,
                    hardware_authorized=False,
                ),
            )
            print("EXPANDED40_NATIVE_REVIEWS_QUEUED", flush=True)
            return
        import mujoco
        import numpy as np

        from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
        from rosclaw_soccer.sim.contracts import hash_json
        from scripts.rsi_expanded_bilateral_review_common import replay_case

        _, root, declaration, complete = sources[0]
        choices = [Path(p) for p in pins if Path(p).name == "entry-affordance.json"]
        if len(choices) != 1:
            raise ValueError("one original right-foot historical prior required")
        stored = json.loads(choices[0].read_text())
        affordance = SkillEntryAffordance(
            tuple(stored["ball_offset_xy_m"]),
            stored["historical_frame"],
            stored["historical_forward_speed_mps"],
            stored["historical_artifact_hash"],
        )
        if affordance.contract_hash != stored["contract_hash"]:
            raise ValueError("right prior differs")
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        model = mujoco.MjModel.from_binary_path(str(args.world))
        for row in complete["results"]:
            index = row["course"]["course_index"]
            progress = {}
            result = replay_case(
                model,
                root / f"course{index:02d}",
                row,
                poses=poses,
                affordance=affordance,
                evidence_hash=hash_json(declaration),
                world_hash=digest(args.world),
                progress=progress,
            )
            results.append(result)
            completed_native += result["additional_native_steps"]
            progress = {}
            write_once(args.output / f"course{index:02d}-complete.json", result)
            print("EXPANDED40_NATIVE_EXACT", index, result["additional_native_steps"], flush=True)
        if (
            completed_native
            != complete["forecast_native_steps"] + complete["endpoint_native_steps"]
        ):
            raise ValueError("full ten-case native review accounting differs")
        summaries = dict(
            baseline_qualified=sum(r["baseline_score"]["qualified"] for r in results),
            selected_qualified=sum(r["selected_score"]["qualified"] for r in results),
            rescues=sum(r["old_teacher_failure_rescued"] for r in results),
            causal_first_touch_rescues=sum(r["causal_first_touch_rescue"] for r in results),
            regressions=sum(r["old_teacher_success_lost"] for r in results),
            added_unsafe=sum(r["added_unsafe_vs_old_teacher"] for r in results),
        )
        if any(complete[k] != v for k, v in summaries.items()):
            raise ValueError("whole shard outcome summary differs from native case results")
        check_pins(pins)
        write_once(
            args.output / "complete.json",
            dict(
                shard=complete["shard"],
                results=results,
                additional_native_steps=completed_native,
                new_neural_calls=0,
                neural_math_replayed=False,
                optimizer_updates=0,
                **summaries,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        body = progress.get("active_body")
        origin = progress.get("active_origin_steps", 0)
        if body is not None:
            import numpy as np

            np.savez_compressed(
                args.output / "last-native-review-state.npz",
                qpos=body.data.qpos.copy(),
                qvel=body.data.qvel.copy(),
                time_sec=np.array([body.data.time]),
            )
        write_once(
            args.output / "failure.json",
            dict(
                error=repr(error),
                completed_courses=len(results),
                complete_case_native_steps=completed_native,
                current_case_completed_steps=progress.get("completed_steps", 0),
                current_case_incomplete_attempted_steps=body.attempted_steps - origin
                if body is not None
                else 0,
                current_case_incomplete_integrated_steps=body.integrated_steps - origin
                if body is not None
                else 0,
                launched_reviews=launches,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
