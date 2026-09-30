"""SIM-only resource recovery, never model tuning or a second fresh exam.

Frozen v310 runner/helper and existing pool allocation are used unchanged.
Alternative A6000s must first byte-reproduce known consumed parent/candidate
physics. Missing work is selected by absent reports, never observed outcomes.
"""

from __future__ import annotations

import argparse
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once
from scripts.rsi_train_protected_online_motor_v308 import _head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root",
        "recovery-root",
        "frozen-source",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "neural-model",
        "parent-motor-policy",
        "stable-root",
        "core-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    args = parser.parse_args()
    commitment = json.loads((args.root / "commitment.json").read_text())
    runner = args.frozen_source / "scripts/rsi_isaac_vector_first_touch.py"
    if (
        _head(args.frozen_source) != commitment["source_commit"]
        or _head(args.core_root) != commitment["core_commit"]
        or hash_bytes(runner.read_bytes()) != commitment["runner_hash"]
        or json.loads(args.neural_model.read_text())["model_hash"]
        != commitment["selected_model_hash"]
        or json.loads(args.parent_motor_policy.read_text())["policy_hash"]
        != commitment["parent_policy_hash"]
        or hash_bytes(args.g1_usd.read_bytes()) != commitment["asset_hash"]
        or len(set(args.gpus)) != len(args.gpus)
        or not args.gpus
        or any(g not in (1, 2, 3) for g in args.gpus)
    ):
        parser.error(
            "unchanged frozen source/core/model/assets and explicit alternative GPUs required"
        )
    quarantine = json.loads(
        (
            args.frozen_source / "docs/rsi/protocols/neural-motor-fresh-quarantine-v307.json"
        ).read_text()
    )
    args.recovery_root.mkdir(parents=True, exist_ok=False)
    (args.recovery_root / "logs").mkdir()
    assignment: list[dict[str, Any]] = []
    for seed in quarantine["seeds"]:
        for lane in quarantine["lanes_per_seed"]:
            needed = [
                arm
                for arm, kind in (
                    ("reproduction", "parent"),
                    ("champion", "actor"),
                    ("candidate", "actor"),
                )
                if not (args.root / f"seed{seed}-lane{lane}-{arm}-{kind}/report.json").is_file()
            ]
            if needed:
                for arm in needed:
                    kind = "parent" if arm == "reproduction" else "actor"
                    folder = args.root / f"seed{seed}-lane{lane}-{arm}-{kind}"
                    if folder.exists():
                        raise ValueError(
                            "preserve and review incomplete output before resource recovery"
                        )
                    log = args.root / "logs" / f"seed{seed}-lane{lane}-{arm}-{kind}.log"
                    if log.exists():
                        raise ValueError("archive original failed startup log before recovery")
                assignment.append(
                    dict(
                        seed=seed,
                        lane=lane,
                        missing=needed,
                        gpu=args.gpus[len(assignment) % len(args.gpus)],
                    )
                )
    receipt = dict(
        schema="soccer.rsi.fresh_resource_recovery.v1",
        original_commitment_hash=hash_json(commitment),
        recovery_source_hash=hash_bytes(Path(__file__).read_bytes()),
        frozen_helper_hash=hash_bytes(
            (
                args.frozen_source / "scripts/rsi_collect_approach_lateral_tracking_v286.py"
            ).read_bytes()
        ),
        assignments=assignment,
        selection_rule="absent reports only, not outcomes",
        model_changed=False,
        physical_configuration_changed=False,
        promotion_authorized=False,
    )
    write_once(args.recovery_root / "commitment.json", receipt)

    helper_path = Path(inspect.getfile(_run))
    if hash_bytes(helper_path.read_bytes()) != receipt["frozen_helper_hash"]:
        raise ValueError("imported execution helper differs from frozen exam source")

    def call(
        root: Path, seed: int, lane: int, gpu: int, arm: str, kind: str, **extra: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return _run(
            root=root,
            runner=runner,
            isaac_python=args.isaac_python,
            g1_usd=args.g1_usd,
            model_root=args.model_root,
            actor=args.late_swing_policy,
            seed=seed,
            lane=lane,
            gpu=gpu,
            arm=arm,
            kind=kind,
            gain=1.2,
            negative_only=True,
            core_root=args.core_root,
            **extra,
        )

    def equivalence(gpu: int) -> dict[str, Any]:
        # Known consumed seed, never any member of the fresh pool.
        seed, lane = 20261177, 0
        arm = f"equivalence-gpu{gpu}"
        parent, _ = call(args.recovery_root, seed, lane, gpu, arm, "parent")
        report, _ = call(
            args.recovery_root,
            seed,
            lane,
            gpu,
            arm,
            "actor",
            motor_online=args.neural_model,
            parent_report_override=args.recovery_root
            / f"seed{seed}-lane{lane}-{arm}-parent/report.json",
        )
        old = json.loads(
            (args.stable_root / f"seed{seed}-lane{lane}-stable-actor/report.json").read_text()
        )
        old_parent = json.loads(
            (
                args.stable_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
            ).read_text()
        )
        if any(
            report[k] != old[k] or parent[k] != old_parent[k]
            for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
        ):
            raise ValueError("alternative GPU changed known physics; fresh recovery prohibited")
        print(f"RECOVERY_GPU_EQUIVALENCE_PASSED gpu={gpu}", flush=True)
        return dict(
            gpu=gpu,
            parent_report_hash=parent["report_hash"],
            candidate_report_hash=report["report_hash"],
            body_and_ball_byte_equal=True,
        )

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        checks = list(pool.map(equivalence, args.gpus))
    write_once(args.recovery_root / "equivalence.json", dict(checks=checks))

    def recover(gpu: int) -> list[dict[str, Any]]:
        results = []
        for task in assignment:
            if task["gpu"] != gpu:
                continue
            seed, lane = task["seed"], task["lane"]
            parent_path = args.root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
            for arm in task["missing"]:
                kind = "parent" if arm == "reproduction" else "actor"
                backend = (
                    {}
                    if kind == "parent"
                    else dict(
                        parent_report_override=parent_path,
                        **(
                            {"motor_policy": args.parent_motor_policy}
                            if arm == "champion"
                            else {"motor_online": args.neural_model}
                        ),
                    )
                )
                report, _ = call(args.root, seed, lane, gpu, arm, kind, **backend)
                results.append(
                    dict(seed=seed, lane=lane, arm=arm, gpu=gpu, report_hash=report["report_hash"])
                )
                print(
                    f"RECOVERY_FRESH_EXECUTED seed={seed} lane={lane} arm={arm} gpu={gpu}",
                    flush=True,
                )
        return results

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        completed = [r for batch in pool.map(recover, args.gpus) for r in batch]
    receipt.update(equivalence=checks, completed=completed, complete=True)
    receipt["report_hash"] = hash_json(receipt)
    write_once(args.recovery_root / "recovery_summary.json", receipt)


if __name__ == "__main__":
    main()
