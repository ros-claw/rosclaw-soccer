"""Declared successor: wait for four frozen experts, authenticate and diagnose.

No dynamic commands, model updates, physics, threshold changes or promotion.
"""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
from scripts.rsi_diagnose_motion_expert_coverage import diagnostic_indices
from scripts.rsi_native_context_recovery_common import check_pins, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", action="append", nargs=2, required=True, metavar=("UNIT", "ROOT"))
    p.add_argument("--expected-head", required=True)
    p.add_argument("--expected-actor-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        len(a.campaign) != 4
        or len({u for u, _ in a.campaign}) != 4
        or len({r for _, r in a.campaign}) != 4
        or a.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or subprocess.check_output(["git", "status", "--porcelain"])
    ):
        raise ValueError(
            "four unique explicit workers, frozen clean source and unused output required"
        )
    a.output.mkdir()
    pins = {
        str(f.resolve()): digest(f)
        for folder in ("src", "scripts")
        for f in Path(folder).rglob("*.py")
    }
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.motion_coverage_review.v1",
            source_head=a.expected_head,
            expected_actor_head=a.expected_actor_head,
            campaigns=a.campaign,
            own_input_file_hashes=pins,
            no_new_physics_or_optimization=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        rows, owners = {}, set()
        for unit, path in a.campaign:
            wait_for_terminal_collection(unit, wait=True)
            root = Path(path)
            d = json.loads((root / "declaration.json").read_text())
            c = json.loads((root / "complete.json").read_text())
            if (
                d["source_head"] != a.expected_actor_head
                or c["source_head"] != a.expected_actor_head
                or c["status"] != "COUNTERFACTUAL_FROZEN_EXPERT_COVERAGE_VERIFIED"
                or c["optimizer_steps"] != 0
                or c["new_native_steps"] != 144000
                or len(d["owners"]) != 1
                or len(c["results"]) != 24
                or tuple(d["indices"]) != diagnostic_indices()
                or not d["reused_consumed_holdout"]
                or d["fresh_test"]
                or c["hardware_authorized"]
                or c["promotion_authorized"]
            ):
                raise ValueError("closed diagnostic campaign differs")
            owner = d["owners"][0]
            if owner not in range(1, 5) or owner in owners:
                raise ValueError("exactly four different expert owners required")
            owners.add(owner)
            for key, value in d["input_file_hashes"].items():
                if key in pins and pins[key] != value:
                    raise ValueError("conflicting pinned evidence")
                pins[key] = value
            pins[str(root / "declaration.json")] = digest(root / "declaration.json")
            pins[str(root / "complete.json")] = digest(root / "complete.json")
            seen = set()
            for r in c["results"]:
                i = r["course_index"]
                if i not in diagnostic_indices() or i in seen or r["owner"] != owner:
                    raise ValueError("unique exact declared expert/courses required")
                seen.add(i)
                folder = root / f"course{i:03d}-owner{owner}"
                if json.loads((folder / "complete.json").read_text()) != r:
                    raise ValueError("per-course receipt differs")
                for name, expected in (
                    ("native.npz", r["native_hash"]),
                    ("sampling.npz", r["sampling_hash"]),
                    ("complete.json", digest(folder / "complete.json")),
                ):
                    pins[str(folder / name)] = expected
                proof, score, prefix = r["replay"], r["score"], r["prefix_diagnostics"]
                if (
                    r["new_native_steps"] != 6000
                    or proof["additional_native_steps"] != 3000
                    or proof["student_neural_and_muscle_native_bit_exact"] is not True
                    or proof["intention_network_calls"] != 300
                    or proof["muscle_decoder_calls"] != 280
                    or proof["teacher_encoder_calls"] != 0
                    or proof["teacher_planner_calls"] != 0
                    or r["causal_qualified"]
                    != bool(
                        score["qualified"]
                        and prefix["prefix_physical_safe"]
                        and prefix["post_boundary_control_can_cause_first_touch"]
                    )
                ):
                    raise ValueError("actual frozen expert/replay proof differs")
                meta = {
                    k: r[k]
                    for k in (
                        "baseline_success",
                        "deployed_portfolio_success",
                        "deployed_owner",
                        "stratum",
                    )
                }
                item = rows.setdefault(i, dict(meta=meta, experts={}))
                if item["meta"] != meta:
                    raise ValueError("paired counterfactual contexts differ")
                item["experts"][str(owner)] = r["causal_qualified"]
        if set(rows) != set(diagnostic_indices()) or owners != set(range(1, 5)):
            raise ValueError("whole24x4 diagnostic matrix required")
        matrix = [
            dict(course_index=i, **rows[i]["meta"], expert_success=rows[i]["experts"])
            for i in sorted(rows)
        ]
        union = sum(r["baseline_success"] or any(r["expert_success"].values()) for r in matrix)
        baseline = sum(r["baseline_success"] for r in matrix)
        deployed = sum(r["deployed_portfolio_success"] for r in matrix)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="FROZEN_EXPERT_COVERAGE_DIAGNOSED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                matrix=matrix,
                contexts=24,
                baseline_success=baseline,
                deployed_portfolio_success=deployed,
                counterfactual_oracle_union_success=union,
                counterfactual_oracle_missed_by_deployed=sum(
                    (r["baseline_success"] or any(r["expert_success"].values()))
                    and not r["deployed_portfolio_success"]
                    for r in matrix
                ),
                expert_success_counts={
                    str(o): sum(r["expert_success"][str(o)] for r in matrix) for o in range(1, 5)
                },
                diagnostic_only=True,
                fresh_test=False,
                oracle_deployable=False,
                reviewed_native_steps=576000,
                new_native_steps=0,
                optimizer_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        print("COVERAGE_DIAGNOSIS_COMPLETE", baseline, deployed, union, flush=True)
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                new_native_steps=0,
                optimizer_steps=0,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
