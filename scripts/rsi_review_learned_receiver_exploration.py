"""Audit a learned-centered receiver exploration pilot against its exact parent."""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
from rosclaw_soccer.rsi.motor_sampling_law import history_motor_logp
from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal, handoff_equal
from scripts.rsi_watch_passer_residual_pair import worker_ready


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("exploration", "parent", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("exploration-head", "parent-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--wait-unit")
    p.add_argument("--precontact", action="store_true")
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= a.timeout_sec <= 7200
        or a.wait_unit is not None
        and (
            not a.wait_unit.startswith("rosclaw-receiver-learned-exploration-")
            or not a.wait_unit.endswith(".service")
            or "/" in a.wait_unit
        )
    ):
        raise ValueError("unused external output required")
    if a.wait_unit:
        deadline = time.monotonic() + a.timeout_sec
        while True:
            result = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    a.wait_unit,
                    "-p",
                    "LoadState",
                    "-p",
                    "ActiveState",
                    "-p",
                    "Result",
                    "-p",
                    "ExecMainStatus",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            if worker_ready(a.exploration, state):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("physical exploration did not complete; do not restart it")
            time.sleep(15)
    roots = (a.exploration / "source", a.parent / "source")
    audits = (a.exploration / "audit/complete.json", a.parent / "audit/complete.json")
    audit_hashes = tuple(map(digest, audits))
    sampled, _, sampled_hash = report(
        roots[0],
        audits[0],
        audit_hashes[0],
        "NEURAL_RELAY_RECEIVER_PRECONTACT_EXPLORATION32_PENDING_OWN_REPLAY"
        if a.precontact
        else "NEURAL_RELAY_RECEIVER_LEARNED_EXPLORATION32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    parent, _, parent_hash = report(
        roots[1],
        audits[1],
        audit_hashes[1],
        "NEURAL_RELAY_RECEIVER_PRECONTACT_PAIR16_PENDING_OWN_REPLAY"
        if a.precontact
        else "NEURAL_RELAY_RECEIVER_DENSE_PAIR16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    decl, parent_decl = [json.loads((r / "declaration.json").read_text()) for r in roots]
    if (
        sampled["source_head"] != a.exploration_head
        or parent["source_head"] != a.parent_head
        or decl["development_context_block"] != parent_decl["development_context_block"]
        or decl["receiver_checkpoint_hash"] != parent_decl["receiver_checkpoint_hash"]
        or decl["receiver_candidate_audit_hash"] != parent_decl["receiver_candidate_audit_hash"]
        or decl["receiver_dataset_hash"] != parent_decl["receiver_dataset_hash"]
    ):
        raise ValueError("exact parent candidate, context block and source heads required")
    configs = [
        Path(k)
        for k in sampled["input_file_hashes"]
        if k.endswith("/velocity/v0/params/deploy.yaml")
    ]
    if len(configs) != 1:
        raise ValueError("unique pinned motor codec required")
    scale = np.asarray(
        VelocityPolicyParameters.load(
            configs[0], expected_hash=sampled["input_file_hashes"][str(configs[0])]
        ).action_scale,
        np.float32,
    )
    trials = {
        (
            r["course"]["context"],
            r["course"]["lateral_m"],
            r["course"]["arm"]["exploration_seed"],
        ): r
        for r in sampled["records"]
    }
    parents = {
        (r["course"]["context"], r["course"]["lateral_m"]): r
        for r in parent["records"]
        if r["course"]["arm"]["receiver_learned"]
    }
    if len(trials) != 32 or len(parents) != 8:
        raise ValueError("complete development pilot factorial required")
    rows, max_error = [], 0.0
    for condition, original in sorted(parents.items()):
        base = trials[(*condition, None)]
        if (
            any(
                base[k] != original[k]
                for k in ("termination", "controls", "native_steps", "result")
            )
            or not handoff_equal(roots[0], base, roots[1], original)
            or not all(
                artifact_equal(roots[0], base, roots[1], original, k) for k in original["artifacts"]
            )
        ):
            raise ValueError("deterministic learned parent trajectory changed")
        for seed in (92117, 92143, 92171):
            trial = trials[(*condition, seed)]
            if not artifact_equal(roots[0], trial, roots[1], original, "initial"):
                raise ValueError("all exploration must start from the parent physical state")
            velocity, physics = (
                bind(roots[0], trial, "receiver_velocity"),
                bind(roots[0], trial, "physics"),
            )
            if velocity is None or physics is None:
                raise ValueError("complete sampled neural and physics history required")
            with (
                np.load(velocity, allow_pickle=False) as v,
                np.load(physics, allow_pickle=False) as raw,
            ):
                n = len(v["control"])
                if (
                    n < 1
                    or not np.array_equal(v["control"], np.arange(n))
                    or not v["SAM_active"].all()
                ):
                    raise ValueError("complete contiguous sampled control history required")
                density = (
                    history_motor_logp(
                        "box-truncated-ar1",
                        torch.from_numpy(v["action"]),
                        torch.from_numpy(v["SAM_executed_action"]),
                        torch.from_numpy(v["reference_action"]),
                        torch.from_numpy(scale),
                        torch.full((29,), math.log(0.02), dtype=torch.float32),
                        rho=0.9,
                    )
                    .detach()
                    .numpy()
                )
                error = float(np.max(np.abs(density - v["SAM_logp"])))
                if not np.isfinite(error) or error > 1e-4:
                    raise ValueError("actual learned-centered sampling density mismatch")
                max_error = max(max_error, error)
                outcome = strict_pass_receive(
                    raw["ball_forces"],
                    raw["body_failed"],
                    handoff=trial["result"]["handoff_executed"],
                )
            rows.append(
                dict(
                    context=condition[0],
                    side=condition[1],
                    seed=seed,
                    source_trial=trial["index"],
                    stochastic_controls=n,
                    baseline=measured(roots[1], original),
                    candidate=measured(roots[0], trial),
                    physical_clean_outcome=outcome,
                )
            )
    frozen = (
        (roots[0] / "complete.json", sampled_hash),
        (roots[1] / "complete.json", parent_hash),
        (audits[0], audit_hashes[0]),
        (audits[1], audit_hashes[1]),
    )
    if len(rows) != 24 or any(digest(path) != h for path, h in frozen):
        raise ValueError("all original source receipts must remain unchanged")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="PRECONTACT_RECEIVER_EXPLORATION_PILOT_REVIEWED"
            if a.precontact
            else "LEARNED_RECEIVER_EXPLORATION_PILOT_REVIEWED",
            source_head=a.expected_head,
            conditions=8,
            sampled_episodes=24,
            sampled_source_hash=sampled_hash,
            sampled_audit_hash=audit_hashes[0],
            parent_source_hash=parent_hash,
            parent_audit_hash=audit_hashes[1],
            receiver_checkpoint_hash=decl["receiver_checkpoint_hash"],
            records=rows,
            maximum_density_error=max_error,
            parent_trajectory_identical=True,
            independent_new_conditions=False,
            all_seeds_or_blocks=False,
            training_performed=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
