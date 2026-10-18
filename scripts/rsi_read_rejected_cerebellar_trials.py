"""Observe exact numeric update replay at its KL-rejection boundary.

No source edits, new behavior data, candidate, physics or limit changes. Python
line tracing records already computed trial statistics before original rollback.
The whole resulting actor/critic/Adam checkpoint must equal the original file.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def rejection_line(source):
    tree = ast.parse(source)
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit_update"
    )
    matches = [
        n.lineno
        for n in ast.walk(function)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "not valid"
    ]
    if len(matches) != 1:
        raise ValueError("one unmodified optimizer validity boundary required")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sys.gettrace() is not None:
        raise ValueError("unused output and no existing tracer required")
    cfg = json.loads(args.config.read_text())
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("exact unchanged frozen producer required")
    sys.path[:0] = [str(source / "src"), str(source), str(Path(cfg["core_root"]) / "src")]
    import numpy as np

    from rosclaw_soccer.rsi import cerebellar_actor_critic as optimizer
    from rosclaw_soccer.rsi import cerebellar_recovery_return as reward
    from rosclaw_soccer.rsi import cerebellar_torque_policy as policy
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once

    pins = {}

    def pin(path):
        pins[str(path)] = digest(path)

    for path in (
        Path(__file__),
        args.config,
        root / "declaration.json",
        Path(optimizer.__file__),
        Path(reward.__file__),
        Path(policy.__file__),
        root / "anchor-features.npz",
    ):
        pin(path)
    boundary = rejection_line(Path(optimizer.__file__).read_text())
    with np.load(root / "anchor-features.npz", allow_pickle=False) as archive:
        anchors = archive["features"]
    rows = []
    for seed in (2026101801, 2026101802, 2026101803):
        paths = [
            root / f"{prefix}-{seed}.{suffix}"
            for prefix, suffix in (
                ("initial-checkpoint", "json"),
                ("trained-checkpoint", "json"),
                ("batch", "json"),
                ("batch", "npz"),
            )
        ]
        for path in paths:
            pin(path)
        initial, original, manifest = [json.loads(path.read_text()) for path in paths[:3]]
        if manifest["report_hash"] != hash_json(
            {k: v for k, v in manifest.items() if k != "report_hash"}
        ):
            raise ValueError("original admitted batch manifest seal required")
        with np.load(paths[3], allow_pickle=False) as archive:
            arrays = dict(archive)
        arrays.update(
            behavior_model_hash=initial["actor"]["model_hash"],
            reward_contract_hash=initial["training_reward_contract_hash"],
        )
        trials = []

        def observe(frame, event, arg, captured=trials):
            if frame.f_code is optimizer.fit_update.__code__:
                if event == "line" and frame.f_lineno == boundary:
                    captured.append(
                        dict(
                            attempt=frame.f_locals["attempted"],
                            valid=bool(frame.f_locals["valid"]),
                            kl=dict(frame.f_locals["kl"]),
                        )
                    )
                return observe
            return None

        try:
            sys.settrace(observe)
            rebuilt = optimizer.fit_update(
                initial,
                arrays,
                batch_hash=manifest["report_hash"],
                anchor_features=anchors,
                steps=4,
            )
        finally:
            sys.settrace(None)
        if (
            rebuilt != original
            or len(trials) != original["learning_receipt"]["attempted_optimizer_steps"]
            or sum(t["valid"] for t in trials)
            != original["learning_receipt"]["accepted_joint_optimizer_steps"]
        ):
            raise ValueError(
                "whole original persistent checkpoint or observed attempt count differs"
            )
        for trial in trials:
            trial["exceeded_kl_thresholds"] = [
                name
                for name in trial["kl"]
                if trial["kl"][name]
                > optimizer.CONFIG[
                    {
                        "rollout_mean": "maximum_rollout_mean_kl",
                        "rollout_max": "maximum_rollout_state_kl",
                        "anchor_mean": "maximum_anchor_mean_kl",
                        "anchor_max": "maximum_anchor_state_kl",
                    }[name]
                ]
            ]
        rows.append(
            dict(
                seed=seed,
                checkpoint_hash=original["checkpoint_hash"],
                whole_checkpoint_exactly_reconstructed=True,
                trials=trials,
            )
        )
        print("ORIGINAL_CHECKPOINT_AND_REJECTED_TRIAL_REPLAYED", seed, flush=True)
    if any(digest(Path(name)) != expected for name, expected in pins.items()):
        raise ValueError("original inputs changed during numeric audit")
    report = dict(
        schema="soccer.rsi.cerebellar_rejected_trial_numeric_readback.v1",
        rows=rows,
        input_file_hashes=pins,
        original_optimizer_source_unmodified=True,
        historical_batch_physical_provenance_reconstructed_here=False,
        additional_audit_optimizer_attempts=sum(len(r["trials"]) for r in rows),
        additional_audit_accepted_step_recomputations=sum(
            t["valid"] for r in rows for t in r["trials"]
        ),
        new_training_updates=0,
        new_candidate_models=0,
        new_physical_steps=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
