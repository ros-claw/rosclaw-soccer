"""Materialize one consistent successful TRAIN behavior; retain all source archives."""

import argparse
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("original", "teacher", "teacher-audit", "replan", "replan-audit"):
        p.add_argument("--" + n + "-root", type=Path, required=True)
        p.add_argument("--" + n + "-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused canonical output and clean frozen source required")
    from rosclaw_soccer.rsi.paid_canonical_behavior import (
        DATA_STATUS,
        SCHEMA,
        SOURCE_NAMES,
        read_witnesses,
    )

    identity = {k: str(getattr(a, k)) for n in SOURCE_NAMES for k in (n + "_root", n + "_hash")}
    _, pins = read_witnesses(identity)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_canonical_behavior import reconstruct

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        parent, baseline, old, z, choices, trajectories, retired = reconstruct(identity, policy)
        del parent
        common = dict(
            **identity,
            source_head=a.expected_head,
            schema=SCHEMA,
            bank="train",
            Fresh_fit_rows=0,
            DEV_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        write_once(
            a.output / "declaration.json",
            dict(
                **common,
                input_file_hashes=pins,
                selection_rule="lowest_worst_abs_lateral_then_min_progress_then_original",
                student_prefix_labels="actual_executed_part_of_audited_success_NOT_expert",
            ),
        )
        archive = a.output / "features.npz"
        np.savez_compressed(archive, **z)
        memory = a.output / "canonical-choice-memory.json"
        write_once(
            memory, dict(choices=choices, retired=retired, original_artifacts_unchanged=True)
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=DATA_STATUS,
                **common,
                declaration_hash=digest(a.output / "declaration.json"),
                input_file_hashes=pins,
                feature_path=str(archive),
                feature_hash=digest(archive),
                choice_memory_hash=digest(memory),
                choices=choices,
                trajectories=trajectories,
                retired_fit_trajectories=retired,
                rows=len(z["features"]),
                original_rows=len(old["features"]),
                teacher_rows=int(z["teacher"].sum()),
                protected_rows=int((~z["teacher"]).sum()),
                active_rows=int(z["active"].sum()),
                unchanged_nonselected_rows=True,
                **{
                    k: baseline[k]
                    for k in ("body_scene_hash", "parent_hash", "parent_checkpoint_hash")
                },
                successful_student_prefix_rows=sum(
                    t.get("student_prefix_rows", 0) for t in trajectories
                ),
                optimizer_updates=0,
                own_native_steps=0,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                optimizer_updates=0,
                own_native_steps=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
