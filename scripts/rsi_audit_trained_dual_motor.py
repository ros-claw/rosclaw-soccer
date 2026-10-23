"""Independent CPU whole-history checkpoint replay; no physical qualification."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-kind",
        choices=(
            "supervised",
            "ar1-actor-critic",
            "contact-consolidated",
            "contact-consolidated-v2",
            "team-pass-consolidated-v1",
        ),
        default="supervised",
    )
    for n in ("root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("source-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("frozen clean independent CPU model audit required")

    clean()
    rp = a.root / "complete.json"
    if (
        a.output.exists()
        or rp.is_symlink()
        or digest(rp) != a.source_hash
        or (a.root / "failed.json").exists()
    ):
        raise ValueError("exact trained research candidate and unused output required")
    r = json.loads(rp.read_text())
    pins = dict(r["input_file_hashes"])
    pins[str(rp)] = a.source_hash
    paths = [
        Path(x)
        for x, h in pins.items()
        if h == r["parent_lineage_hash"] and x.endswith("/complete.json")
    ]
    data_paths = [
        Path(x) for x, h in pins.items() if h == r["data_hash"] and x.endswith("/complete.json")
    ]
    if len(paths) != 1 or len(data_paths) != 1:
        raise ValueError("unique exact model and actual-history data lineage required")
    parent_record = json.loads(paths[0].read_text())
    data = json.loads(data_paths[0].read_text())
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    for path in (
        a.root / "declaration.json",
        Path(r["initial_checkpoint_path"]),
        Path(r["checkpoint_path"]),
    ):
        pins[str(path)] = digest(path)
    check_pins(pins)
    a.output.mkdir()
    calls = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from rosclaw_soccer.rsi.contact_consolidation_data import context_witness_hash

        witness_hash = context_witness_hash(data, a.checkpoint_kind)
        teacher_path = next(
            Path(x) for x, h in pins.items() if h == witness_hash and x.endswith("/complete.json")
        )
        case = json.loads((teacher_path.parent / "declaration.json").read_text())["source_case"]
        _, policy, scene = open_paid_bench(pins, [case])
        parent, _ = load_paid_student(
            policy,
            checkpoint=Path(parent_record["checkpoint_path"]),
            checkpoint_hash=parent_record["checkpoint_hash"],
            training_receipt=paths[0],
            training_receipt_hash=r["parent_lineage_hash"],
            scene_hash=scene,
        )
        model, _ = load_dual_candidate_for_sim_audit(
            parent,
            parent_lineage_hash=r["parent_lineage_hash"],
            root=a.root,
            expected_receipt_hash=a.source_hash,
            checkpoint_kind=a.checkpoint_kind,
        )
        expected_sequences = (
            177
            if a.checkpoint_kind == "team-pass-consolidated-v1"
            else 110
            if a.checkpoint_kind == "contact-consolidated-v2"
            else 65
            if a.checkpoint_kind == "contact-consolidated"
            else 24
            if data["status"] == "DUAL_MOTOR_ACTUAL_DAGGER_SEQUENCES_READY_NOT_TRAINED"
            else 22
        )
        if len(data["records"]) != expected_sequences or len(r["cpu_replay"]) != expected_sequences:
            raise ValueError("all explicitly declared original/DAgger sequences required")
        outcomes = []
        for source, production in zip(data["records"], r["cpu_replay"], strict=True):
            path = data_paths[0].parent / source["raw_path"]
            pred = a.root / production["prediction_path"]
            if (
                path.is_symlink()
                or pred.is_symlink()
                or not path.resolve().is_relative_to(data_paths[0].parent.resolve())
                or not pred.resolve().is_relative_to(a.root.resolve())
                or digest(path) != source["raw_hash"]
                or digest(pred) != production["prediction_hash"]
            ):
                raise ValueError(
                    "confined exact full-history inputs and claimed predictions required"
                )
            pins[str(path)] = source["raw_hash"]
            pins[str(pred)] = production["prediction_hash"]
            with np.load(path, allow_pickle=False) as z:
                obs = z["observations"].copy()
                targets = z["targets"].copy()
                scale = z["action_scale"].copy()
                reference = z["reference_nominal"] + z["reference_delta"] / scale
            with np.load(pred, allow_pickle=False) as z:
                claimed = z["means"].copy()
            private = model.initial_state()
            means = []
            for i, row in enumerate(obs):
                with torch.no_grad():
                    mean, private, e = model.forward_with_evidence(
                        torch.from_numpy(row).reshape(1, 1, 160), private
                    )
                calls += 1
                if (
                    not np.array_equal(e["reference_mean"][0, 0].numpy(), reference[i])
                    or (e["final_mean_delta_radians"].abs() > 0.2).any()
                ):
                    raise ValueError(
                        "private reference history or deterministic motor authority differs"
                    )
                means.append(mean[0, 0].numpy().copy())
            actual = np.stack(means)
            mse = float(np.mean(((actual - targets) * scale) ** 2))
            if (
                not np.array_equal(actual, claimed)
                or mse != production["motor_MSE"]
                or production["index"] != source["index"]
                or len(obs) != source["rows"]
            ):
                raise ValueError("independent CPU prediction/whole-history count differs")
            outcomes.append(
                dict(index=source["index"], role=source["role"], rows=len(obs), motor_MSE=mse)
            )
            print(json.dumps(outcomes[-1]), flush=True)
        if calls != r["cpu_offline_decisions"]:
            raise ValueError("all whole-history decisions required")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                source_hash=a.source_hash,
                checkpoint_kind=a.checkpoint_kind,
                checkpoint_hash=r["checkpoint_hash"],
                data_hash=r["data_hash"],
                results=outcomes,
                own_actor_decisions=calls,
                recurrent_branch_queries=2 * calls,
                optimizer_metadata_checked_only=True,
                optimizer_reexecuted=False,
                physical_steps_executed=0,
                closed_loop_executed=False,
                old_retention_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                own_actor_decisions=calls,
                physical_steps=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
