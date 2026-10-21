"""Read-only per-trajectory gate/fit diagnostics on hash-bound TRAIN features.

Recorded teacher histories are not the student's on-policy state distribution;
this report is not physical qualification or an additional training execution.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-root", type=Path, required=True)
    p.add_argument("--student-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compare-root", type=Path)
    p.add_argument("--compare-hash")
    a = p.parse_args()
    if (a.compare_root is None) != (a.compare_hash is None):
        p.error("comparison root and hash must be provided together")
    rp = a.student_root / "complete.json"
    if (
        a.output.exists()
        or digest(rp) != a.student_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("exact closed student and unused pinned diagnostic output required")
    r, d = [json.loads(path.read_text()) for path in (rp, a.student_root / "declaration.json")]
    data_root = Path(d["data_root"])
    data = json.loads((data_root / "complete.json").read_text())
    pins = dict(r["input_file_hashes"])
    pins.update({str(rp): a.student_hash, r["checkpoint_path"]: r["checkpoint_hash"]})
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.paid_successor_student_actor import load_successor_student

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    policy = next(Path(path) for path in pins if path.endswith("/ckp/policy_30000.onnx"))
    actor, _ = load_successor_student(
        policy,
        checkpoint=Path(r["checkpoint_path"]),
        checkpoint_hash=r["checkpoint_hash"],
        training_receipt=rp,
        training_receipt_hash=a.student_hash,
        scene_hash=r["body_scene_hash"],
    )
    comparison = None
    if a.compare_root is not None:
        other_rp = a.compare_root / "complete.json"
        if digest(other_rp) != a.compare_hash:
            raise ValueError("exact comparison student receipt required")
        other = json.loads(other_rp.read_text())
        old_actor, _ = load_successor_student(
            policy,
            checkpoint=Path(other["checkpoint_path"]),
            checkpoint_hash=other["checkpoint_hash"],
            training_receipt=other_rp,
            training_receipt_hash=a.compare_hash,
            scene_hash=r["body_scene_hash"],
        )
        pins.update(
            {str(other_rp): a.compare_hash, other["checkpoint_path"]: other["checkpoint_hash"]}
        )
        if other["data_hash"] != r["data_hash"] or other["config"]["seed"] != r["config"]["seed"]:
            raise ValueError("paired ablation requires identical data and initialization seed")
        same_gate = all(
            torch.equal(v, old_actor.residual.gate.state_dict()[k])
            for k, v in actor.residual.gate.state_dict().items()
        )
        same_base = all(
            torch.equal(v, old_actor.base.state_dict()[k])
            for k, v in actor.base.state_dict().items()
        )
        if not same_gate or not same_base:
            raise ValueError("paired amplitude ablation changed classifier or frozen motor")
        comparison = dict(
            data_and_seed_identical=True,
            gate_tensors_bitwise_identical=True,
            base_tensors_bitwise_identical=True,
            comparison_receipt_hash=a.compare_hash,
        )
    with np.load(data["feature_path"], allow_pickle=False) as z:
        x, y, active = [torch.from_numpy(z[key].copy()) for key in ("features", "target", "active")]
    with torch.no_grad():
        predicted = actor.residual(x)
        logits = actor.residual.gate(x)[:, 0]
        capacity = (
            0.2 * torch.sigmoid(logits)
            if r["config"].get("gate_scales_amplitude", True)
            else torch.full_like(logits, 0.2)
        )
        capacity = torch.where(logits >= 0, capacity, torch.zeros_like(capacity))
        unavoidable = torch.relu(y.abs() - capacity[:, None]).square()
        rejected_error = torch.where((logits < 0)[:, None], y.square(), torch.zeros_like(y))
        rows = []
        for row in data["trajectories"]:
            start, stop = row["start"], row["stop"]
            mask = active[start:stop]
            rows.append(
                dict(
                    case=row["case"],
                    skill=row["skill"],
                    role=row["role"],
                    rows=stop - start,
                    active_rows=int(mask.sum()),
                    active_open=int((logits[start:stop][mask] >= 0).sum()),
                    maximum_logit=float(logits[start:stop].max()),
                    minimum_active_logit=float(logits[start:stop][mask].min())
                    if mask.any()
                    else None,
                    active_joint_MSE=float(
                        (predicted[start:stop][mask] - y[start:stop][mask]).square().mean()
                    )
                    if mask.any()
                    else None,
                    amplitude_MSE_lower_bound=float(unavoidable[start:stop][mask].mean())
                    if mask.any()
                    else None,
                    amplitude_limited_active_rows=int(
                        (
                            y[start:stop][mask].abs().max(dim=1).values > capacity[start:stop][mask]
                        ).sum()
                    )
                    if mask.any()
                    else 0,
                    rejected_gate_MSE_lower_bound=float(rejected_error[start:stop][mask].mean())
                    if mask.any()
                    else None,
                    admitted_amplitude_MSE_lower_bound=float(
                        (unavoidable[start:stop][mask] - rejected_error[start:stop][mask]).mean()
                    )
                    if mask.any()
                    else None,
                )
            )
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="PAID_STUDENT_RECORDED_HISTORY_FIT_DIAGNOSTIC",
            input_file_hashes=pins,
            source_head=a.expected_head,
            rows=rows,
            paired_ablation=comparison,
            own_residual_forward_rows=len(x),
            own_gate_forward_rows=len(x),
            native_steps=0,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps([r for r in rows if r["role"] == "teacher"]), flush=True)


if __name__ == "__main__":
    main()
