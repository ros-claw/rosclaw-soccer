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
    a = p.parse_args()
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
