"""Read-only teacher-forcing vs own-state drift diagnosis; no physical integration."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "train-root", "evaluation", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in (
        "expected-head",
        "expected-bank-hash",
        "expected-training-hash",
        "expected-evaluation-hash",
    ):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != args.expected_head
        or args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    ):
        raise ValueError("sealed diagnostic source and unused evidence required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    for path, expected in (
        (args.bank / "complete.json", args.expected_bank_hash),
        (args.train_root / "complete.json", args.expected_training_hash),
        (args.evaluation / "complete.json", args.expected_evaluation_hash),
    ):
        if digest(path) != expected:
            raise ValueError("diagnostic input identity differs")
    bd = json.loads((args.bank / "declaration.json").read_text())
    tc = json.loads((args.train_root / "complete.json").read_text())
    ec = json.loads((args.evaluation / "complete.json").read_text())
    ed = json.loads((args.evaluation / "declaration.json").read_text())
    pins = dict(ed["input_file_hashes"])
    pins.update({str(p): digest(p) for p in args.evaluation.rglob("*") if p.is_file()})
    pins.update({str(p): digest(p) for p in args.bank.iterdir() if p.is_file()})
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    check_pins(pins)
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.neural_motor_student import create_motor_student
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motor_student_rollout import action_target

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = args.train_root / tc["checkpoint"]
    if digest(checkpoint) != tc["checkpoint_hash"]:
        raise ValueError("motor weight identity differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = create_motor_student()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    bankproof = json.loads((args.bank / "complete.json").read_text())
    with np.load(args.bank / bankproof["artifact"], allow_pickle=False) as z:
        x = z["features"].copy()
        y = z["targets"].copy()
        eligible = z["actor_eligible"].copy()
        train = z["train"].copy()
    results = []
    calls = 0
    for row in ec["results"]:
        i = row["course"]["course_index"]
        if row["variant"] != "trained" or not eligible[i] or row["status"] != "COMPLETE":
            continue
        p = args.evaluation / f"trained-course{i:02d}" / row["artifact"]
        if digest(p) != row["artifact_hash"]:
            raise ValueError("own-state trace changed")
        with np.load(p, allow_pickle=False) as z:
            own = {k: z[k].copy() for k in z.files}
        with torch.no_grad():
            batch, _, _ = model(torch.from_numpy(x[i : i + 1]))
            calls += 1
            batch = batch[0].numpy().copy()
            hidden = None
            sequential = []
            for frame in range(300):
                a, _, hidden = model(torch.from_numpy(x[i, frame])[None, None], hidden)
                calls += 1
                sequential.append(a[0, 0].numpy().copy())
        sequence = np.stack(sequential)

        def first(mask):
            indices = np.flatnonzero(mask)
            return int(indices[0]) if len(indices) else None

        q = own["target_body_canonical_qpos_after_substep"]
        upright = 1 - 2 * (q[:, :, 4] ** 2 + q[:, :, 5] ** 2)
        rms = np.sqrt(np.mean((own["student_features"] - x[i]) ** 2, axis=1))
        results.append(
            dict(
                course_index=i,
                split="train" if train[i] else "validation",
                prefix_and_boundary_input_max_error=float(
                    abs(own["student_features"][:21] - x[i, :21]).max()
                ),
                teacher_forced_step_vs_batch_max_action_error=float(abs(sequence - batch).max()),
                boundary_step20_own_vs_teacher_forced_action_error=float(
                    abs(own["student_proposed_action_isaac"][20] - sequence[20]).max()
                ),
                boundary_step20_teacher_action_error_linf=float(abs(sequence[20] - y[i, 20]).max()),
                boundary_step20_target_error_linf_rad=float(
                    abs(action_target(sequence[20]) - action_target(y[i, 20])).max()
                ),
                teacher_forced_mse=float(np.mean((sequence - y[i]) ** 2)),
                teacher_forced_mse_20_40=float(np.mean((sequence[20:40] - y[i, 20:40]) ** 2)),
                own_state_action_mse_to_old_teacher=float(
                    np.mean((own["student_proposed_action_isaac"][20:] - y[i, 20:]) ** 2)
                ),
                first_input_rms_deviation_above_point1=first((np.arange(300) >= 20) & (rms > 0.1)),
                first_height_below_point65_frame=first(np.any(q[:, :, 2] < 0.65, axis=1)),
                first_upright_below_point9_frame=first(np.any(upright < 0.9, axis=1)),
                final_qualified=row["score"]["qualified"],
            )
        )
    check_pins(pins)
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output / "declaration.json", dict(source_head=head, input_file_hashes=pins))
    report = dict(
        results=results,
        teacher_forced_model_calls=calls,
        new_native_steps=0,
        optimizer_steps=0,
        teacher_and_own_action_difference_is_not_same_state_error=True,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "complete.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
