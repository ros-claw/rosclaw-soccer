"""Read-only local teacher/student feedback audit; derivatives are not stability proof."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "train-root", "model-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("expected-head", "expected-bank-hash", "expected-training-hash"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != args.expected_head
        or args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    ):
        raise ValueError("clean frozen read-only audit source and unused output required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        digest(args.bank / "complete.json") != args.expected_bank_hash
        or digest(args.train_root / "complete.json") != args.expected_training_hash
    ):
        raise ValueError("bank/training identity differs")
    bd = json.loads((args.bank / "declaration.json").read_text())
    tc = json.loads((args.train_root / "complete.json").read_text())
    td = json.loads((args.train_root / "declaration.json").read_text())
    if (
        td["seed"] != 17
        or tc["optimizer_steps"] != 3000
        or tc["bank_complete_hash"] != args.expected_bank_hash
    ):
        raise ValueError("fixed failed seed17 step3000 required")
    weight = args.train_root / tc["checkpoint"]
    if digest(weight) != tc["checkpoint_hash"]:
        raise ValueError("student weight identity differs")
    pins = dict(td["input_file_hashes"])
    pins.update(bd["input_file_hashes"])
    for root in (args.bank, args.train_root):
        pins.update({str(p): digest(p) for p in root.iterdir() if p.is_file()})
    for name in (
        "planner_sonic.onnx",
        "low_latency/model_encoder.onnx",
        "low_latency/model_decoder.onnx",
        "low_latency/observation_config.yaml",
    ):
        path = args.model_root / name
        if pins.get(str(path)) != digest(path):
            raise ValueError("original qualified SONIC network required")
    check_pins(pins)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    import collections
    from types import SimpleNamespace

    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig, G1SonicRunupController
    from rosclaw_soccer.rsi.neural_motor_student import create_motor_student
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motor_student_rollout import action_target, exact_array, prediction

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    payload = torch.load(weight, map_location="cpu", weights_only=True)
    model = create_motor_student()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    backend = G1SonicRunupController(
        args.model_root, G1SonicRunupConfig(model_variant="low_latency"), inference_threads=1
    )
    # Position and velocity groups have different units; compare each separately.
    groups = (
        ("joint_position_rad", 29, 0.001),
        ("joint_velocity_rad_s", 29, 0.01),
        ("body_angular_velocity_rad_s", 3, 0.01),
        ("root_roll_pitch_rad", 2, 0.001),
    )
    results = []
    teacher_calls = student_calls = 0
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            source_head=head,
            input_file_hashes=pins,
            course_indices=[2, 22],
            probe_frames=[20, 60, 120, 240],
            current_state_only_perturbed=True,
            past_history_held_fixed=True,
            reference_is_privileged_teacher_context=True,
            new_native_steps=0,
            optimizer_steps=0,
            hardware_authorized=False,
        ),
    )
    for index in (2, 22):
        record = bd["rows"][index]
        path = Path(record["source_artifact"])
        if not record["actor_eligible"] or digest(path) != record["source_artifact_hash"]:
            raise ValueError("original verified positive teacher trajectory required")
        with np.load(path, allow_pickle=False) as z:
            raw = {k: z[k].copy() for k in z.files}
        backend._history.clear()
        previous = np.zeros(29, dtype=np.float32)
        hidden = None
        samples = []
        baseline_error = []
        for frame in range(300):
            q, v = raw["canonical_qpos"][frame], raw["canonical_qvel"][frame]
            state = SimpleNamespace(qpos=q, qvel=v)
            entry = backend._history_entry(state, previous)
            if frame == 0:
                backend._history.extend(tuple(a.copy() for a in entry) for _ in range(10))
            else:
                backend._history.append(entry)
            backend.reference = raw["encoder_reference_poses"][frame].copy()
            backend._update_from_reference(state, 0)
            teacher_calls += 1
            exact_array(backend.action, raw["neural_action_isaac"][frame])
            before = hidden
            _, student, _, hidden, _ = prediction(model, q, v, previous, frame, before)
            student_calls += 1
            baseline_error.append(float(np.mean((student - backend.action) ** 2)))
            if frame in (20, 60, 120, 240):
                history = [tuple(a.copy() for a in item) for item in backend._history]
                for group, width, epsilon in groups:
                    teacher_jac, student_jac = [], []
                    for column in range(width):
                        teacher_pair, student_pair = [], []
                        for sign in (-1, 1):
                            pq, pv = q.copy(), v.copy()
                            if group == "joint_position_rad":
                                pq[7 + column] += sign * epsilon
                            elif group == "joint_velocity_rad_s":
                                pv[6 + column] += sign * epsilon
                            elif group == "body_angular_velocity_rad_s":
                                pv[3 + column] += sign * epsilon
                            else:
                                rotation = np.zeros(4)
                                rotation[0], rotation[1 + column] = (
                                    np.cos(epsilon / 2),
                                    sign * np.sin(epsilon / 2),
                                )
                                mujoco.mju_mulQuat(pq[3:7], q[3:7], rotation)
                            perturbed = SimpleNamespace(qpos=pq, qvel=pv)
                            backend._history = collections.deque(
                                history[:-1] + [backend._history_entry(perturbed, previous)],
                                maxlen=10,
                            )
                            backend._update_from_reference(perturbed, 0)
                            teacher_calls += 1
                            teacher_pair.append(action_target(backend.action))
                            _, sa, _, _, _ = prediction(model, pq, pv, previous, frame, before)
                            student_calls += 1
                            student_pair.append(action_target(sa))
                        teacher_jac.append((teacher_pair[1] - teacher_pair[0]) / (2 * epsilon))
                        student_jac.append((student_pair[1] - student_pair[0]) / (2 * epsilon))
                    tj, sj = np.stack(teacher_jac, axis=1), np.stack(student_jac, axis=1)
                    tn, sn = float(np.linalg.norm(tj)), float(np.linalg.norm(sj))
                    samples.append(
                        dict(
                            frame=frame,
                            group=group,
                            epsilon=epsilon,
                            teacher_target_jacobian_frobenius=tn,
                            student_target_jacobian_frobenius=sn,
                            response_norm_ratio=sn / tn if tn else None,
                            cosine=float(np.sum(tj * sj) / (tn * sn)) if tn and sn else None,
                            relative_jacobian_error=float(np.linalg.norm(sj - tj) / tn)
                            if tn
                            else None,
                        )
                    )
                    np.savez_compressed(
                        args.output / f"course{index:02d}-frame{frame:03d}-{group}.npz",
                        teacher=tj,
                        student=sj,
                    )
                backend._history = collections.deque(history, maxlen=10)
            backend.action = raw["neural_action_isaac"][frame].copy()
            previous = backend.action.copy()
        result = dict(
            course_index=index,
            teacher_actions_recomputed_exact=300,
            teacher_forced_student_mse=float(np.mean(baseline_error)),
            probes=samples,
            derivatives_are_not_stability_proofs=True,
        )
        results.append(result)
        write_once(args.output / f"course{index:02d}.json", result)
        print("LOCAL_FEEDBACK_AUDIT", index, json.dumps(samples), flush=True)
    check_pins(pins)
    write_once(
        args.output / "complete.json",
        dict(
            results=results,
            teacher_encoder_decoder_calls=teacher_calls,
            student_calls=student_calls,
            new_native_steps=0,
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
            derivatives_are_not_stability_proofs=True,
        ),
    )


if __name__ == "__main__":
    main()
