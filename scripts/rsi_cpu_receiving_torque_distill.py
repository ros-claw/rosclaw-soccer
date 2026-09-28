"""SIM_ONLY neural distillation of a coupled receiving teacher, with teacher-free physics gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from rsi_cpu_body_bias_recalibration import WEIGHTS, _summary
from rsi_mjx_clean_touch_body_coord_es import FULL_JOINTS, _validate_body_joints
from rsi_mjx_clean_touch_control_es import COURSES, FRESH8, _capture
from rsi_receiving_contact_dynamics_audit import _run

from rosclaw_soccer.providers.g1.receiving_torque_student import (
    ACTION_COUNT,
    FEATURE_COUNT,
    HIDDEN_COUNT,
    MAX_ADDED_TORQUE_NM,
    FrozenReceivingTorqueStudent,
    receiving_torque_features,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SEED = 20260928
EPOCHS = 200
BATCH = 256


def _network() -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(FEATURE_COUNT, HIDDEN_COUNT),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_COUNT, HIDDEN_COUNT),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_COUNT, ACTION_COUNT),
        torch.nn.Tanh(),
    )


def train(
    *,
    asset_root: Path,
    captured: Path,
    fidelity: Path,
    warm_start: Path,
    prior: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    helper = source.with_name("rsi_receiving_contact_dynamics_audit.py")
    student_source = (
        source.parents[1] / "src/rosclaw_soccer/providers/g1/receiving_torque_student.py"
    )
    source_hash = hash_bytes(source.read_bytes())
    helper_hash = hash_bytes(helper.read_bytes())
    student_source_hash = hash_bytes(student_source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY student directory required")
    warm: dict[str, Any] = json.loads(warm_start.read_text(encoding="utf-8"))
    warm_hash = warm.pop("report_hash")
    prior_report: dict[str, Any] = json.loads(prior.read_text(encoding="utf-8"))
    prior_hash = prior_report.pop("report_hash")
    weights = np.asarray(warm["full_weights"], dtype=np.float32)
    if (
        warm_hash != hash_json(warm)
        or prior_hash != hash_json(prior_report)
        or prior_report["schema"] != "rosclaw_soccer.rsi.cpu_receiving_support_feedback.v1"
        or prior_report["warm_start_hash"] != warm_hash
        or prior_report["selected_gain"] != 0.20
        or prior_report["results"][-1]["training_gate_passed"] is not True
        or prior_report["fresh8_opened"] is not False
        or prior_report["promotion_authorized"] is not False
        or warm["training_gate_passed"] is not False
        or warm["fresh8_opened"] is not False
        or weights.shape != (WEIGHTS,)
        or not np.isfinite(weights).all()
        or warm["full_weights_hash"] != hash_bytes(weights.tobytes())
        or tuple(tuple(row) for row in warm["train_courses"][:8]) != COURSES
        or len(warm["train_courses"]) != 9
        or tuple(tuple(row) for row in warm["reserved_fresh8"]) != FRESH8
    ):
        raise ValueError("sealed successful coupled teacher and nine-course seed required")
    arrays = _capture(captured, fidelity)
    center = (
        float(arrays["sonic_recorded_qpos"][45, 36]),
        float(arrays["sonic_recorded_qpos"][45, 37]),
        float(arrays["sonic_recorded_qvel"][45, 35]),
    )
    courses = COURSES + (center,)
    if tuple(tuple(row) for row in warm["train_courses"]) != courses:
        raise ValueError("consumed nine-course identity mismatch")
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    _validate_body_joints(model)
    model.opt.timestep = 0.002
    full = weights.astype(np.float64)
    course_features: list[NDArray[np.float32]] = []
    course_labels: list[NDArray[np.float32]] = []
    teacher_rows = []
    raw_clipped = 0
    raw_total = 0
    for course in courses:
        features: list[NDArray[np.float32]] = []
        labels: list[NDArray[np.float32]] = []

        def sample(
            qpos: NDArray[np.float64],
            qvel: NDArray[np.float64],
            has_foot: bool,
            elapsed: float,
            torque: NDArray[np.float64],
            feature_rows: list[NDArray[np.float32]] = features,
            label_rows: list[NDArray[np.float32]] = labels,
        ) -> None:
            nonlocal raw_clipped, raw_total
            feature_rows.append(receiving_torque_features(qpos, qvel, has_foot, elapsed))
            raw_clipped += int(np.count_nonzero(np.abs(torque) > MAX_ADDED_TORQUE_NM))
            raw_total += ACTION_COUNT
            label_rows.append(
                np.asarray(
                    np.clip(torque, -MAX_ADDED_TORQUE_NM, MAX_ADDED_TORQUE_NM),
                    dtype=np.float32,
                )
            )

        teacher_rows.append(
            _run(
                model,
                arrays,
                full,
                FULL_JOINTS,
                course,
                privileged_teacher_lateral_sign=1.0,
                privileged_teacher_torque_scale=0.25,
                support_posture_gain=0.20,
                sample_hook=sample,
            )
        )
        course_features.append(np.stack(features))
        course_labels.append(np.stack(labels))
    teacher = _summary(teacher_rows)
    if teacher != prior_report["results"][-1]["summary"]:
        raise ValueError("coupled teacher physics changed during labeled replay")

    x_train = np.concatenate(course_features[:8])
    y_train = np.concatenate(course_labels[:8])
    x_val, y_val = course_features[8], course_labels[8]
    mean = x_train.mean(axis=0)
    scale = np.maximum(x_train.std(axis=0), 0.05)
    x_train = np.clip((x_train - mean) / scale, -10.0, 10.0)
    x_val = np.clip((x_val - mean) / scale, -10.0, 10.0)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    network = _network()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-5)
    train_x = torch.from_numpy(x_train)
    train_y = torch.from_numpy(y_train)
    val_x = torch.from_numpy(x_val)
    val_y = torch.from_numpy(y_val)
    generator = torch.Generator().manual_seed(SEED)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] = {}
    loss_history = []
    for epoch in range(EPOCHS):
        network.train()
        order = torch.randperm(len(train_x), generator=generator)
        for batch in order.split(BATCH):
            estimate = MAX_ADDED_TORQUE_NM * network(train_x[batch])
            loss = torch.mean((estimate - train_y[batch]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            validation_loss = float(torch.mean((MAX_ADDED_TORQUE_NM * network(val_x) - val_y) ** 2))
        loss_history.append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().clone() for key, value in network.state_dict().items()
            }
    network.load_state_dict(best_state)
    student = FrozenReceivingTorqueStudent(
        mean=np.asarray(mean, dtype=np.float32),
        scale=np.asarray(scale, dtype=np.float32),
        w1=best_state["0.weight"].numpy().T,
        b1=best_state["0.bias"].numpy(),
        w2=best_state["2.weight"].numpy().T,
        b2=best_state["2.bias"].numpy(),
        w3=best_state["4.weight"].numpy().T,
        b3=best_state["4.bias"].numpy(),
    )
    with torch.no_grad():
        check = MAX_ADDED_TORQUE_NM * network(val_x[:1])
    np_check = student.predict(
        arrays["sonic_recorded_qpos"][45],
        arrays["sonic_recorded_qvel"][45],
        False,
        -1.0,
    )
    # A strict network/NumPy check is performed below on a recorded validation sample.
    normalized = x_val[0]
    numpy_action = MAX_ADDED_TORQUE_NM * np.tanh(
        np.maximum(
            np.maximum(normalized @ student.w1 + student.b1, 0) @ student.w2 + student.b2,
            0,
        )
        @ student.w3
        + student.b3
    )
    if (
        not np.allclose(numpy_action, check.numpy()[0], atol=1e-4)
        or not np.isfinite(np_check).all()
    ):
        raise RuntimeError("trained actor NumPy/Torch parity failed")

    parent_rows = [
        _run(model, arrays, np.zeros(WEIGHTS, dtype=np.float64), FULL_JOINTS, course)
        for course in courses
    ]
    student_rows = [
        _run(model, arrays, full, FULL_JOINTS, course, actor_torque_fn=student.predict)
        for course in courses
    ]
    parent, outcome = _summary(parent_rows), _summary(student_rows)
    ratios = {
        "mean_speed": outcome["mean_ball_speed_mps"] / parent["mean_ball_speed_mps"],
        "mean_distance": outcome["mean_ball_pelvis_distance_m"]
        / parent["mean_ball_pelvis_distance_m"],
        "center_speed": student_rows[-1]["exam_ball_speed_mps"]
        / parent_rows[-1]["exam_ball_speed_mps"],
        "center_distance": student_rows[-1]["exam_ball_pelvis_distance_m"]
        / parent_rows[-1]["exam_ball_pelvis_distance_m"],
    }
    passed = bool(
        outcome["safe_count"] == 9
        and outcome["clean_foot_count"] == 9
        and outcome["nonfoot_count"] == 0
        and ratios["mean_speed"] <= 0.85
        and ratios["mean_distance"] <= 0.90
        and ratios["center_speed"] <= 0.85
        and ratios["center_distance"] <= 0.90
    )
    if (
        hash_bytes(source.read_bytes()) != source_hash
        or hash_bytes(helper.read_bytes()) != helper_hash
        or hash_bytes(student_source.read_bytes()) != student_source_hash
    ):
        raise RuntimeError("student source changed during training or CPU physics")
    output_dir.mkdir(parents=True)
    model_path = output_dir / "student.npz"
    np.savez_compressed(
        model_path,
        **{key: getattr(student, key) for key in student.__dataclass_fields__},
    )
    model_hash = hash_bytes(model_path.read_bytes())
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.cpu_receiving_torque_distill.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "helper_hash": helper_hash,
        "student_source_hash": student_source_hash,
        "warm_start_hash": warm_hash,
        "prior_report_hash": prior_hash,
        "compiled_model_hash": compiled_model_hash(model),
        "model_hash": model_hash,
        "train_courses": [list(row) for row in courses[:8]],
        "consumed_center_course": list(center),
        "reserved_fresh8": [list(row) for row in FRESH8],
        "training_samples": len(x_train),
        "validation_samples": len(x_val),
        "label_clipped_fraction": raw_clipped / raw_total,
        "best_epoch": best_epoch,
        "best_validation_mse_nm2": best_loss,
        "final_validation_mse_nm2": loss_history[-1],
        "teacher": teacher,
        "parent": parent,
        "student": outcome,
        "ratios": ratios,
        "training_gate_passed": passed,
        "teacher_used_during_student_exam": False,
        "fresh8_opened": False,
        "shared_world_qualified": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--fidelity", required=True, type=Path)
    parser.add_argument("--warm-start", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = train(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "report_hash",
                    "model_hash",
                    "training_samples",
                    "validation_samples",
                    "label_clipped_fraction",
                    "best_epoch",
                    "best_validation_mse_nm2",
                    "student",
                    "ratios",
                    "training_gate_passed",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
