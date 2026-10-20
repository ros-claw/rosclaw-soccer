"""Curate closed own-state corrections and empirical successes, no NN training."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--expected-inputs-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or digest(a.inputs) != a.expected_inputs_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("new frozen curator and exact inputs declaration required")
    inputs = json.loads(a.inputs.read_text())
    roots = {}
    pins = {str(a.inputs): a.expected_inputs_hash}
    for role in ("bank", "parent", "baseline_training", "labels", "exploration"):
        entries = inputs[role] if role in ("labels", "exploration") else [inputs[role]]
        roots[role] = []
        for entry in entries:
            root = Path(entry["root"])
            if entry.get("unit"):
                wait_for_terminal_collection(entry["unit"])
            if digest(root / "complete.json") != entry["complete_hash"]:
                raise ValueError("closed input identity differs: " + role)
            d = json.loads((root / "declaration.json").read_text())
            pins.update(d["input_file_hashes"])
            pins.update({str(f): digest(f) for f in root.rglob("*") if f.is_file()})
            roots[role].append(root)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", Path(inputs["core_source"])):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.motion_memory_consolidation_bank.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            inputs_hash=a.expected_inputs_hash,
            optimizer_steps=0,
            world_integrations=0,
            only_training_courses_in_output=True,
            future_reference_in_actor_inputs=False,
            teacher_timeout_advice_not_supervised=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.motion_learning_windows import correction_window, success_memory_window
    from scripts.rsi_motion_student_rollout import validate_motion_recording
    from scripts.rsi_motor_student_rollout import exact_array

    bank = roots["bank"][0]
    bd = json.loads((bank / "declaration.json").read_text())
    bc = json.loads((bank / "complete.json").read_text())
    parent = roots["parent"][0]
    pc = json.loads((parent / "complete.json").read_text())
    tc = json.loads((roots["baseline_training"][0] / "complete.json").read_text())
    if (
        len(bd["rows"]) != 40
        or len(pc["results"]) != 80
        or pc["seed"] != 29
        or tc["optimizer_steps"] != 3000
        or tc["model_kind"] != "motion_intention_fsq64"
    ):
        raise ValueError("original complete40/80 and fixed BC baseline required")
    if digest(bank / bc["artifact"]) != bc["artifact_hash"]:
        raise ValueError("nominal bank artifact differs")
    with np.load(bank / bc["artifact"], allow_pickle=False) as z:
        nominal_x, nominal_y, train, eligible = (
            z[k].copy() for k in ("features", "motion_tokens", "train", "actor_eligible")
        )
    if (
        train.shape != (40,)
        or train.dtype != np.bool_
        or eligible.shape != (40,)
        or not np.array_equal(train, np.asarray([r["split"] == "train" for r in bd["rows"]]))
    ):
        raise ValueError("unchanged whole40 train/validation partition required")
    world_path = Path(inputs["world"])
    if pins.get(str(world_path)) != digest(world_path):
        raise ValueError("original native world inspection required")
    model = mujoco.MjModel.from_binary_path(str(world_path))
    feet = [
        model.body(s + t).id
        for s in ("left", "right")
        for t in ("_ankle_roll_link", "_ankle_pitch_link")
    ]
    root_id = model.body_rootid[model.body("pelvis").id]
    nonfeet = np.flatnonzero(
        (model.body_rootid == root_id) & ~np.isin(np.arange(model.nbody), feet)
    )
    if model.nq != 43 or model.nv != 41 or not len(nonfeet):
        raise ValueError("original native body classification required")
    labels = {}
    for root in roots["labels"]:
        d = json.loads((root / "declaration.json").read_text())
        c = json.loads((root / "complete.json").read_text())
        if (
            c["status"] != "OWN_STATE_TEACHER_LABELS_NATIVE_VERIFIED"
            or d["producer_hash"] != inputs["parent"]["complete_hash"]
            or d["bank_hash"] != inputs["bank"]["complete_hash"]
        ):
            raise ValueError("teacher labels must replay this original parent")
        for row in c["results"]:
            i = row["course_index"]
            folder = root / f"course{i:02d}"
            if (
                i in labels
                or not row["own_student_physics_bit_exact"]
                or not row["own_decoder_history930_bit_exact"]
                or row["teacher_advice_executed"]
                or row["native_steps"] != 3000
                or row["labels"] != 300
                or digest(folder / "labels.npz") != row["artifact_hash"]
            ):
                raise ValueError("whole unique native-verified label course required")
            labels[i] = (folder / "labels.npz", row)
    if set(labels) != set(range(40)):
        raise ValueError("all40 label outcomes required, no incomplete bank")
    rows, arrays = [], []

    def append(kind, index, x, y, mask, weight, source):
        if (
            not train[index]
            or x.shape != (300, 108)
            or y.shape != (300, 64)
            or x.dtype != np.float32
            or y.dtype != np.float32
            or mask.shape != (300,)
            or mask.dtype != np.bool_
            or mask[:20].any()
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or not np.array_equal(y * 16, np.round(y * 16))
            or y.min() < -1
            or y.max() > 0.9375
        ):
            raise ValueError(
                "training-only current108/executedFSQ64 and explicit owned mask required"
            )
        rows.append(
            dict(
                kind=kind,
                course_index=index,
                supervised_frames=int(mask.sum()),
                weight=weight,
                source_artifact=str(source),
                source_hash=digest(source),
            )
        )
        arrays.append((x.copy(), y.copy(), mask.copy(), np.float32(weight)))

    full = np.asarray([False] * 20 + [True] * 280, dtype=np.bool_)
    parent_success = set()
    for i in range(40):
        row = pc["results"][40 + i]
        if row["variant"] != "trained" or row["course"] != bd["rows"][i]["course"]:
            raise ValueError("ordered original parent course identity required")
        if row["score"]["qualified"]:
            parent_success.add(i)
        if not train[i]:
            continue
        path = parent / f"trained-course{i:02d}/native.npz"
        if digest(path) != row["artifact_hash"] or row["weights_hash"] != tc["checkpoint_hash"]:
            raise ValueError("actual original baseline checkpoint/trace differs")
        with np.load(path, allow_pickle=False) as z:
            own = {k: z[k].copy() for k in z.files}
        validate_motion_recording(own, nbody=model.nbody)
        label_path, label_row = labels[i]
        if (
            label_row["source_student_hash"] != row["artifact_hash"]
            or label_row["split"] != "train"
        ):
            raise ValueError("own teacher label/student binding differs")
        with np.load(label_path, allow_pickle=False) as z:
            lx, ly = z["features"].copy(), z["teacher_motion_token"].copy()
        exact_array(lx, own["student_features"])
        if eligible[i] and not row["score"]["qualified"]:
            append(
                "nominal_qualified_teacher",
                i,
                nominal_x[i],
                nominal_y[i],
                full,
                1.0,
                bank / bc["artifact"],
            )
        if row["score"]["qualified"]:
            mask = success_memory_window(
                first_foot_frame=row["score"]["first_foot_contact_global_frame"],
                qualified=True,
                full_retention=True,
            )
            append(
                "parent_success_retention",
                i,
                own["student_features"],
                own["student_motion_token"],
                mask,
                2.0,
                path,
            )
        else:
            bad = np.flatnonzero(
                own["ball_contact_force_by_body_n"][:, :, nonfeet].max(axis=(1, 2)) > 1
            )
            mask = correction_window(
                first_foot_frame=row["score"]["first_foot_contact_global_frame"],
                first_nonfoot_frame=int(bad[0]) if len(bad) else None,
                teacher_handoff_frame=label_row["teacher_report"]["handoff_frame"],
                parent_qualified=False,
            )
            append("own_state_approach_advice", i, lx, ly, mask, 0.25, label_path)
    successes = []
    for root in roots["exploration"]:
        d = json.loads((root / "declaration.json").read_text())
        c = json.loads((root / "complete.json").read_text())
        if (
            d["mode"] != "collect"
            or d["base_weights_hash"] != tc["checkpoint_hash"]
            or d["bank_hash"] != inputs["bank"]["complete_hash"]
        ):
            raise ValueError("actual baseline on-policy exploration required")
        for row in c["results"]:
            if not row["score"]["qualified"]:
                continue
            i = row["course_index"]
            if (
                not train[i]
                or row["split"] != "train"
                or i in successes
                or not row["prefix_diagnostics"]["post_boundary_control_can_cause_first_touch"]
            ):
                raise ValueError("unique causal training success required")
            path = root / f"course{i:02d}/native.npz"
            if digest(path) != row["native_hash"] or row["native_steps"] != 6000:
                raise ValueError("exploration native+replay commitment differs")
            with np.load(path, allow_pickle=False) as z:
                own = {k: z[k].copy() for k in z.files}
            validate_motion_recording(own, nbody=model.nbody)
            mask = success_memory_window(
                first_foot_frame=row["score"]["first_foot_contact_global_frame"],
                qualified=True,
                full_retention=False,
            )
            append(
                "empirical_successful_contact",
                i,
                own["student_features"],
                own["student_motion_token"],
                mask,
                2.0,
                path,
            )
            successes.append(i)
    if (
        len(rows) != 42
        or parent_success != {0, 10}
        or set(successes) != {6, 15}
        or {r["course_index"] for r in rows} != set(np.flatnonzero(train))
    ):
        raise ValueError("preregistered42 sequences/32train courses and empirical4 memories differ")
    np.savez_compressed(
        a.output / "consolidation.npz",
        features=np.stack([r[0] for r in arrays]),
        motion_tokens=np.stack([r[1] for r in arrays]),
        mask=np.stack([r[2] for r in arrays]),
        weights=np.asarray([r[3] for r in arrays], dtype=np.float32),
    )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="MOTION_CONSOLIDATION_DATA_CLOSED",
            source_head=a.expected_head,
            rows=rows,
            sequences=len(rows),
            unique_training_courses=32,
            validation_sequences=0,
            artifact="consolidation.npz",
            artifact_hash=digest(a.output / "consolidation.npz"),
            base_training_hash=inputs["baseline_training"]["complete_hash"],
            base_weights_hash=tc["checkpoint_hash"],
            original_bank_hash=inputs["bank"]["complete_hash"],
            new_native_steps=0,
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
