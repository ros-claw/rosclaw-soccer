"""Recompute all40 teacher neural actions before extracting64 motion labels. No physics."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "audit", "model-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("expected-head", "expected-bank-hash", "expected-audit-hash"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != a.expected_head
        or a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    ):
        raise ValueError("frozen label source and unused bank required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        digest(a.bank / "complete.json") != a.expected_bank_hash
        or digest(a.audit / "complete.json") != a.expected_audit_hash
    ):
        raise ValueError("original verified40 bank and feedback-audit identities differ")
    bd = json.loads((a.bank / "declaration.json").read_text())
    bc = json.loads((a.bank / "complete.json").read_text())
    ac = json.loads((a.audit / "complete.json").read_text())
    if (
        [r["teacher_actions_recomputed_exact"] for r in ac["results"]] != [300, 300]
        or bc["total_courses"] != 40
        or digest(a.bank / bc["artifact"]) != bc["artifact_hash"]
    ):
        raise ValueError("full audited teacher neural reconstruction and original40 required")
    pins = dict(bd["input_file_hashes"])
    for root in (a.bank, a.audit):
        pins.update({str(p): digest(p) for p in root.rglob("*") if p.is_file()})
    for name in (
        "planner_sonic.onnx",
        "low_latency/model_encoder.onnx",
        "low_latency/model_decoder.onnx",
        "low_latency/observation_config.yaml",
    ):
        path = a.model_root / name
        if pins.get(str(path)) != digest(path):
            raise ValueError("frozen SONIC muscle and motion networks required")
    check_pins(pins)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    protocol = Path.cwd() / "docs/rsi/foundation-motion-intention-preregister-20261020.zh-CN.md"
    pins[str(protocol)] = digest(protocol)
    import numpy as np

    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig, G1SonicRunupController
    from rosclaw_soccer.rsi.foundation_motor_codec import FoundationMotorCodec
    from rosclaw_soccer.rsi.neural_motor_student import recorded_student_sequence
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motor_student_rollout import exact_array

    with np.load(a.bank / bc["artifact"], allow_pickle=False) as z:
        features, targets, train, actor = (
            z[k].copy() for k in ("features", "targets", "train", "actor_eligible")
        )
    backend = G1SonicRunupController(
        a.model_root, G1SonicRunupConfig(model_variant="low_latency"), inference_threads=1
    )
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=head,
            schema="soccer.rsi.foundation_motion_labels.v1",
            input_file_hashes=pins,
            original_bank_hash=a.expected_bank_hash,
            audit_hash=a.expected_audit_hash,
            rows=bd["rows"],
            encoder_reference_is_label_only=True,
            student_input_features=108,
            new_native_steps=0,
            optimizer_steps=0,
            hardware_authorized=False,
        ),
    )
    all_tokens, results = [], []
    if [r["course"]["course_index"] for r in bd["rows"]] != list(range(40)):
        raise ValueError("all40 ordered courses required")
    for index, record in enumerate(bd["rows"]):
        path = Path(record["source_artifact"])
        if digest(path) != record["source_artifact_hash"]:
            raise ValueError("native-reviewed teacher recording differs")
        with np.load(path, allow_pickle=False) as z:
            raw = {k: z[k].copy() for k in z.files}
        x, actions = recorded_student_sequence(raw)
        exact_array(x, features[index])
        exact_array(actions, targets[index])
        codec = FoundationMotorCodec(backend)
        tokens = []
        previous = np.zeros(29, dtype=np.float32)
        for frame in range(300):
            codec.observe(
                raw["canonical_qpos"][frame], raw["canonical_qvel"][frame], previous, frame=frame
            )
            token = codec.encode_teacher_reference(raw["encoder_reference_poses"][frame])
            action = codec.decode_motion(token)
            exact_array(action, actions[frame])
            codec.commit_consumed(action)
            previous = action
            tokens.append(token)
        all_tokens.append(np.stack(tokens))
        result = dict(
            course=record["course"],
            split=record["split"],
            actor_eligible=record["actor_eligible"],
            teacher_encoder_calls=300,
            muscle_decoder_calls=300,
            actions_byte_exact=300,
        )
        results.append(result)
        write_once(a.output / f"course{index:02d}.json", result)
        print("FOUNDATION_MUSCLE_NN_RECONSTRUCTED", index, 300, flush=True)
    tokens = np.stack(all_tokens)
    selected = tokens[train & actor]
    norm = np.linalg.norm(selected, axis=2)
    artifact = a.output / "foundation-motion-bank.npz"
    np.savez_compressed(
        artifact,
        features=features,
        motion_tokens=tokens,
        muscle_actions=targets,
        train=train,
        actor_eligible=actor,
    )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="ALL40_TEACHER_NN_EXACT_MOTION_LABELS_NOT_TRAINED_POLICY",
            results=results,
            artifact=artifact.name,
            artifact_hash=digest(artifact),
            original_bank_hash=a.expected_bank_hash,
            total_courses=40,
            train_courses=int(train.sum()),
            validation_courses=int((~train).sum()),
            qualified_train_courses=int((train & actor).sum()),
            qualified_validation_courses=int((~train & actor).sum()),
            teacher_encoder_calls=12000,
            muscle_decoder_calls=12000,
            actions_byte_exact=12000,
            positive_training_token_min=float(selected.min()),
            positive_training_token_max=float(selected.max()),
            positive_training_token_norm_min=float(norm.min()),
            positive_training_token_norm_max=float(norm.max()),
            positive_training_token_norm_mean=float(norm.mean()),
            new_native_steps=0,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
