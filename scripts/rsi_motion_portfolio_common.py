"""Frozen motion portfolio evidence admission and current-state training table."""

import json
import subprocess
from pathlib import Path


def admit(a):
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or digest(a.inputs) != a.expected_inputs_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("exact frozen portfolio source/manifest and unused output required")
    spec = json.loads(a.inputs.read_text())
    if [r["id"] for r in spec["experts"]] != [
        "bc29",
        "fsq_mass",
        "memory17",
        "memory29",
        "memory43",
    ]:
        raise ValueError("preregistered ordered five memory slots required")
    bank = Path(spec["bank"]["root"])
    if digest(bank / "complete.json") != spec["bank"]["complete_hash"]:
        raise ValueError("original40 task bank differs")
    bd = json.loads((bank / "declaration.json").read_text())
    if len(bd["rows"]) != 40 or sum(r["split"] == "train" for r in bd["rows"]) != 32:
        raise ValueError("fixed32/8 development split required")
    pins = {**bd["input_file_hashes"], str(a.inputs): a.expected_inputs_hash}
    pins.update({str(f): digest(f) for f in bank.glob("*.json")})
    cases = []
    for expert in spec["experts"]:
        path = Path(expert["checkpoint"])
        if digest(path) != expert["checkpoint_hash"]:
            raise ValueError("frozen memory checkpoint differs")
        pins[str(path)] = expert["checkpoint_hash"]
        rows = {}
        if expert["format"] == "plain_fsq64":
            review = expert["independent_review"]
            wait_for_terminal_collection(review["unit"])
            root = Path(review["root"])
            if (
                digest(root / "complete.json") != review["complete_hash"]
                or json.loads((root / "complete.json").read_text())["status"]
                != "ALL80_MOTION_INTENTION_MUSCLE_NATIVE_VERIFIED"
            ):
                raise ValueError("original all80 neural/native review required")
            pins.update({str(f): digest(f) for f in root.rglob("*.json")})
        elif expert["format"] != "actorcritic_fsq64":
            raise ValueError("unknown frozen motion policy format")
        for part in expert["evaluations"]:
            wait_for_terminal_collection(part["unit"])
            root = Path(part["root"])
            if digest(root / "complete.json") != part["complete_hash"]:
                raise ValueError("closed physical evidence commitment differs")
            d = json.loads((root / "declaration.json").read_text())
            c = json.loads((root / "complete.json").read_text())
            if any(k in pins and pins[k] != h for k, h in d["input_file_hashes"].items()):
                raise ValueError("conflicting historical input identities cannot be composed")
            pins.update(d["input_file_hashes"])
            pins.update({str(f): digest(f) for f in root.glob("*.json")})
            if d["mode"] != "evaluate":
                raise ValueError("only actual evaluated policies may enter candidate portfolio")
            if expert["format"] == "actorcritic_fsq64" and (
                d["candidate_hash"] != expert["checkpoint_hash"]
                or d.get("candidate_source", d["source_head"]) != expert["training_source"]
            ):
                raise ValueError("actual consumed checkpoint/source differs")
            for row in c["results"]:
                if expert["format"] == "plain_fsq64":
                    if row["variant"] != "trained":
                        continue
                    i = row["course"]["course_index"]
                    if (
                        row["status"] != "COMPLETE"
                        or row["weights_hash"] != expert["checkpoint_hash"]
                    ):
                        raise ValueError("original baseline physical result differs")
                    folder = root / f"trained-course{i:02d}"
                    artifact_hash = row["artifact_hash"]
                else:
                    i = row["course_index"]
                    if (
                        not row["replay"]["student_neural_and_muscle_native_bit_exact"]
                        or not row["regenerated_noise_and_prequant_likelihood_bit_exact"]
                    ):
                        raise ValueError("full independent policy and native replay required")
                    folder = root / f"course{i:02d}"
                    artifact_hash = row["native_hash"]
                path = folder / "native.npz"
                if (
                    i in rows
                    or i not in range(40)
                    or row["split"] != bd["rows"][i]["split"]
                    or digest(path) != artifact_hash
                    or type(row["score"]["qualified"]) is not bool
                ):
                    raise ValueError("unique bound complete40 physical outcomes required")
                pins[str(path)] = artifact_hash
                pins[str(folder / "complete.json")] = digest(folder / "complete.json")
                rows[i] = (row, path)
        if set(rows) != set(range(40)):
            raise ValueError("every memory must have all40 outcomes, not cherry-picked successes")
        cases.append(rows)
    assets = [Path(spec["world"])] + [
        Path(spec["model_root"]) / n
        for n in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]
    if any(bd["input_file_hashes"].get(str(f)) != digest(f) for f in assets):
        raise ValueError("portfolio cannot change original native world or muscle assets")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    return spec, bd, pins, cases


def training_table(bd, cases):
    import numpy as np

    from rosclaw_soccer.rsi.motion_capability_router import motion_routing_features
    from scripts.rsi_motor_student_rollout import exact_array

    x, success, indices = [], [], []
    for i, record in enumerate(bd["rows"]):
        if record["split"] != "train":
            continue  # Do not read validation labels for any training target.
        features = []
        quality = []
        for rows in cases:
            row, path = rows[i]
            with np.load(path, allow_pickle=False) as z:
                current = z["student_features"][20].copy()
            features.append(current)
            quality.append(row["score"]["qualified"])
        for other in features[1:]:
            exact_array(other, features[0])
        x.append(motion_routing_features(features[0]))
        success.append(quality)
        indices.append(i)
    x = np.stack(x)
    success = np.asarray(success, dtype=np.float32)
    targets = success.argmax(axis=1).astype(np.int64)  # first/oldest success, else baseline
    if x.shape != (32, 8) or success.shape != (32, 5) or int(success.any(axis=1).sum()) != 7:
        raise ValueError("preregistered TRAIN-only feature/outcome matrix differs")
    return x, success, targets, indices


def load_frozen_experts(spec):
    import torch

    from rosclaw_soccer.rsi.motion_actor_critic import create_motion_actor_critic

    models = []
    for entry in spec["experts"]:
        payload = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
        if payload["source_head"] != entry["training_source"]:
            raise ValueError("original frozen motion checkpoint training source differs")
        model = create_motion_actor_critic()
        if entry["format"] == "plain_fsq64":
            model.base.load_state_dict(payload["state_dict"], strict=True)
        else:
            model.load_state_dict(payload["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        models.append(model)
    return models
