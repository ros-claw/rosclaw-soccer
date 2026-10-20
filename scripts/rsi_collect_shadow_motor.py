"""SIM-only same-state teacher collection; assisted performance is not student skill."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "world", "model-root", "motion-csv", "core-source", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-bank-hash", required=True)
    parser.add_argument(
        "--mode", choices=("identity", "collect", "pipeline", "verify"), required=True
    )
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--expected-training-hash")
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--collection-unit")
    parser.add_argument("--expected-collection-source")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != args.expected_head
        or args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    ):
        raise ValueError("clean frozen source and unused output required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if args.mode == "pipeline":
        import sys

        args.output.mkdir(parents=True, exist_ok=False)
        for mode in ("identity", "collect"):
            command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode]
            for name, value in vars(args).items():
                if name not in ("mode", "output") and value is not None:
                    command.extend(("--" + name.replace("_", "-"), str(value)))
            command.extend(("--output", str(args.output / mode)))
            subprocess.run(command, check=True)
            proof = json.loads((args.output / mode / "complete.json").read_text())
            if mode == "identity" and proof["status"] != "SHADOW_IDENTITY_EXACT":
                raise ValueError("shadow interface identity proof required before collection")
        from scripts.rsi_atomic_artifacts import write_once

        write_once(
            args.output / "complete.json",
            dict(
                status="IDENTITY_THEN_ASSISTED_COLLECTION_COMPLETE",
                identity_hash=digest(args.output / "identity/complete.json"),
                collection_hash=digest(args.output / "collect/complete.json"),
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        return
    if digest(args.bank / "complete.json") != args.expected_bank_hash:
        raise ValueError("all40 bank identity differs")
    bank = json.loads((args.bank / "declaration.json").read_text())
    pins = dict(bank["input_file_hashes"])
    pins.update({str(p): digest(p) for p in args.bank.iterdir() if p.is_file()})
    if args.mode == "verify":
        if (
            args.collection is None
            or not args.collection_unit
            or not args.expected_collection_source
        ):
            raise ValueError("independent review requires an explicit closed collection")
        terminal = dict(
            line.split("=", 1)
            for line in subprocess.check_output(
                [
                    "systemctl",
                    "--user",
                    "show",
                    args.collection_unit,
                    "-p",
                    "MainPID",
                    "-p",
                    "SubState",
                    "-p",
                    "Result",
                    "-p",
                    "ExecMainStatus",
                ],
                text=True,
            ).splitlines()
        )
        if terminal != dict(MainPID="0", SubState="exited", Result="success", ExecMainStatus="0"):
            raise ValueError("all40 collection must actually have closed successfully")
        prior_collection = json.loads((args.collection / "complete.json").read_text())
        prior_declaration = json.loads((args.collection / "declaration.json").read_text())
        if (
            prior_collection["status"] != "ASSISTED_DAGGER_COLLECTION_NOT_STUDENT_QUALIFICATION"
            or len(prior_collection["results"]) != 40
            or prior_collection["native_steps"] != 120000
            or prior_collection["optimizer_steps"] != 0
            or prior_declaration["source_head"] != args.expected_collection_source
            or prior_declaration["bank_hash"] != args.expected_bank_hash
            or prior_declaration["training_hash"] != args.expected_training_hash
            or prior_declaration["beta"] != 0.9
        ):
            raise ValueError("complete original fixed-beta same-state collection required")
        if [r["course"] for r in prior_collection["results"]] != [
            r["course"] for r in bank["rows"]
        ]:
            raise ValueError("all40 ordered course identities required")
        pins.update(prior_declaration["input_file_hashes"])
        pins.update({str(p): digest(p) for p in args.collection.rglob("*") if p.is_file()})
        # Only runner orchestration differs; policy, feature, handoff and physics
        # implementation must remain byte-identical to the original producer.
        for p in (Path.cwd() / "src").rglob("*.py"):
            old = [
                h
                for name, h in prior_declaration["input_file_hashes"].items()
                if name.endswith("/" + str(p.relative_to(Path.cwd())))
            ]
            if digest(p) not in old:
                raise ValueError("review policy/physics source differs from collection")
    check_pins(pins)
    for p in (args.world, args.motion_csv):
        if pins.get(str(p)) != digest(p):
            raise ValueError("unchanged world and source motion required")
    for name in (
        "planner_sonic.onnx",
        "low_latency/model_encoder.onnx",
        "low_latency/model_decoder.onnx",
        "low_latency/observation_config.yaml",
    ):
        p = args.model_root / name
        if pins.get(str(p)) != digest(p):
            raise ValueError("unchanged foundation required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.bilateral_entry import contact_foot_context
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.full_entry_forecast import (
        EntryCommandLaw,
        nominal_prefix20,
        run_entry_suffix,
        score_full_entry,
    )
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motor_student_rollout import exact_array, prediction

    choices = [Path(p) for p in pins if Path(p).name == "entry-affordance.json"]
    if len(choices) != 1:
        raise ValueError("one original entry prior required")
    stored = json.loads(choices[0].read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("entry prior differs")
    policy = None
    if args.mode in ("collect", "verify"):
        if (
            args.train_root is None
            or digest(args.train_root / "complete.json") != args.expected_training_hash
        ):
            raise ValueError("explicit trained student identity required")
        tc = json.loads((args.train_root / "complete.json").read_text())
        if tc["bank_complete_hash"] != args.expected_bank_hash:
            raise ValueError("student trained on another bank")
        weight = args.train_root / tc["checkpoint"]
        if digest(weight) != tc["checkpoint_hash"]:
            raise ValueError("student weights differ")
        pins.update({str(p): digest(p) for p in args.train_root.iterdir() if p.is_file()})
        import torch

        from rosclaw_soccer.rsi.neural_motor_student import create_motor_student

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        payload = torch.load(weight, map_location="cpu", weights_only=True)
        policy = create_motor_student()
        policy.load_state_dict(payload["state_dict"], strict=True)
        policy.eval()
    records = bank["rows"] if args.mode != "identity" else [bank["rows"][i] for i in (2, 22)]
    if [r["course"]["course_index"] for r in bank["rows"]] != list(range(40)):
        raise ValueError("entire ordered40 declaration required")
    model = mujoco.MjModel.from_binary_path(str(args.world))
    source = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            source_head=head,
            mode=args.mode,
            input_file_hashes=pins,
            beta=1.0 if policy is None else 0.9,
            bank_hash=args.expected_bank_hash,
            training_hash=args.expected_training_hash,
            student_only_performance=False,
            hardware_authorized=False,
        ),
    )
    results = []
    for record in records:
        item = record["course"]
        raw = Path(record["source_artifact"])
        if digest(raw) != record["source_artifact_hash"]:
            raise ValueError("expert archive differs")
        producer = raw.parent.parent
        for name in ("complete.json", "declaration.json"):
            if pins.get(str(producer / name)) != digest(producer / name):
                raise ValueError("selected expert control provenance not pinned")
        pc = json.loads((producer / "complete.json").read_text())
        r = next(r for r in pc["results"] if r["course"] == item)
        selected = r["selected"]
        option = r["candidates"][selected["selected_index"]]
        poses, affordance, _ = contact_foot_context(
            source, prior, selected["intended_reference_foot"]
        )
        nav = G1SonicNavigation(
            args.model_root,
            item["agent_id"],
            SonicNavigationConfig(
                maximum_frames=300,
                model_variant="low_latency",
                inference_threads=1,
                record_default_planner_events=True,
            ),
        )
        body = G1TargetEpisode(
            model,
            world_hash=digest(args.world),
            kp=nav.backend.kp.copy(),
            kd=nav.backend.kd.copy(),
            course=tuple(item["course"]),
            defaults=G1SonicRunupController.default_angles.copy(),
            record_all_ball_contacts=True,
        )
        prefix = nominal_prefix20(nav, body, prior)
        with np.load(raw, allow_pickle=False) as z:
            for k in prefix:
                exact_array(prefix[k], z[k][:20])
        hidden, previous = None, np.zeros(29, dtype=np.float32)
        labels = []
        if policy is not None:
            for f in range(20):
                feature, proposed, _, hidden, _ = prediction(
                    policy,
                    prefix["canonical_qpos"][f],
                    prefix["canonical_qvel"][f],
                    previous,
                    f,
                    hidden,
                )
                previous = prefix["neural_action_isaac"][f].copy()
                labels.append((feature, proposed, previous.copy(), previous.copy()))

        def provider(teacher, obs, labels=labels):
            nonlocal hidden, previous
            advice = teacher.prepare_shadow_action(obs)
            if policy is None:
                consumed = advice.copy()
            else:
                feature, proposed, _, hidden, _ = prediction(
                    policy, np.asarray(obs.qpos), np.asarray(obs.qvel), previous, obs.frame, hidden
                )
                consumed = (np.float32(0.9) * advice + np.float32(0.1) * proposed).astype(
                    np.float32
                )
                labels.append((feature, proposed, advice.copy(), consumed.copy()))
            target = teacher.commit_batched_action(consumed)
            previous = consumed.copy()
            return target

        suffix, report = run_entry_suffix(
            nav,
            body,
            model_root=args.model_root,
            source_poses=poses,
            source_evidence_hash=hash_json(json.loads((producer / "declaration.json").read_text())),
            affordance=affordance,
            law=EntryCommandLaw(**selected["report"]["law"]),
            contact_reference_offset_m=tuple(option["offset_root_m"]),
            motor_provider=provider,
        )
        arrays = {k: np.concatenate((prefix[k], suffix[k])) for k in prefix}
        if body.integrated_steps != 3000 or body.attempted_steps != 3000:
            raise ValueError("full H300 native integration required")
        if policy is None:
            with np.load(raw, allow_pickle=False) as z:
                for k in arrays:
                    exact_array(arrays[k], z[k])
            if report["handoff_frame"] is None:
                raise ValueError("identity probe must cover pose-reference handoff")
        else:
            if len(labels) != 300:
                raise ValueError("every pre-step state and consumed action required")
            arrays["consumed_action_isaac"] = arrays.pop("neural_action_isaac")
            arrays["teacher_reference_poses"] = arrays.pop("encoder_reference_poses")
            for i, key in enumerate(
                (
                    "student_features",
                    "student_proposed_action_isaac",
                    "teacher_advice_action_isaac",
                    "mixed_consumed_action_isaac",
                )
            ):
                arrays[key] = np.stack([label[i] for label in labels])
            exact_array(arrays["consumed_action_isaac"], arrays["mixed_consumed_action_isaac"])
        folder = args.output / f"course{item['course_index']:02d}"
        folder.mkdir()
        if args.mode == "verify":
            original_folder = args.collection / folder.name
            original = prior_collection["results"][item["course_index"]]
            if digest(original_folder / "trace.npz") != original["artifact_hash"]:
                raise ValueError("mixed native source artifact differs")
            with np.load(original_folder / "trace.npz", allow_pickle=False) as z:
                if set(z.files) != set(arrays):
                    raise ValueError("mixed teacher/student recording fields differ")
                for k in arrays:
                    exact_array(arrays[k], z[k])
        np.savez_compressed(folder / "trace.npz", **arrays)
        score = score_full_entry(model, suffix)
        pd = diagnose_entry_prefix(model, prefix)
        score["qualified"] = bool(score["qualified"] and pd["prefix_physical_safe"])
        if args.mode == "verify" and (
            score != original["score"]
            or pd != original["prefix_diagnostics"]
            or report != original["report"]
        ):
            raise ValueError("mixed native qualification or controller provenance differs")
        result = dict(
            course=item,
            split=record["split"],
            mode=args.mode,
            score=score,
            prefix_diagnostics=pd,
            report=report,
            native_steps=body.integrated_steps,
            artifact_hash=digest(folder / "trace.npz"),
            student_only_performance=False,
        )
        write_once(folder / "complete.json", result)
        results.append(result)
        print(
            json.dumps(
                dict(course=item["course_index"], qualified=score["qualified"], mode=args.mode)
            ),
            flush=True,
        )
    check_pins(pins)
    write_once(
        args.output / "complete.json",
        dict(
            status=(
                "SHADOW_IDENTITY_EXACT"
                if policy is None
                else "ALL40_MIXED_NN_AND_NATIVE_REPLAY_VERIFIED"
                if args.mode == "verify"
                else "ASSISTED_DAGGER_COLLECTION_NOT_STUDENT_QUALIFICATION"
            ),
            results=results,
            native_steps=sum(r["native_steps"] for r in results),
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
