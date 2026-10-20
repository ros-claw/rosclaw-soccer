"""Full80 motion-intention/muscle native evaluation or independent neural replay."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("evaluate", "replay"), required=True)
    for key in ("train-root", "bank", "world", "model-root", "core-source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "expected-head",
        "train-unit",
        "expected-training-hash",
        "expected-training-source",
        "expected-bank-hash",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--producer", type=Path)
    p.add_argument("--producer-unit")
    p.add_argument("--expected-producer-source")
    p.add_argument("--wait-for-producer", action="store_true")
    a = p.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != a.expected_head
        or a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    ):
        raise ValueError("clean frozen motion evaluator and unused output required")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    wait_for_terminal_collection(a.train_unit)
    if (
        digest(a.train_root / "complete.json") != a.expected_training_hash
        or digest(a.bank / "complete.json") != a.expected_bank_hash
    ):
        raise ValueError("closed training/data identities differ")
    tc = json.loads((a.train_root / "complete.json").read_text())
    td = json.loads((a.train_root / "declaration.json").read_text())
    bd = json.loads((a.bank / "declaration.json").read_text())
    if (
        tc["source_head"] != a.expected_training_source
        or td["source_head"] != a.expected_training_source
        or tc["optimizer_steps"] != 3000
        or tc["model_kind"] != "motion_intention_fsq64"
        or tc["bank_complete_hash"] != a.expected_bank_hash
        or tc["parameter_change_l2_squared"] <= 0
        or td["seed"] not in (17, 29, 43)
        or [r["course"]["course_index"] for r in bd["rows"]] != list(range(40))
    ):
        raise ValueError("fixed trained motion model and complete40 required")
    pins = dict(td["input_file_hashes"])
    pins.update(bd["input_file_hashes"])
    for root in (a.bank, a.train_root):
        pins.update({str(f): digest(f) for f in root.iterdir() if f.is_file()})
    producer_complete = None
    if a.mode == "replay":
        if a.producer is None or not a.producer_unit or not a.expected_producer_source:
            raise ValueError("explicit closed full80 producer required")
        wait_for_terminal_collection(a.producer_unit, wait=a.wait_for_producer)
        pd = json.loads((a.producer / "declaration.json").read_text())
        producer_complete = json.loads((a.producer / "complete.json").read_text())
        if (
            pd["source_head"] != a.expected_producer_source
            or pd["mode"] != "evaluate"
            or pd["training_hash"] != a.expected_training_hash
            or pd["bank_hash"] != a.expected_bank_hash
            or producer_complete["seed"] != td["seed"]
            or len(producer_complete["results"]) != 80
        ):
            raise ValueError("same fixed seed/weights/data and all80 evaluation required")
        pins.update(pd["input_file_hashes"])
        pins.update({str(f): digest(f) for f in a.producer.rglob("*") if f.is_file()})
        for row, (variant, index) in zip(
            producer_complete["results"],
            [(v, i) for v in ("initial", "trained") for i in range(40)],
            strict=True,
        ):
            folder = a.producer / f"{variant}-course{index:02d}"
            if (
                row["variant"] != variant
                or row["course"] != bd["rows"][index]["course"]
                or row["split"] != bd["rows"][index]["split"]
                or row != json.loads((folder / "complete.json").read_text())
            ):
                raise ValueError("ordered full80 identity/receipt mismatch")
            if row["status"] == "COMPLETE" and (
                digest(folder / "native.npz") != row["artifact_hash"] or row["native_steps"] != 3000
            ):
                raise ValueError("complete physical artifact/counter differs")
    check_pins(pins)
    for relative in (
        "src/rosclaw_soccer/rsi/neural_motion_intention.py",
        "src/rosclaw_soccer/rsi/foundation_motor_codec.py",
        "scripts/rsi_motion_student_rollout.py",
    ):
        if {h for name, h in td["input_file_hashes"].items() if name.endswith("/" + relative)} != {
            digest(Path.cwd() / relative)
        }:
            raise ValueError("trained and smoked intention/muscle execution code changed")
    for path in [a.world] + [
        a.model_root / name
        for name in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]:
        if pins.get(str(path)) != digest(path):
            raise ValueError("original native world and frozen muscle weights required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.motion_intention_native.v1",
            mode=a.mode,
            source_head=head,
            input_file_hashes=pins,
            training_hash=a.expected_training_hash,
            bank_hash=a.expected_bank_hash,
            seed=td["seed"],
            producer_hash=digest(a.producer / "complete.json") if a.producer else None,
            teacher_encoder_and_planner_disabled_on_muscle_backend=True,
            foundation_prefix_neural_math_replayed=False if a.mode == "replay" else None,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    if producer_complete is not None and any(
        r["status"] != "COMPLETE" for r in producer_complete["results"]
    ):
        write_once(
            a.output / "complete.json",
            dict(
                status="FULL_MOTION_QUALIFICATION_REJECTED_EXECUTION_FAILURE",
                all80_verified=False,
                new_native_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        return
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig, G1SonicRunupController
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.bilateral_entry import observed_contact_foot
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.foundation_motor_codec import FoundationMotorCodec
    from rosclaw_soccer.rsi.full_entry_forecast import nominal_prefix20, score_full_entry
    from rosclaw_soccer.rsi.neural_motion_intention import create_motion_intention_student
    from scripts.rsi_motion_student_rollout import replay_motion_student, rollout_motion_student

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = {}
    weights = {}
    for variant, path, expected, steps in (
        ("initial", a.train_root / "initial.pt", tc["initial_checkpoint_hash"], 0),
        ("trained", a.train_root / tc["checkpoint"], tc["checkpoint_hash"], 3000),
    ):
        if digest(path) != expected:
            raise ValueError("motion weights changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload["seed"] != td["seed"]
            or payload["optimizer_steps"] != steps
            or payload["model_kind"] != "motion_intention_fsq64"
            or payload["source_head"] != a.expected_training_source
            or payload["bank_hash"] != a.expected_bank_hash
        ):
            raise ValueError("motion checkpoint provenance differs")
        m = create_motion_intention_student()
        m.load_state_dict(payload["state_dict"], strict=True)
        m.eval()
        models[variant] = m
        weights[variant] = expected
    world = mujoco.MjModel.from_binary_path(str(a.world))
    backend = G1SonicRunupController(
        a.model_root, G1SonicRunupConfig(model_variant="low_latency"), inference_threads=1
    )
    prior_paths = [Path(name) for name in pins if Path(name).name == "entry-affordance.json"]
    if len(prior_paths) != 1:
        raise ValueError("one original Foundation prefix prior required")
    stored = json.loads(prior_paths[0].read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("prefix prior changed")
    results = []
    for variant in ("initial", "trained"):
        for index, record in enumerate(bd["rows"]):
            item = record["course"]
            folder = a.output / f"{variant}-course{index:02d}"
            folder.mkdir()
            body = G1TargetEpisode(
                world,
                world_hash=digest(a.world),
                kp=backend.kp.copy(),
                kd=backend.kd.copy(),
                course=tuple(item["course"]),
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )
            progress = {}
            try:
                codec = FoundationMotorCodec(backend)
                if a.mode == "evaluate":
                    nav = G1SonicNavigation(
                        a.model_root,
                        item["agent_id"],
                        SonicNavigationConfig(
                            maximum_frames=300,
                            model_variant="low_latency",
                            inference_threads=1,
                            record_default_planner_events=True,
                        ),
                    )
                    prefix = nominal_prefix20(nav, body, prior)
                    del nav
                    raw_path = Path(record["source_artifact"])
                    if digest(raw_path) != record["source_artifact_hash"]:
                        raise ValueError("teacher prefix archive changed")
                    with np.load(raw_path, allow_pickle=False) as z:
                        for k in prefix:
                            np.testing.assert_array_equal(prefix[k], z[k][:20])
                    trace = rollout_motion_student(
                        models[variant], codec, body, prefix, progress=progress
                    )
                    np.savez_compressed(folder / "native.npz", **trace)
                    proof = None
                else:
                    source = producer_complete["results"][
                        (0 if variant == "initial" else 40) + index
                    ]
                    if source["weights_hash"] != weights[variant]:
                        raise ValueError("actual consumed motion weights differ")
                    with np.load(
                        a.producer / f"{variant}-course{index:02d}" / "native.npz",
                        allow_pickle=False,
                    ) as z:
                        trace = {k: z[k].copy() for k in z.files}
                    proof = replay_motion_student(models[variant], codec, body, trace)
                if body.integrated_steps != 3000 or body.attempted_steps != 3000:
                    raise ValueError("full H300 native execution required")
                prefix_diagnostics = diagnose_entry_prefix(
                    world, {k: v[:20] for k, v in trace.items()}
                )
                score = score_full_entry(world, {k: v[20:] for k, v in trace.items()})
                score["original_suffix_qualified"] = score["qualified"]
                score["common_prefix_physical_safe"] = prefix_diagnostics["prefix_physical_safe"]
                if not prefix_diagnostics["prefix_physical_safe"]:
                    score["qualified"], score["cost"], score["physical_prefix_and_suffix_safe"] = (
                        False,
                        None,
                        False,
                    )
                foot = observed_contact_foot(world, trace, global_start_frame=0)
                causal = bool(
                    score["qualified"]
                    and prefix_diagnostics["post_boundary_control_can_cause_first_touch"]
                )
                if a.mode == "replay" and (
                    score != source["score"]
                    or prefix_diagnostics != source["prefix_diagnostics"]
                    or foot != source["observed_contact_foot"]
                    or causal != source["causal_student_first_touch"]
                    or body.review() != source["body"]
                ):
                    raise ValueError("native score/body/touch attribution differs")
                row = dict(
                    variant=variant,
                    course=item,
                    split=record["split"],
                    status="COMPLETE",
                    score=score,
                    prefix_diagnostics=prefix_diagnostics,
                    observed_contact_foot=foot,
                    causal_student_first_touch=causal,
                    native_steps=3000,
                    weights_hash=weights[variant],
                    body=body.review(),
                    teacher_encoder_and_planner_calls_after_prefix=0,
                )
                if a.mode == "evaluate":
                    row["artifact_hash"] = digest(folder / "native.npz")
                else:
                    row["replay"] = proof
            except Exception as error:
                rows = progress.get("rows", [])
                if rows:
                    np.savez_compressed(
                        folder / "partial.npz",
                        **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]},
                    )
                np.savez_compressed(
                    folder / "last-native.npz",
                    qpos=body.data.qpos.copy(),
                    qvel=body.data.qvel.copy(),
                )
                row = dict(
                    variant=variant,
                    course=item,
                    split=record["split"],
                    status="EXECUTION_FAILED",
                    error=repr(error),
                    native_steps=body.integrated_steps,
                    attempted_steps=body.attempted_steps,
                    score=None,
                    weights_hash=weights[variant],
                )
                if a.mode == "replay":
                    write_once(folder / "failure.json", row)
                    raise
            write_once(folder / "complete.json", row)
            results.append(row)
            print(
                "MOTION_INTENTION_NATIVE",
                a.mode,
                td["seed"],
                variant,
                index,
                row["status"],
                row["score"]["qualified"] if row["score"] else False,
                flush=True,
            )
    summary = dict(
        seed=td["seed"],
        results=results,
        new_native_steps=sum(r["native_steps"] for r in results),
        initial_qualified=sum(
            r["variant"] == "initial" and r["score"] is not None and r["score"]["qualified"]
            for r in results
        ),
        trained_qualified=sum(
            r["variant"] == "trained" and r["score"] is not None and r["score"]["qualified"]
            for r in results
        ),
        trained_causal_qualified=sum(
            r["variant"] == "trained" and r.get("causal_student_first_touch", False)
            for r in results
        ),
        execution_failed=sum(r["status"] != "COMPLETE" for r in results),
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if a.mode == "replay" and any(
        summary[k] != producer_complete[k]
        for k in (
            "new_native_steps",
            "initial_qualified",
            "trained_qualified",
            "trained_causal_qualified",
            "execution_failed",
        )
    ):
        raise ValueError("all80 physical outcomes/counters differ")
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="ALL80_MOTION_INTENTION_MUSCLE_NATIVE_VERIFIED"
            if a.mode == "replay"
            else "ALL80_MOTION_INTENTION_EVALUATED_NOT_VERIFIED",
            **summary,
        ),
    )


if __name__ == "__main__":
    main()
