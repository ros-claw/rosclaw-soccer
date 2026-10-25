"""Recover all original full kick histories and diagnose rejected PPO readout drift."""

import argparse
import contextlib
import io
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--shard-root", type=Path, nargs=4, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--candidate-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or a.audit.is_symlink() or digest(a.audit) != a.audit_hash:
        raise ValueError("new memory output and exact complete native candidate audit required")
    audit = json.loads(a.audit.read_text())
    if audit["status"] != "ALL80_KICK_HEAD_PPO_CANDIDATE_TRIALS_NATIVE_VERIFIED" or (
        audit["candidate_report_hash"] != a.candidate_hash
        or audit["promotion_authorized"] is not False
    ):
        raise ValueError("whole paired physical audit of the exact provisional candidate required")
    from rosclaw_soccer.rsi.kick_ppo_candidate import admit_pilot

    pins, candidate_state = admit_pilot(a.candidate, a.candidate_hash)
    for name, h in audit["input_file_hashes"].items():
        if name in pins and pins[name] != h:
            raise ValueError("conflicting full-history physical identity")
        pins[name] = h
    pins[str(a.audit)] = a.audit_hash
    originals = {}
    for shard, root in enumerate(a.shard_root):
        path = root / "complete.json"
        if path.is_symlink() or pins.get(str(path)) != digest(path):
            raise ValueError("exact fully audited original producer required")
        report = json.loads(path.read_text())
        if report["candidate_report_hash"] != a.candidate_hash or (
            [r["course_index"] for r in report["results"]] != list(range(shard, 40, 4))
        ):
            raise ValueError("all40 unique original courses required")
        for item in report["results"]:
            index = item["course_index"]
            rp = root / f"course-{index:02d}" / "onnx.json"
            if pins.get(str(rp)) != digest(rp):
                raise ValueError("exact audited original ONNX record required")
            originals[index] = rp, item["rows"]["onnx"]
    if sorted(originals) != list(range(40)):
        raise ValueError("complete original40 required, no selection of good cases")
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique full-history source asset required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    import torch
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo

    from rosclaw_soccer.rsi.kick_entry_reconstruction import KickEntryReconstructor
    from rosclaw_soccer.rsi.kick_prior_actor import KICK_PRIOR_HASH, KickPriorActor
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.sim.contracts import hash_json

    torch.set_num_threads(1)
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    state, output, policy_type, mapping = _load_robonaldo(
        one("/policy/robonaldo/FreeKick.py").parents[2]
    )
    with contextlib.redirect_stdout(io.StringIO()):
        policy = policy_type(state(29), output(29))
    reconstructor = KickEntryReconstructor(model, layout, policy, mapping)
    base = KickPriorActor.from_onnx(
        one("/policy/robonaldo/model/policy-obs-aic.onnx"), expected_hash=KICK_PRIOR_HASH
    ).eval()
    changed = KickPriorActor.from_onnx(
        one("/policy/robonaldo/model/policy-obs-aic.onnx"), expected_hash=KICK_PRIOR_HASH
    ).eval()
    changed.load_state_dict(candidate_state["actor"], strict=True)
    if hash_json(layout.profile()) != candidate_state["body_hash"]:
        raise ValueError("memory source body differs from policy contract")
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            original_courses=40,
            every_neural_PD_target_must_reconstruct_exactly=True,
            no_admission_failures_preserved_separately=True,
            new_physical_steps=0,
            optimizer_updates=0,
            counterfactual_drift_on_original_full_trajectories_not_new_execution=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    results, neural_rows, retained_rows = [], 0, 0
    try:
        for index in range(40):
            rp, row = originals[index]
            if hash_json(json.loads(rp.read_text())) != hash_json(row):
                raise ValueError("audited original row join differs")
            if row["entry_control"] is None:
                results.append(
                    dict(course_index=index, status="NO_ADMISSION_PRESERVED", neural_rows=0)
                )
                continue
            with np.load(rp.parent / row["artifact"]["path"]) as saved:
                memory = reconstructor.reconstruct_trace(row, dict(saved))
            obs = torch.from_numpy(memory["observations"])
            with torch.no_grad():
                old_raw, new_raw = base(obs).numpy(), changed(obs).numpy()
            delta = new_raw - old_raw
            kl = 0.5 * np.square(delta / 0.015).sum(axis=1)
            low, high = policy.action_clip_lo_il, policy.action_clip_hi_il
            old_pd = (
                policy.default_q_mj
                + policy.action_scale_mj * np.clip(old_raw, low, high)[:, mapping]
            )
            new_pd = (
                policy.default_q_mj
                + policy.action_scale_mj * np.clip(new_raw, low, high)[:, mapping]
            )
            keep = row["outcome"]["task_pass"] and row["healthy_full_horizon"]
            neural_rows += len(obs)
            retained_rows += len(obs) if keep else 0
            artifact = a.output / f"memory-{index:02d}.npz"
            np.savez_compressed(artifact, **memory)
            result = dict(
                course_index=index,
                source_record=str(rp),
                source_record_hash=digest(rp),
                status="ALL_NEURAL_TARGETS_EXACT",
                neural_rows=len(obs),
                original_success_to_retain=keep,
                entry_readout_raw_max_change=float(abs(delta[0]).max()),
                whole_history_readout_raw_max_change=float(abs(delta).max()),
                whole_history_readout_PD_max_change=float(abs(new_pd - old_pd).max()),
                whole_history_original_prior_KL_mean=float(kl.mean()),
                whole_history_original_prior_KL_max=float(kl.max()),
                memory=dict(path=artifact.name, hash=digest(artifact)),
            )
            write_once(a.output / f"record-{index:02d}.json", result)
            results.append(result)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="ALL_ORIGINAL40_FULL_NEURAL_HISTORIES_EXACTLY_RECONSTRUCTED_NOT_TRAINED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                candidate_report_hash=a.candidate_hash,
                results=results,
                original_courses=40,
                neural_rows=neural_rows,
                successful_preservation_rows=retained_rows,
                new_physical_steps=0,
                optimizer_updates=0,
                physical_football_gain_proven=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                completed_contexts=len(results),
                no_automatic_retry=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
