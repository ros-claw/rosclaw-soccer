"""Re-execute recorded motion-student physics while querying a shadow teacher.

Labels only: teacher proposals NEVER replace the recorded student actions.
No training, teacher qualification, candidate activation or hardware authority.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "producer", "world", "model-root", "motion-csv", "core-source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-bank-hash", "expected-producer-hash", "producer-unit"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--indices", nargs="+", type=int, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or len(a.indices) != len(set(a.indices))
        or any(i not in range(40) for i in a.indices)
    ):
        raise ValueError("new frozen label source/output and distinct declared courses required")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    wait_for_terminal_collection(a.producer_unit)
    if (
        digest(a.bank / "complete.json") != a.expected_bank_hash
        or digest(a.producer / "complete.json") != a.expected_producer_hash
    ):
        raise ValueError("closed original motion bank and actual student producer required")
    bd = json.loads((a.bank / "declaration.json").read_text())
    pd = json.loads((a.producer / "declaration.json").read_text())
    pc = json.loads((a.producer / "complete.json").read_text())
    if pc["seed"] != 29 or len(pc["results"]) != 80 or pd["mode"] != "evaluate":
        raise ValueError("fixed seed29 baseline whole80 producer required")
    pins = {**bd["input_file_hashes"], **pd["input_file_hashes"]}
    for root in (a.bank, a.producer):
        pins.update({str(f): digest(f) for f in root.glob("*.json")})
    for i in a.indices:
        row = pc["results"][40 + i]
        if (
            row["variant"] != "trained"
            or row["status"] != "COMPLETE"
            or row["course"] != bd["rows"][i]["course"]
            or row["native_steps"] != 3000
            or row["artifact_hash"] != digest(a.producer / f"trained-course{i:02d}/native.npz")
        ):
            raise ValueError("ordered own-policy physics recording differs")
        for f in (a.producer / f"trained-course{i:02d}").iterdir():
            if f.is_file():
                pins[str(f)] = digest(f)
    for asset in [a.world, a.motion_csv] + [
        a.model_root / n
        for n in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]:
        if pins.get(str(asset)) != digest(asset):
            raise ValueError("unchanged physical world/source/teacher network required")
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.own_motion_state_teacher_labels.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            indices=a.indices,
            producer_hash=a.expected_producer_hash,
            bank_hash=a.expected_bank_hash,
            new_student_policy_rollouts=0,
            student_actions_replayed_not_neural_policy_recomputed=True,
            teacher_uses_privileged_motion_reference=True,
            teacher_controls_physics=False,
            teacher_advice_recoverability_not_yet_proven=True,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
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
    from rosclaw_soccer.rsi.full_entry_forecast import (
        EntryCommandLaw,
        nominal_prefix20,
        run_entry_suffix,
    )
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_motion_student_rollout import validate_motion_recording
    from scripts.rsi_motion_teacher_tap import MotionTeacherTap
    from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, exact_array

    world = mujoco.MjModel.from_binary_path(str(a.world))
    source = np.loadtxt(a.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
    priors = [Path(k) for k in pins if Path(k).name == "entry-affordance.json"]
    if len(priors) != 1:
        raise ValueError("one original body entry prior required")
    stored = json.loads(priors[0].read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("prior changed")
    results = []
    for index in a.indices:
        record = bd["rows"][index]
        folder = a.output / f"course{index:02d}"
        folder.mkdir()
        with np.load(a.producer / f"trained-course{index:02d}/native.npz", allow_pickle=False) as z:
            trace = {k: z[k].copy() for k in z.files}
        validate_motion_recording(trace, nbody=world.nbody)
        original = Path(record["source_artifact"])
        parent = original.parent.parent
        for name in ("complete.json", "declaration.json"):
            if pins.get(str(parent / name)) != digest(parent / name):
                raise ValueError("original teacher control provenance differs")
        control = json.loads((parent / "complete.json").read_text())
        item = next(x for x in control["results"] if x["course"] == record["course"])
        selected = item["selected"]
        option = item["candidates"][selected["selected_index"]]
        poses, affordance, _ = contact_foot_context(
            source, prior, selected["intended_reference_foot"]
        )
        nav = G1SonicNavigation(
            a.model_root,
            record["course"]["agent_id"],
            SonicNavigationConfig(
                maximum_frames=300,
                model_variant="low_latency",
                inference_threads=1,
                record_default_planner_events=True,
            ),
        )
        body = G1TargetEpisode(
            world,
            world_hash=digest(a.world),
            kp=nav.backend.kp.copy(),
            kd=nav.backend.kd.copy(),
            course=tuple(record["course"]["course"]),
            defaults=G1SonicRunupController.default_angles.copy(),
            record_all_ball_contacts=True,
        )
        taps = {id(nav.backend): MotionTeacherTap(nav.backend)}
        labels = []

        def capture(teacher, frame, taps=taps, labels=labels, trace=trace):
            label = taps[id(teacher.backend)].label(trace["muscle_proprioception"][frame])
            labels.append(label)
            return label

        def prefix_observer(row, labels=labels, nav=nav, trace=trace):
            frame = len(labels)
            capture(nav, frame)
            exact_array(row["neural_action_isaac"][0], trace["consumed_action_isaac"][frame])

        def provider(teacher, obs, taps=taps, trace=trace):
            if id(teacher.backend) not in taps:
                taps[id(teacher.backend)] = MotionTeacherTap(teacher.backend)
            advice = teacher.prepare_shadow_action(obs)
            label = capture(teacher, obs.frame)
            exact_array(advice, label["teacher_advice_action"])
            return teacher.commit_batched_action(trace["consumed_action_isaac"][obs.frame].copy())

        try:
            prefix = nominal_prefix20(nav, body, prior, step_observer=prefix_observer)
            suffix, report = run_entry_suffix(
                nav,
                body,
                model_root=a.model_root,
                source_poses=poses,
                source_evidence_hash=hash_json(
                    json.loads((parent / "declaration.json").read_text())
                ),
                affordance=affordance,
                law=EntryCommandLaw(**selected["report"]["law"]),
                contact_reference_offset_m=tuple(option["offset_root_m"]),
                motor_provider=provider,
            )
            for key in PHYSICAL_FIELDS:
                exact_array(np.concatenate((prefix[key], suffix[key])), trace[key])
            if len(labels) != 300 or body.integrated_steps != 3000:
                raise ValueError("full own-policy native and teacher label sequence required")
        except Exception as error:
            if labels:
                np.savez_compressed(
                    folder / "partial-labels.npz",
                    **{k: np.stack([r[k] for r in labels]) for k in labels[0]},
                )
            write_once(
                folder / "failure.json",
                dict(
                    error=repr(error),
                    native_steps=body.integrated_steps,
                    labels=len(labels),
                    optimizer_steps=0,
                ),
            )
            raise
        np.savez_compressed(
            folder / "labels.npz",
            features=trace["student_features"],
            executed_student_action=trace["consumed_action_isaac"],
            **{k: np.stack([r[k] for r in labels]) for k in labels[0]},
        )
        row = dict(
            course_index=index,
            split=record["split"],
            labels=300,
            native_steps=3000,
            own_student_physics_bit_exact=True,
            own_decoder_history930_bit_exact=True,
            teacher_advice_executed=False,
            teacher_report=report,
            source_student_hash=pc["results"][40 + index]["artifact_hash"],
            artifact_hash=digest(folder / "labels.npz"),
        )
        write_once(folder / "complete.json", row)
        results.append(row)
        print("OWN_MOTION_TEACHER_LABELS_NATIVE_VERIFIED", index, len(labels), flush=True)
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="OWN_STATE_TEACHER_LABELS_NATIVE_VERIFIED",
            source_head=a.expected_head,
            results=results,
            new_native_steps=sum(r["native_steps"] for r in results),
            optimizer_steps=0,
            physics_teacher_recoverability_verified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
