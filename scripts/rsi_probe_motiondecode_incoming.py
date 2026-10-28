"""Execute a bounded incoming-ball probe with MotionDecode FK reference.

The source clips and frames may be correlated; no receiving labels are
inferred from them and this is not a training or promotion result.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("qualification-root", "reference-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("qualification-hash", "reference-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--launch-frame", choices=("world", "heading"), default="world")
    p.add_argument("--lateral-study", action="store_true")
    p.add_argument("--receive-window", type=int, choices=(40, 60), default=40)
    p.add_argument("--intercept-teacher", action="store_true")
    p.add_argument("--robust-teacher", action="store_true")
    a = p.parse_args()
    if a.lateral_study and a.launch_frame != "heading":
        raise ValueError("lateral study requires an explicit body-heading launch")
    if a.intercept_teacher and (not a.lateral_study or a.receive_window != 60):
        raise ValueError("teacher requires the declared 60-control lateral study")
    if a.robust_teacher and not a.intercept_teacher:
        raise ValueError("robust authority bound requires explicit teacher")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if a.output.exists() or subprocess.check_output(["git", "status", "--porcelain"]):
        raise ValueError("unused output and clean source required")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != a.expected_head:
        raise ValueError("exact frozen source required")
    qual_path, ref_path = a.qualification_root / "complete.json", a.reference_root / "complete.json"
    if digest(qual_path) != a.qualification_hash or digest(ref_path) != a.reference_hash:
        raise ValueError("exact input receipts required")
    qual, reference = json.loads(qual_path.read_text()), json.loads(ref_path.read_text())
    multi = reference.get("status") == "MOTIONDECODE_TRAIN8_KINEMATIC_FK_NOT_TRACKED_SKILL"
    if qual.get("status") != "NATIVE_CONTACT24_COURSE_QUALIFIED_NOT_SKILL" or (
        not multi
        and reference.get("status")
        != "MOTIONDECODE_KINEMATIC_NATIVE_FK_REFERENCE_NOT_TRACKED_SKILL"
    ):
        raise ValueError("qualified actor assets and explicit nonphysical reference required")
    pins = {
        **qual["input_file_hashes"],
        str(qual_path): a.qualification_hash,
        str(ref_path): a.reference_hash,
    }
    if multi:
        if len(reference["records"]) != 8:
            raise ValueError("all eight declared motion references required")
        for i, record in enumerate(reference["records"]):
            pins[str(a.reference_root / f"reference-{i:02d}.npz")] = record["reference_hash"]
    else:
        pins[str(a.reference_root / "reference.npz")] = reference["raw_hash"]
    check_pins(pins)

    def pinned(suffix):
        found = [Path(k) for k in pins if k.endswith(suffix)]
        if len(found) != 1:
            raise ValueError("one exact asset required: " + suffix)
        return found[0]

    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
    from rosclaw_soccer.rsi.native_contact_course import (
        NativeContactCourse,
        balanced_course_indices,
    )
    from rosclaw_soccer.rsi.native_predictive_foot_shin_teacher import (
        NativePredictiveFootShinTeacher,
    )
    from rosclaw_soccer.rsi.native_receive_bank import build_incoming_bank
    from rosclaw_soccer.rsi.native_teacher_action import (
        require_teacher_slew,
        teacher_target_to_action,
    )
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
    from rosclaw_soccer.training.role_receiving_courses import ROSTER, ReceivingCourse

    torch.set_num_threads(1)
    cfg = pinned("/policy/velocity/v0/params/deploy.yaml")
    onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
    checkpoint = pinned("/ball-aware-update200.pt")
    model_path = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
    if pins[str(model_path)] != reference["model_hash"]:
        raise ValueError("reference must use exact same model")
    params = VelocityPolicyParameters.load(cfg, expected_hash=pins[str(cfg)])
    model = mujoco.MjModel.from_binary_path(str(model_path))
    plans, refs = [], []
    for i in range(8):
        path = a.reference_root / (f"reference-{i:02d}.npz" if multi else "reference.npz")
        with np.load(path, allow_pickle=False) as z:
            plans.append(
                {k.removeprefix("plan_"): z[k].copy() for k in z.files if k.startswith("plan_")}
            )
            refs.append(
                {k.removeprefix("body_"): z[k].copy() for k in z.files if k.startswith("body_")}
            )
    world = np.arange(256, dtype=np.int64)
    source = world % 8
    frame = 20 + source * 25 + (0 if a.lateral_study else (world // 8) * 5)
    if any(
        int(f) + a.receive_window >= len(plans[int(s)]["joint"])
        for f, s in zip(frame, source, strict=True)
    ):
        raise ValueError("all declared complete reference windows required")
    bank = dict(
        world=world,
        source_id=source,
        reference_frame=frame,
        lead_controls=np.array([10, 20, 40, 80], dtype=np.int64)[(world // 8) % 4],
        qpos=np.stack(
            [plans[int(s)]["native_qpos"][int(f)] for f, s in zip(frame, source, strict=True)]
        ),
        qvel=np.stack(
            [plans[int(s)]["native_qvel"][int(f)] for f, s in zip(frame, source, strict=True)]
        ),
        qacc_warmstart=np.zeros((256, model.nv)),
        previous_action=np.zeros((256, 29), dtype=np.float32),
        phase_before=np.zeros(256, dtype=np.float32),
        observation=np.zeros((256, 177), dtype=np.float32),
        origin=np.zeros((256, 3), dtype=np.float32),
        goal_world=np.tile(np.array([[2, 0, 0.11]], dtype=np.float32), (256, 1)),
        expected_foot=(world % 2).astype(np.int8),
    )
    courses = tuple(
        ReceivingCourse(
            ROSTER[i % 8],
            930000 + i,
            1.25 if a.lateral_study else (1.25, 2.0)[(i // 8) % 2],
            (-0.24, -0.08, 0.08, 0.24)[(i // 32) % 4]
            if a.lateral_study
            else (-0.08, 0.08)[(i // 16) % 2],
        )
        for i in range(256)
    )
    bank, binding = build_incoming_bank(
        model=model,
        params=params,
        bank=bank,
        plans=plans,
        courses=courses,
        launch_frame=a.launch_frame,
    )
    if a.lateral_study:
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        ids = balanced_course_indices(bank, worlds=32, episode=0, lead=10)
        for i in range(8):
            group = ids[i::8]
            robot_q = np.delete(bank["qpos"][group], np.arange(layout.bq, layout.bq + 7), axis=1)
            robot_v = np.delete(bank["qvel"][group], np.arange(layout.bv, layout.bv + 6), axis=1)
            if (
                not np.all(robot_q == robot_q[0])
                or not np.all(robot_v == robot_v[0])
                or not np.all(bank["reference_frame"][group] == bank["reference_frame"][group[0]])
                or [courses[int(j)].lateral_m for j in group] != [-0.24, -0.08, 0.08, 0.24]
                or any(courses[int(j)].speed_mps != 1.25 for j in group)
            ):
                raise ValueError("all four lateral arms must share exact robot start and speed")
    actor = VelocityPriorActor.from_onnx(
        onnx, expected_hash=pins[str(onnx)], reference_features=79
    ).eval()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor.load_state_dict(
        {k: ckpt["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()},
        strict=True,
    )
    a.output.mkdir()
    np.savez_compressed(a.output / "incoming-bank.npz", **bank)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            reference_hash=a.reference_hash,
            qualification_hash=a.qualification_hash,
            course_binding=binding,
            launch_frame=a.launch_frame,
            source_clip_count=8 if multi else 1,
            correlated_reference_frames=True,
            source_fps_assumption=reference["source_fps_assumption"],
            scene="single G1, one free ball",
            worlds=32 if a.lateral_study else 8,
            lateral_study=a.lateral_study,
            lateral_m=[-0.24, -0.08, 0.08, 0.24] if a.lateral_study else None,
            fixed_speed_mps=1.25 if a.lateral_study else None,
            robot_initial_state_matched_by_source=a.lateral_study,
            controls=a.receive_window,
            no_mid_episode_resets=True,
            training=False,
            intercept_teacher=a.intercept_teacher,
            teacher_unlearned=a.intercept_teacher,
            teacher_configuration=dict(
                early_intercept=True,
                contact_lead=True,
                axis_local_fallback=True,
                all_knee_colliders=True,
                reject_actual_slew_conflict=True,
                max_delta_rad=0.012 if a.robust_teacher else 0.16,
            )
            if a.intercept_teacher
            else None,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = NativeContactCourse(
        model=model,
        params=params,
        bank=bank,
        plans=plans,
        body_references=refs,
        worlds=32 if a.lateral_study else 8,
        lead=10,
        threads=2,
        receive_reward=True,
        receive_window=a.receive_window,
    )
    try:
        env.reset()
        teachers = (
            [
                NativePredictiveFootShinTeacher(
                    axis_local_fallback=True,
                    early_intercept=True,
                    contact_lead=True,
                    all_knee_colliders=True,
                    max_delta_rad=0.012 if a.robust_teacher else 0.16,
                )
                for _ in range(env.worlds)
            ]
            if a.intercept_teacher
            else None
        )
        rows = []
        while not env.ended.all():
            with torch.no_grad():
                action = actor(torch.from_numpy(env.observation)).numpy().copy()
            nominal_action = action.copy()
            teacher_delta = np.zeros((env.worlds, 29), dtype=np.float64)
            teacher_active = np.zeros(env.worlds, dtype=bool)
            if teachers is not None:
                assert env.batch is not None
                for i in np.flatnonzero(~env.ended):
                    nominal = target29(action[i], params)
                    previous_delta = teachers[i].previous_delta.copy()
                    try:
                        proposal = teachers[i].propose(
                            model=model,
                            snapshot=env.batch.data[i],
                            layout=env.layout,
                            nominal=nominal,
                            enabled=True,
                            frame=int(env.elapsed[i]),
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"teacher failed at world {i}, control {env.elapsed[i]}, "
                            f"previous max {np.max(np.abs(teachers[i].previous_delta))}"
                        ) from error
                    if np.any(proposal["applied_delta"]):
                        action[i], teacher_delta[i] = teacher_target_to_action(
                            action[i],
                            proposal["target"],
                            proposal["applied_delta"],
                            params,
                            max_delta_rad=0.012 if a.robust_teacher else 0.16,
                        )
                        teachers[i].previous_delta = teacher_delta[i].copy()
                    try:
                        require_teacher_slew(previous_delta, teacher_delta[i])
                    except ValueError as error:
                        raise ValueError(
                            f"teacher slew conflict at world {i}, control {env.elapsed[i]}"
                        ) from error
                    teacher_active[i] = proposal["active"]
            result = env.step(action)
            rows.append(
                {
                    **result,
                    "action": action,
                    "nominal_action": nominal_action,
                    "teacher_delta": teacher_delta,
                    "teacher_active": teacher_active,
                }
            )
        raw = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        np.savez_compressed(a.output / "physical.npz", bank_index=env.index, **raw)
        check_pins(pins)
        if subprocess.check_output(["git", "status", "--porcelain"]):
            raise ValueError("source changed during physical probe")
        write_once(
            a.output / "complete.json",
            dict(
                status="MOTIONDECODE_INCOMING_SINGLE_ACTOR_PROBE_NOT_SKILL",
                source_head=a.expected_head,
                reference_hash=a.reference_hash,
                course_binding=binding,
                launch_frame=a.launch_frame,
                lateral_study=a.lateral_study,
                receive_window=a.receive_window,
                worlds=32 if a.lateral_study else 8,
                reference_clip_count=8 if multi else 1,
                bank_hash=digest(a.output / "incoming-bank.npz"),
                raw_hash=digest(a.output / "physical.npz"),
                controls=len(rows),
                native_steps=env.native_steps,
                valid_transitions=int(raw["valid"].sum()),
                clean_foot=int(raw["credit_first_clean_foot"].sum()),
                nonfoot=int(raw["credit_first_nonfoot"].sum()),
                capture=int(raw["credit_sustained_capture"].sum()),
                failed=int(raw["terminated"].sum()),
                training=False,
                intercept_teacher=a.intercept_teacher,
                robust_teacher=a.robust_teacher,
                teacher_active_controls=int(raw["teacher_active"].sum()),
                teacher_changed_controls=int(np.any(raw["teacher_delta"] != 0, axis=-1).sum()),
                independent_replay=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
