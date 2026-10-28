"""Execute a bounded incoming-ball probe with MotionDecode FK reference.

The source clips and frames may be correlated; no receiving labels are
inferred from them and this is not a training or promotion result.
"""

import argparse
import json
import subprocess
from pathlib import Path


def require_unpromoted_imitation(receipt):
    if (
        receipt.get("status") != "INCOMING_PPO_PILOT_REQUIRES_INDEPENDENT_EXAM"
        or not receipt.get("imitation_hash")
        or any(
            receipt.get(key) is not False
            for key in ("hardware_authorized", "promotion_authorized", "independent_fresh_exam")
        )
    ):
        raise ValueError("unpromoted actual imitation training receipt required")


def imitation_reference_bound(declaration, reference_path, reference_hash):
    """Allow the training reference or an exact path pinned at training time."""
    inputs = declaration.get("input_file_hashes")
    return bool(
        declaration.get("motion_reference_hash") == reference_hash
        or (isinstance(inputs, dict) and inputs.get(str(reference_path)) == reference_hash)
    )


def experimental_actor_state(checkpoint, kind):
    """Extract only the requested actor; never silently fall back to a baseline."""
    if kind == "imitation":
        if set(checkpoint) != {"actor_state_dict"}:
            raise ValueError("exact tensor-only imitation checkpoint required")
        return checkpoint["actor_state_dict"]
    if kind != "candidate" or not isinstance(checkpoint.get("policy_state_dict"), dict):
        raise ValueError("complete PPO candidate policy required")
    actor = {
        key.removeprefix("actor."): value
        for key, value in checkpoint["policy_state_dict"].items()
        if key.startswith("actor.")
    }
    if not actor:
        raise ValueError("candidate checkpoint must contain actor weights")
    return actor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("qualification-root", "reference-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("qualification-hash", "reference-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--launch-frame", choices=("world", "heading"), default="world")
    p.add_argument("--lateral-study", action="store_true")
    p.add_argument("--lateral-speed", type=float, choices=(1.25, 1.75, 2.25, 3.0), default=1.25)
    p.add_argument("--receive-window", type=int, choices=(40, 60), default=40)
    p.add_argument("--intercept-teacher", action="store_true")
    p.add_argument("--robust-teacher", action="store_true")
    p.add_argument("--reserve-teacher", action="store_true")
    p.add_argument("--parent-relative-teacher", action="store_true")
    p.add_argument("--freeze-teacher-ankles", action="store_true")
    p.add_argument("--isolate-teacher-rejections", action="store_true")
    p.add_argument("--target-rate-filter", action="store_true")
    p.add_argument("--ankle-rate-filter", action="store_true")
    p.add_argument("--strict-native-capture", action="store_true")
    p.add_argument("--experimental-imitation-receipt", type=Path)
    p.add_argument("--experimental-imitation-hash")
    p.add_argument("--experimental-imitation-declaration-hash")
    p.add_argument(
        "--experimental-actor-kind", choices=("imitation", "candidate"), default="imitation"
    )
    a = p.parse_args()
    if a.lateral_study and a.launch_frame != "heading":
        raise ValueError("lateral study requires an explicit body-heading launch")
    if a.strict_native_capture and (not a.lateral_study or a.receive_window != 60):
        raise ValueError("strict native capture requires the 60-control lateral study")
    if not a.lateral_study and a.lateral_speed != 1.25:
        raise ValueError("explicit lateral speed requires lateral study")
    if a.intercept_teacher and (not a.lateral_study or a.receive_window != 60):
        raise ValueError("teacher requires the declared 60-control lateral study")
    if (
        a.robust_teacher or a.reserve_teacher or a.parent_relative_teacher
    ) and not a.intercept_teacher:
        raise ValueError("robust authority bound requires explicit teacher")
    if sum((a.robust_teacher, a.reserve_teacher, a.parent_relative_teacher)) > 1:
        raise ValueError("distinct teacher intervention modes cannot be combined")
    if a.isolate_teacher_rejections and not (a.reserve_teacher or a.parent_relative_teacher):
        raise ValueError("isolated rejection requires an explicit teacher study")
    if a.parent_relative_teacher and not a.isolate_teacher_rejections:
        raise ValueError("parent-relative study requires isolated infeasible worlds")
    if a.freeze_teacher_ankles and not a.parent_relative_teacher:
        raise ValueError("ankle-freeze study requires the parent-relative teacher")
    if (
        len(
            {
                bool(a.experimental_imitation_receipt),
                bool(a.experimental_imitation_hash),
                bool(a.experimental_imitation_declaration_hash),
            }
        )
        != 1
    ):
        raise ValueError("experimental imitation requires exact completed receipt and hash")
    if a.experimental_imitation_receipt and (
        a.intercept_teacher or a.target_rate_filter or a.ankle_rate_filter
    ):
        raise ValueError("standalone own-feedback imitation evaluation required")
    if a.experimental_actor_kind != "imitation" and not a.experimental_imitation_receipt:
        raise ValueError("candidate evaluation requires an exact completed training receipt")
    if a.target_rate_filter and a.ankle_rate_filter:
        raise ValueError("choose exactly one rate filter scope")
    if (a.target_rate_filter or a.ankle_rate_filter) and (
        a.intercept_teacher or not a.lateral_study or a.receive_window != 60
    ):
        raise ValueError("target rate filter requires standalone 60-control lateral study")
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
    imitation_path = None
    if a.experimental_imitation_receipt:
        if (
            a.experimental_imitation_receipt.is_symlink()
            or digest(a.experimental_imitation_receipt) != a.experimental_imitation_hash
        ):
            raise ValueError("exact completed imitation receipt required")
        imitation = json.loads(a.experimental_imitation_receipt.read_text())
        require_unpromoted_imitation(imitation)
        actor_file = "candidate.pt" if a.experimental_actor_kind == "candidate" else "imitation.pt"
        actor_hash_key = (
            "checkpoint_hash" if a.experimental_actor_kind == "candidate" else "imitation_hash"
        )
        imitation_path = a.experimental_imitation_receipt.parent / actor_file
        imitation_declaration_path = a.experimental_imitation_receipt.parent / "declaration.json"
        if (
            imitation_declaration_path.is_symlink()
            or digest(imitation_declaration_path) != a.experimental_imitation_declaration_hash
        ):
            raise ValueError("exact imitation training declaration required")
        imitation_declaration = json.loads(imitation_declaration_path.read_text())
        if (
            imitation_declaration.get("source_head") != imitation.get("source_head")
            or not imitation_reference_bound(imitation_declaration, ref_path, a.reference_hash)
            or not isinstance(imitation_declaration.get("source_receipts"), list)
            or len(imitation_declaration["source_receipts"]) != 2
            or imitation_declaration["source_receipts"][0] != a.qualification_hash
        ):
            raise ValueError("matching body, training reference and declared source required")
        if imitation_path.is_symlink() or digest(imitation_path) != imitation[actor_hash_key]:
            raise ValueError("exact confined imitation actor required")
        pins[str(a.experimental_imitation_receipt)] = a.experimental_imitation_hash
        pins[str(imitation_path)] = imitation[actor_hash_key]
        pins[str(imitation_declaration_path)] = a.experimental_imitation_declaration_hash
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
    from rosclaw_soccer.rsi.native_target_rate_filter import (
        ANKLE_JOINT_INDICES,
        limit_ankle_target_rate,
        limit_target_rate,
    )
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
            a.lateral_speed if a.lateral_study else (1.25, 2.0)[(i // 8) % 2],
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
                or any(courses[int(j)].speed_mps != a.lateral_speed for j in group)
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
    if imitation_path is not None:
        learned = torch.load(imitation_path, map_location="cpu", weights_only=True)
        actor.load_state_dict(
            experimental_actor_state(learned, a.experimental_actor_kind), strict=True
        )
        if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
            raise ValueError("finite experimental imitation actor required")
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
            fixed_speed_mps=a.lateral_speed if a.lateral_study else None,
            robot_initial_state_matched_by_source=a.lateral_study,
            controls=a.receive_window,
            no_mid_episode_resets=True,
            training=False,
            experimental_imitation_hash=a.experimental_imitation_hash,
            experimental_imitation_declaration_hash=a.experimental_imitation_declaration_hash,
            experimental_actor_kind=a.experimental_actor_kind if imitation_path else None,
            experimental_actor=imitation_path is not None,
            intercept_teacher=a.intercept_teacher,
            teacher_unlearned=a.intercept_teacher,
            isolate_teacher_rejections=a.isolate_teacher_rejections,
            target_rate_filter=a.target_rate_filter,
            max_total_target_step_rad=0.08 if a.target_rate_filter else None,
            ankle_rate_filter=a.ankle_rate_filter,
            strict_native_capture=a.strict_native_capture,
            ankle_joint_indices=list(ANKLE_JOINT_INDICES) if a.ankle_rate_filter else None,
            max_ankle_target_step_rad=0.08 if a.ankle_rate_filter else None,
            teacher_configuration=dict(
                early_intercept=True,
                contact_lead=True,
                axis_local_fallback=True,
                all_knee_colliders=True,
                reject_actual_slew_conflict=True,
                max_delta_rad=0.012 if a.robust_teacher else 0.16,
                joint_margin_rad=0.2 if a.reserve_teacher else 0.0,
                parent_relative_envelope=a.parent_relative_teacher,
                freeze_ankle_correction=a.freeze_teacher_ankles,
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
        strict_native_capture=a.strict_native_capture,
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
                    joint_margin_rad=0.2 if a.reserve_teacher else 0.0,
                    parent_relative_envelope=a.parent_relative_teacher,
                    freeze_ankle_correction=a.freeze_teacher_ankles,
                )
                for _ in range(env.worlds)
            ]
            if a.intercept_teacher
            else None
        )
        rows = []
        rejected = []
        while not env.ended.all():
            with torch.no_grad():
                action = actor(torch.from_numpy(env.observation)).numpy().copy()
            nominal_action = action.copy()
            teacher_delta = np.zeros((env.worlds, 29), dtype=np.float64)
            teacher_active = np.zeros(env.worlds, dtype=bool)
            filter_delta = np.zeros((env.worlds, 29), dtype=np.float64)
            rejected_mask = np.zeros(env.worlds, dtype=bool)
            if a.target_rate_filter or a.ankle_rate_filter:
                filter_function = (
                    limit_target_rate if a.target_rate_filter else limit_ankle_target_rate
                )
                for i in np.flatnonzero(~env.ended):
                    action[i], filter_delta[i] = filter_function(
                        action=action[i], previous_action=env.last_action[i], params=params
                    )
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
                        if a.parent_relative_teacher:
                            limits = model.jnt_range[layout.joints]
                            actual = target29(action[i], params)
                            parent_violation = np.maximum(limits[:, 0] - nominal, 0) + np.maximum(
                                nominal - limits[:, 1], 0
                            )
                            actual_violation = np.maximum(limits[:, 0] - actual, 0) + np.maximum(
                                actual - limits[:, 1], 0
                            )
                            if np.any(actual_violation > parent_violation + 1e-6):
                                raise ValueError("actual target worsens parent range violation")
                    except ValueError as error:
                        if not a.isolate_teacher_rejections:
                            raise ValueError(
                                f"teacher slew conflict at world {i}, control {env.elapsed[i]}"
                            ) from error
                        rejected_mask[i] = True
                        rejected.append(
                            dict(
                                world=int(i),
                                control=int(env.elapsed[i]),
                                reason=str(error),
                                max_residual_jump=float(
                                    np.max(np.abs(teacher_delta[i] - previous_delta))
                                ),
                                qpos=env.batch.data[i].qpos.copy(),
                                qvel=env.batch.data[i].qvel.copy(),
                                rejected_action=action[i].copy(),
                                previous_delta=previous_delta.copy(),
                                rejected_delta=teacher_delta[i].copy(),
                            )
                        )
                    teacher_active[i] = proposal["active"]
            if rejected_mask.any():
                env.reject_worlds(rejected_mask)
            if env.ended.all():
                break
            result = env.step(action)
            rows.append(
                {
                    **result,
                    "action": action,
                    "nominal_action": nominal_action,
                    "teacher_delta": teacher_delta,
                    "teacher_active": teacher_active,
                    "filter_delta": filter_delta,
                    "teacher_rejected": rejected_mask,
                }
            )
        raw = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        np.savez_compressed(a.output / "physical.npz", bank_index=env.index, **raw)
        if rejected:
            np.savez_compressed(
                a.output / "teacher-rejections.npz",
                **{key: np.asarray([row[key] for row in rejected]) for key in rejected[0]},
            )
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
                fixed_speed_mps=a.lateral_speed if a.lateral_study else None,
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
                experimental_imitation_hash=a.experimental_imitation_hash,
                experimental_imitation_declaration_hash=a.experimental_imitation_declaration_hash,
                experimental_actor_kind=a.experimental_actor_kind if imitation_path else None,
                experimental_actor=imitation_path is not None,
                intercept_teacher=a.intercept_teacher,
                robust_teacher=a.robust_teacher,
                reserve_teacher=a.reserve_teacher,
                parent_relative_teacher=a.parent_relative_teacher,
                freeze_teacher_ankles=a.freeze_teacher_ankles,
                target_rate_filter=a.target_rate_filter,
                ankle_rate_filter=a.ankle_rate_filter,
                strict_native_capture=a.strict_native_capture,
                ankle_joint_indices=list(ANKLE_JOINT_INDICES) if a.ankle_rate_filter else None,
                target_filter_changed_controls=int(
                    (np.any(raw["filter_delta"] != 0, axis=-1) & raw["valid"]).sum()
                ),
                teacher_active_controls=int((raw["teacher_active"] & raw["valid"]).sum()),
                teacher_changed_controls=int(
                    (np.any(raw["teacher_delta"] != 0, axis=-1) & raw["valid"]).sum()
                ),
                teacher_guard_rejections=len(rejected),
                teacher_rejection_hash=digest(a.output / "teacher-rejections.npz")
                if rejected
                else None,
                independent_replay=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
