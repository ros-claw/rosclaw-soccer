"""All256 native episode/reward/reset qualification, plus reference-fault fixture."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("feedback-root", "reference-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("feedback-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and frozen qualifier required")
    path = a.feedback_root / "complete.json"
    if digest(path) != a.feedback_hash:
        raise ValueError("immutable full feedback receipt required")
    proof = json.loads(path.read_text())
    if proof.get("status") != "NATIVE_CONTINUOUS_FEEDBACK_REPLAY_EXACT_NOT_LEARNER" or proof.get(
        "max_errors"
    ) != dict(qpos=0.0, qvel=0.0, forces=0.0):
        raise ValueError("full continuous neural/native strict replay required")
    pins = {
        **proof["input_file_hashes"],
        str(path): a.feedback_hash,
        str(a.feedback_root / "feedback.npz"): proof["raw_hash"],
    }
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    for f in (a.reference_root / "src/assets/soccer/motions/shooter").glob("*.npz"):
        pins[str(f)] = digest(f)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            protocol="native-contact24-v1",
            worlds=256,
            controls=24,
            independent_normal_episodes=2,
            maximum_native_steps=123030,
            maximum_gpu_nn_rows=12304,
            optimizer_updates=0,
            actor_abi=177,
            critic_abi=177,
            legacy_critic304_not_loaded=True,
            unchanged_body_motor_ball_and_physical_colliders=True,
            retained_source_guards=dict(
                fell_over_degrees=70, anchor_z_m=0.25, gravity_z=0.8, end_effector_z_m=0.25
            ),
            tracking_oracle_atol=1e-5,
            guards_exact=True,
            reward_reconstruction_atol=1e-12,
            new_failure_penalty=-400.0,
            dt=0.02,
            new_failure_penalty_exceeds_entire24_control_max_positive_return=True,
            reference_corruption_fixture=dict(
                worlds=8,
                controls=2,
                source=0,
                left_ankle_z_offset_m=1.0,
                applied_before_allocation=True,
                fixture_only_not_skill=True,
            ),
            no_early_active_body_state_overwrite=True,
            no_optimizer_or_promotion=True,
            hardware_authorized=False,
        ),
    )
    course = fixture = None
    nn_rows = 0
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.native_contact_course import NativeContactCourse
        from rosclaw_soccer.rsi.native_course_tracking import TRACKING_BODIES
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
        from scripts.rsi_native_course_inputs import load_body_references
        from scripts.rsi_native_tracking_oracle import source_tracking_oracle

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")

        def pinned(suffix):
            matches = [Path(k) for k in pins if k.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError("one exact pinned input required: " + suffix)
            return matches[0]

        cfg = pinned("/policy/velocity/v0/params/deploy.yaml")
        onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
        cp = pinned("/ball-aware-update200.pt")
        bp = pinned("/rsi-body-bound-precontact-curriculum-20261021/precontact-resets.npz")
        mp = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        params = VelocityPolicyParameters.load(cfg, expected_hash=pins[str(cfg)])
        with np.load(bp, allow_pickle=False) as z:
            bank = {k: z[k].copy() for k in z.files}
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES

        plans = []
        for name in TRAIN_MOTION_NAMES:
            f = pinned(
                "/rsi-train-starts-body-bound-index-check-20261021/"
                + name.removesuffix(".npz")
                + "-plan.npz"
            )
            with np.load(f, allow_pickle=False) as z:
                plans.append({k: z[k].copy() for k in z.files})
        model = mujoco.MjModel.from_binary_path(str(mp))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        refs = load_body_references(a.reference_root, model, layout, plans)
        course = NativeContactCourse(
            model=model,
            params=params,
            bank=bank,
            plans=plans,
            body_references=refs,
            worlds=256,
            lead=None,
        )
        actor = (
            VelocityPriorActor.from_onnx(onnx, expected_hash=pins[str(onnx)], reference_features=79)
            .cuda()
            .eval()
        )
        checkpoint = torch.load(cp, map_location="cpu", weights_only=True)
        actor.load_state_dict(
            {
                k: checkpoint["algorithm"]["actor_state_dict"]["mlp." + k]
                for k in actor.state_dict()
            },
            strict=True,
        )

        def act(env):
            nonlocal nn_rows
            with torch.no_grad():
                action = actor(torch.from_numpy(env.observation).cuda()).cpu().numpy().copy()
            nn_rows += env.worlds
            return action

        rows = []
        raw_hashes = []
        max_tracking = max_reward_error = 0.0
        first_data = None
        first_arrays = None
        for episode in range(2):
            course.reset()
            if first_data is not None and any(
                new is old for new, old in zip(course.batch.data, first_data, strict=True)
            ):
                raise ValueError("normal episode reset reused old native data")
            if (
                course.credit.seen.any()
                or course.credit.forbidden.any()
                or course.credit.remaining.any()
                or course.credit.last_step is not None
                or course.credit.cached is not None
            ):
                raise ValueError("credit state leaked across actual reset")
            try:
                course.reset()
            except ValueError:
                pass
            else:
                raise ValueError("active reset accepted")
            episode_rows = []
            while not course.ended.all():
                result = course.step(act(course))
                oracle, guards = source_tracking_oracle(course, result["frame_before"])
                valid = result["valid"]
                err = float(abs(result["tracking"][valid] - oracle[valid]).max())
                max_tracking = max(max_tracking, err)
                if err > 1e-5 or not np.array_equal(result["guards"][valid], guards[valid]):
                    raise ValueError("native tracking/guards differ from original source formulas")
                credit = {
                    k: result["credit_" + k]
                    for k in ("contact", "wrong_contact", "alignment", "speed", "forbidden_contact")
                }
                reconstructed = 0.02 * (
                    result["tracking"].sum(1)
                    + result["proximity"]
                    - 0.1 * result["action_rate"]
                    - 10 * result["limits"]
                    + 50 * credit["contact"]
                    - 10 * credit["wrong_contact"]
                    + 30 * credit["alignment"]
                    + 10 * credit["speed"]
                    - 300 * credit["forbidden_contact"]
                    - 400 * result["terminated"]
                )
                reconstructed[~valid] = 0
                max_reward_error = max(
                    max_reward_error, float(abs(reconstructed - result["reward"]).max())
                )
                if max_reward_error > 1e-12:
                    raise ValueError("actual course reward does not match declared arithmetic")
                episode_rows.append(result)
                print(
                    json.dumps(
                        dict(
                            episode=episode,
                            control=course.control,
                            ended=int(course.ended.sum()),
                            native_steps=course.native_steps,
                            gpu_nn_rows=nn_rows,
                        )
                    ),
                    flush=True,
                )
            raw = {k: np.stack([r[k] for r in episode_rows], axis=1) for k in episode_rows[0]}
            path = a.output / f"episode{episode}.npz"
            np.savez_compressed(path, bank_index=course.index, **raw)
            raw_hashes.append(digest(path))
            if first_arrays is not None and (
                set(raw) != set(first_arrays)
                or any(not np.array_equal(raw[k], first_arrays[k]) for k in raw)
            ):
                raise ValueError(
                    "normal repeated native/NN course trajectories or credit not exact"
                )
            first_arrays = raw
            first_data = course.batch.data
            rows.append(
                dict(
                    episode=episode,
                    controls=len(episode_rows),
                    actual_valid_samples=int(raw["valid"].sum()),
                    terminations=int(raw["terminated"].sum()),
                    timeouts=int(raw["truncated"].sum()),
                    reward_sum=float(raw["reward"].sum()),
                    positive_contact_worlds=int((raw["credit_contact"] > 0).any(1).sum()),
                    forbidden_contact_worlds=int(
                        (raw["credit_forbidden_contact"] > 0).any(1).sum()
                    ),
                )
            )
        if not course.credit.seen.any() or not course.credit.forbidden.any():
            raise ValueError("normal reset must exercise nontrivial contact and forbidden history")
        # Fixed reference fault, NEVER a new football success or dataset example.
        fixture_refs = [{k: v.copy() for k, v in ref.items()} for ref in refs]
        fixture_refs[0]["position"][:, TRACKING_BODIES.index("left_ankle_roll_link"), 2] += 1.0
        fixture = NativeContactCourse(
            model=model,
            params=params,
            bank=bank,
            plans=plans,
            body_references=fixture_refs,
            worlds=8,
            lead=10,
        )
        fixture.reset()
        r0 = fixture.step(act(fixture))
        if not r0["terminated"][0] or not r0["guards"][0, 3] or fixture.ended.all():
            raise ValueError(
                "reference fault must actually terminate source0 while other worlds live"
            )
        stopped = (
            fixture.batch.data[0].qpos.copy(),
            fixture.batch.data[0].qvel.copy(),
            fixture.batch.data[0].time,
        )
        r1 = fixture.step(act(fixture))
        if (
            r1["valid"][0]
            or r1["reward"][0] != 0
            or any(r1["credit_" + k][0] != 0 for k in credit)
            or not np.array_equal(fixture.batch.data[0].qpos, stopped[0])
            or not np.array_equal(fixture.batch.data[0].qvel, stopped[1])
            or fixture.batch.data[0].time != stopped[2]
        ):
            raise ValueError("terminated world moved or earned new training credit")
        np.savez_compressed(
            a.output / "fault-fixture.npz",
            **{"first_" + k: v for k, v in r0.items()},
            **{"second_" + k: v for k, v in r1.items()},
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_CONTACT24_COURSE_QUALIFIED_NOT_SKILL",
                input_file_hashes=pins,
                episodes=rows,
                raw_hashes=raw_hashes,
                fixture_hash=digest(a.output / "fault-fixture.npz"),
                normal_native_steps=course.native_steps,
                fault_fixture_native_steps=fixture.native_steps,
                gpu_nn_rows=nn_rows,
                optimizer_updates=0,
                max_tracking_error=max_tracking,
                max_reward_error=max_reward_error,
                independent_repeat_exact=True,
                active_reset_rejected=True,
                credit_history_cleared=True,
                failed_world_frozen=True,
                actor_abi=177,
                critic_abi=177,
                protocol=course.protocol,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc),
            native_steps=(course.native_steps if course else 0)
            + (fixture.native_steps if fixture else 0),
            gpu_nn_rows=nn_rows,
            optimizer_updates=0,
            hardware_authorized=False,
        )
        print(json.dumps(failure), flush=True)
        write_once(a.output / "failed.json", failure)
        raise
    finally:
        if course is not None:
            course.close()
        if fixture is not None:
            fixture.close()


if __name__ == "__main__":
    main()
