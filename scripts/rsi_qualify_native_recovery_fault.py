"""Declared SIM fault injections: failed head and actual crouch failure in tail.

No robot/ball state overrides, external forces, policy updates or gain changes.
Other six worlds continue normal feedback. Exact native recorded-control replay.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for key in ("proof-root", "reference-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("proof-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        )
    ):
        raise ValueError("unused output and clean frozen source required")
    pf = a.proof_root / "complete.json"
    if digest(pf) != a.proof_hash:
        raise ValueError("exact continuous recovery proof required")
    proof = json.loads(pf.read_text())
    if proof["status"] != "NATIVE_CONTACT24_RECOVERY60_PROBE_NOT_LEARNER" or any(
        proof[k] is not True
        for k in ("prefix_exact", "handoff_state_exact", "repeats_exact", "strict_replays_exact")
    ):
        raise ValueError("actual continuous baseline required")
    pins = {**proof["input_file_hashes"], str(pf): a.proof_hash}
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    old_sources = {
        h
        for f, h in proof["input_file_hashes"].items()
        if f.endswith("/src/rosclaw_soccer/rsi/native_recovery_tail.py")
    }
    if old_sources != {digest(Path("src/rosclaw_soccer/rsi/native_recovery_tail.py"))}:
        raise ValueError("unchanged actual recovery implementation required")
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            worlds=8,
            lead=10,
            head_reference_fault=dict(
                source=0, body="left_ankle_roll_link", z_offset_m=1.0, before_allocation=True
            ),
            tail_action_fault=dict(
                world=1,
                start_control=0,
                hip_pitch_rad=-1.2,
                knee_rad=2.4,
                ankle_pitch_rad=-0.8,
                both_legs=True,
                other_targets="current immutable foundation",
                only_while_active=True,
            ),
            maximum_live_steps=6720,
            maximum_replay_steps=6720,
            maximum_nn_rows=672,
            optimizer_updates=0,
            no_external_force_or_state_overwrite=True,
            fixture_only_not_skill=True,
            hardware_authorized=False,
        ),
    )
    env = None
    replay = neural = 0
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_contact_course import NativeContactCourse
        from rosclaw_soccer.rsi.native_course_tracking import TRACKING_BODIES
        from rosclaw_soccer.rsi.native_recovery_tail import NativeRecoveryTail
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
        from scripts.rsi_native_course_inputs import load_body_references

        torch.set_num_threads(1)

        def pinned(end):
            found = [Path(f) for f in pins if f.endswith(end)]
            if len(found) != 1:
                raise ValueError("unambiguous pinned prerequisite required: " + end)
            return found[0]

        cfg = pinned("/policy/velocity/v0/params/deploy.yaml")
        onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
        cp = pinned("/ball-aware-update200.pt")
        mp = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        bp = pinned("/rsi-body-bound-precontact-curriculum-20261021/precontact-resets.npz")
        model = mujoco.MjModel.from_binary_path(str(mp))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        params = VelocityPolicyParameters.load(cfg, expected_hash=pins[str(cfg)])
        with np.load(bp, allow_pickle=False) as z:
            bank = {k: z[k].copy() for k in z.files}
        plans = []
        for name in TRAIN_MOTION_NAMES:
            path = pinned(
                "/rsi-train-starts-body-bound-index-check-20261021/"
                + name.removesuffix(".npz")
                + "-plan.npz"
            )
            with np.load(path, allow_pickle=False) as z:
                plans.append({k: z[k].copy() for k in z.files})
        refs = load_body_references(a.reference_root, model, layout, plans)
        refs[0]["position"][:, TRACKING_BODIES.index("left_ankle_roll_link"), 2] += 1.0
        fault_ids = np.array([0, 3, 4, 6, 9, 10])
        fault_targets = np.array([-1.2, 2.4, -0.8, -1.2, 2.4, -0.8], dtype=np.float32)
        bounds = model.jnt_range[layout.joints][fault_ids]
        if np.any(fault_targets < bounds[:, 0]) or np.any(fault_targets > bounds[:, 1]):
            raise ValueError("declared fixture targets outside actual joints; no repair")
        env = NativeContactCourse(
            model=model,
            params=params,
            bank=bank,
            plans=plans,
            body_references=refs,
            worlds=8,
            lead=10,
            threads=8,
        )
        actor = VelocityPriorActor.from_onnx(
            onnx, expected_hash=pins[str(onnx)], reference_features=79
        ).eval()
        state = torch.load(cp, map_location="cpu", weights_only=True)
        actor.load_state_dict(
            {k: state["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()},
            strict=True,
        )

        def act(obs, ids):
            nonlocal neural
            out = np.zeros((8, 29), dtype=np.float32)
            if len(ids):
                with torch.no_grad():
                    out[ids] = actor(torch.from_numpy(obs[ids])).numpy()
                neural += len(ids)
            return out

        env.reset()
        failed = np.zeros(8, dtype=bool)
        head = []
        while not env.ended.all():
            ids = np.flatnonzero(~env.ended)
            info = env.step(act(env.observation, ids))
            failed |= info["terminated"]
            head.append(info)
        if not failed[0] or failed[1:].any():
            raise ValueError("declared source0 fixture did not isolate expected head failure")
        tail = NativeRecoveryTail(env, contact_failed=failed)
        rows = []
        for _ in range(60):
            ids = tail.active_ids
            if not len(ids):
                break
            snapshot = {
                int(i): (
                    d.qpos.copy(),
                    d.qvel.copy(),
                    d.time,
                    tail.observation[i].copy(),
                    tail.phase[i].copy(),
                )
                for i, d in enumerate(env.batch.data)
                if tail.ended[i]
            }
            action = act(tail.observation, ids)
            if 1 in ids:
                action[1, fault_ids] = (
                    fault_targets - np.array(params.action_offset, dtype=np.float32)[fault_ids]
                ) / np.array(params.action_scale, dtype=np.float32)[fault_ids]
            info = tail.step(action)
            for i, (q, v, t, obs, phase) in snapshot.items():
                d = env.batch.data[i]
                if (
                    not np.array_equal(d.qpos, q)
                    or not np.array_equal(d.qvel, v)
                    or d.time != t
                    or not np.array_equal(tail.observation[i], obs)
                    or tail.phase[i] != phase
                ):
                    raise ValueError("failed world state/clock/observation advanced")
            rows.append(info)
        raw = {"head_" + k: np.array([r[k] for r in head]) for k in head[0]}
        raw.update({"tail_" + k: np.array([r[k] for r in rows]) for k in rows[0]})
        raw["bank_index"] = env.index.copy()
        path = a.output / "fault.npz"
        np.savez_compressed(path, **raw)
        if (
            not tail.failed[1]
            or not 0 < tail.elapsed[1] < 60
            or tail.failed[2:].any()
            or not np.all(tail.elapsed[2:] == 60)
        ):
            raise ValueError(
                "actual tail failure fixture not isolated/observed; no invented failure"
            )
        for world, index in enumerate(env.index):
            d = mujoco.MjData(model)
            d.qpos[:], d.qvel[:] = bank["qpos"][index], bank["qvel"][index]
            d.time = float(bank["reference_frame"][index] * 0.02)
            d.ctrl[layout.ai] = target29(bank["previous_action"][index], params)
            mujoco.mj_forward(model, d)
            d.qacc_warmstart[:] = bank["qacc_warmstart"][index]
            for prefix in ("head_", "tail_"):
                for step in range(len(raw[prefix + "valid"])):
                    if not raw[prefix + "valid"][step, world]:
                        continue
                    d.ctrl[layout.ai] = raw[prefix + "target"][step, world]
                    for sub in range(10):
                        mujoco.mj_step(model, d)
                        replay += 1
                        if (
                            not np.array_equal(d.qpos, raw[prefix + "qpos"][step, world, sub])
                            or not np.array_equal(d.qvel, raw[prefix + "qvel"][step, world, sub])
                            or not np.array_equal(
                                env._forces(d), raw[prefix + "forces"][step, world, sub]
                            )
                        ):
                            raise ValueError("fault strict native replay differs")
                    mujoco.mj_forward(model, d)
        if neural * 10 != env.native_steps:
            raise ValueError("inactive worlds received inference or physical work")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_RECOVERY_FAILED_WORLD_PATHS_QUALIFIED_NOT_SKILL",
                input_file_hashes=pins,
                raw_hash=digest(path),
                head_failed_worlds=np.flatnonzero(failed).tolist(),
                tail_failed_worlds=np.flatnonzero(tail.failed & ~failed).tolist(),
                tail_controls=tail.elapsed.tolist(),
                live_native_steps=env.native_steps,
                replay_native_steps=replay,
                policy_rows=neural,
                failed_state_clock_observation_frozen=True,
                inference_only_active=True,
                strict_replay_exact=True,
                optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                live_native_steps=env.native_steps if env else 0,
                replay_native_steps=replay,
                policy_rows=neural,
                hardware_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
