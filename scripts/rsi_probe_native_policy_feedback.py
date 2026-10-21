"""SIM continuous native/GPU feedback and independent recorded-control replay.

Fixed all256 starts x24 controls. No optimization, edited poses, selective cases,
or policy promotion. Physical CPU contact forces feed actual identity-bound credit.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("boundary-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("boundary-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and frozen probe required")
    f = a.boundary_root / "complete.json"
    if digest(f) != a.boundary_hash:
        raise ValueError("immutable prior boundary proof required")
    proof = json.loads(f.read_text())
    if proof.get("status") != "NATIVE_GPU_POLICY_ONE_CONTROL_QUALIFIED_NOT_LEARNER":
        raise ValueError("all256 actual neural/native first-control proof required")
    pins = {**proof["input_file_hashes"], str(f): a.boundary_hash}
    pins[str(a.boundary_root / "native-gpu-policy-control.npz")] = proof["raw_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            worlds=256,
            controls=24,
            substeps=10,
            maximum_feedback_native_steps=61440,
            maximum_replay_native_steps=61440,
            maximum_gpu_nn_rows=6144,
            optimizer_updates=0,
            input_abi=177,
            actuator_codec="native PD position29",
            actual_contact_threshold_n=10.0,
            forbidden_body_threshold_n=1.0,
            physical_replay_required="all substep q/v/feet/nonfoot bitexact",
            episode_reset_implemented=False,
            training_env_qualified=False,
            no_active_state_overwrites=True,
            no_threshold_widening=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    batch = None
    replay_steps = nn_rows = 0
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
        from rosclaw_soccer.rsi.identity_kick_credit import IdentityKickCredit
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_physics_batch import NativePhysicsBatch
        from rosclaw_soccer.rsi.native_reference_observation import native_reference_observation177
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")

        def pinned(suffix):
            paths = [Path(k) for k in pins if k.endswith(suffix)]
            if len(paths) != 1:
                raise ValueError(f"one immutable input required: {suffix}")
            return paths[0]

        bp = pinned("/rsi-body-bound-precontact-curriculum-20261021/precontact-resets.npz")
        mp = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        cp = pinned("/ball-aware-update200.pt")
        cfg = pinned("/policy/velocity/v0/params/deploy.yaml")
        onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
        params = VelocityPolicyParameters.load(cfg, expected_hash=pins[str(cfg)])
        with np.load(bp, allow_pickle=False) as z:
            bank = {k: z[k].copy() for k in z.files}
        plans = []
        for name in TRAIN_MOTION_NAMES:
            f = pinned(
                "/rsi-train-starts-body-bound-index-check-20261021/"
                + name.removesuffix(".npz")
                + "-plan.npz"
            )
            with np.load(f, allow_pickle=False) as z:
                plans.append({k: z[k].copy() for k in z.files})
        if any(
            int(frame) + 24 > len(plans[int(s)]["joint"])
            for frame, s in zip(bank["reference_frame"], bank["source_id"], strict=True)
        ):
            raise ValueError("all256 must have entire declared reference window")
        model = mujoco.MjModel.from_binary_path(str(mp))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        torso = model.body("robot/torso_link").id
        feet = [model.body("robot/" + side + "_ankle_roll_link").id for side in ("left", "right")]

        def forces(data):
            ff = np.zeros(2)
            nonfoot = 0.0
            for j in range(data.ncon):
                c = data.contact[j]
                x, y = (int(model.geom_bodyid[g]) for g in c.geom)
                other = y if x == layout.ball else x if y == layout.ball else -1
                if other >= 0 and (model.body(other).name or "").startswith("robot/"):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, j, wrench)
                    mag = float(np.linalg.norm(wrench[:3]))
                    if other in feet:
                        ff[feet.index(other)] = max(ff[feet.index(other)], mag)
                    else:
                        nonfoot = max(nonfoot, mag)
            return np.r_[ff, nonfoot]

        old = np.zeros((256, model.nu))
        old[:, layout.ai] = np.stack([target29(x, params) for x in bank["previous_action"]])
        batch = NativePhysicsBatch(model, 256, threads=8)
        batch.restore(
            qpos=bank["qpos"],
            qvel=bank["qvel"],
            previous_ctrl=old,
            warmstart=bank["qacc_warmstart"],
            time=bank["reference_frame"] * 0.02,
        )
        actor = (
            VelocityPriorActor.from_onnx(onnx, expected_hash=pins[str(onnx)], reference_features=79)
            .cuda()
            .eval()
        )
        state = torch.load(cp, map_location="cpu", weights_only=True)
        actor.load_state_dict(
            {k: state["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()},
            strict=True,
        )
        phase = bank["phase_before"].astype(np.float32)
        last_action = bank["previous_action"].copy()
        credit = IdentityKickCredit(256, "cpu")
        rows = []
        for control in range(24):
            observations, next_phase = [], []
            for i, d in enumerate(batch.data):
                if control:
                    mujoco.mj_forward(model, d)
                obs, phase_i = native_reference_observation177(
                    data=d,
                    layout=layout,
                    params=params,
                    plan=plans[int(bank["source_id"][i])],
                    frame=int(bank["reference_frame"][i]) + control,
                    origin=bank["origin"][i],
                    goal=bank["goal_world"][i],
                    last_action=last_action[i],
                    phase=phase[i],
                    torso=torso,
                )
                observations.append(obs)
                next_phase.append(phase_i)
            obs = np.array(observations)
            if control == 0 and not np.array_equal(obs, bank["observation"]):
                raise ValueError("first actual input differs from qualified measured state")
            with torch.no_grad():
                action = actor(torch.from_numpy(obs).cuda()).cpu().numpy().copy()
            nn_rows += 256
            target = np.stack([target29(x, params) for x in action])
            ctrl = np.zeros((256, model.nu))
            ctrl[:, layout.ai] = target
            qs, vs, fs = [], [], []
            for _ in range(10):
                result = batch.advance(ctrl, substeps=1)
                qs.append(result["qpos"][:, 0])
                vs.append(result["qvel"][:, 0])
                fs.append(np.stack([forces(d) for d in batch.data]))
            qs, vs, fs = (np.stack(x, axis=1) for x in (qs, vs, fs))
            scores = credit.step(
                step=control,
                forces=torch.from_numpy(fs[:, :, :2]),
                expected=torch.from_numpy(bank["expected_foot"]),
                ball_velocity=torch.from_numpy(vs[:, -1, layout.bv : layout.bv + 3]),
                goal_delta=torch.from_numpy(
                    bank["goal_world"] - qs[:, -1, layout.bq : layout.bq + 3]
                ),
                forbidden_forces=torch.from_numpy(fs[:, :, 2]),
            )
            rows.append(
                dict(
                    observation=obs,
                    action=action,
                    target=target,
                    phase_before=phase.copy(),
                    qpos=qs,
                    qvel=vs,
                    forces=fs,
                    **{"credit_" + k: v.numpy().copy() for k, v in scores.items()},
                )
            )
            phase = np.array(next_phase, dtype=np.float32)
            last_action = action
            print(
                json.dumps(
                    dict(
                        control=control + 1,
                        native_steps=batch.native_steps,
                        gpu_nn_rows=nn_rows,
                        first_contact_seen=int(credit.seen.sum()),
                        forbidden_worlds=int(credit.forbidden.sum()),
                    )
                ),
                flush=True,
            )
        live = {k: np.stack([row[k] for row in rows], axis=1) for k in rows[0]}
        path = a.output / "feedback.npz"
        np.savez_compressed(path, **live)
        # Deliberately plain independent MjData/mj_step, not batch.advance.
        max_errors = dict(qpos=0.0, qvel=0.0, forces=0.0)
        for i in range(256):
            d = mujoco.MjData(model)
            d.qpos[:] = bank["qpos"][i]
            d.qvel[:] = bank["qvel"][i]
            d.ctrl[:] = old[i]
            d.time = int(bank["reference_frame"][i]) * 0.02
            d.qacc_warmstart[:] = bank["qacc_warmstart"][i]
            mujoco.mj_forward(model, d)
            d.qacc_warmstart[:] = bank["qacc_warmstart"][i]
            for control in range(24):
                if control:
                    mujoco.mj_forward(model, d)
                d.ctrl[layout.ai] = live["target"][i, control]
                for sub in range(10):
                    mujoco.mj_step(model, d)
                    replay_steps += 1
                    for k, val in (("qpos", d.qpos), ("qvel", d.qvel), ("forces", forces(d))):
                        err = float(abs(val - live[k][i, control, sub]).max())
                        max_errors[k] = max(max_errors[k], err)
            if i % 32 == 31:
                print(
                    json.dumps(
                        dict(
                            replayed_worlds=i + 1, replay_steps=replay_steps, max_errors=max_errors
                        )
                    ),
                    flush=True,
                )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_CONTINUOUS_FEEDBACK_REPLAY_EXACT_NOT_LEARNER"
                if all(v == 0 for v in max_errors.values())
                else "NATIVE_CONTINUOUS_FEEDBACK_REPLAY_REJECTED",
                input_file_hashes=pins,
                raw_hash=digest(path),
                max_errors=max_errors,
                feedback_native_steps=batch.native_steps,
                replay_native_steps=replay_steps,
                gpu_nn_rows=nn_rows,
                optimizer_updates=0,
                contact_credit_worlds=int((live["credit_contact"] > 0).any(axis=1).sum()),
                wrong_contact_credit_worlds=int(
                    (live["credit_wrong_contact"] > 0).any(axis=1).sum()
                ),
                forbidden_contact_worlds=int(
                    (live["credit_forbidden_contact"] > 0).any(axis=1).sum()
                ),
                full_training_env_qualified=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc),
            feedback_native_steps=batch.native_steps if batch is not None else 0,
            replay_native_steps=replay_steps,
            gpu_nn_rows=nn_rows,
            optimizer_updates=0,
            hardware_authorized=False,
        )
        print(json.dumps(failure), flush=True)
        write_once(a.output / "failed.json", failure)
        raise
    finally:
        if batch is not None:
            batch.close()


if __name__ == "__main__":
    main()
