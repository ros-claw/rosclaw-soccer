"""Check measured native177 -> actual GPU actor -> native vector PD integration.

This is a one-control hybrid boundary test, not a qualified training environment
or proof of improved kicking. Compare ALL256 measured starts with their original
independent neural/physics trajectories, never a hand-selected successful subset.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("vector-proof-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("vector-proof-hash", "expected-head"):
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
        raise ValueError("unused output and frozen source required")
    f = a.vector_proof_root / "complete.json"
    if digest(f) != a.vector_proof_hash:
        raise ValueError("immutable vector proof required")
    proof = json.loads(f.read_text())
    if (
        proof.get("status") != "NATIVE_VECTOR_PHYSICS_BITEXACT_NOT_LEARNER"
        or len(proof.get("results", [])) != 2
        or any(r.get("passed") is not True for r in proof["results"])
    ):
        raise ValueError("full256 two-thread-count native vector proof required")
    pins = {**proof["input_file_hashes"], str(f): a.vector_proof_hash}
    for r in proof["results"]:
        pins[str(a.vector_proof_root / f"threads{r['threads']}.npz")] = r["raw_hash"]
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
            maximum_native_steps=2560,
            maximum_gpu_nn_rows=256,
            optimizer_updates=0,
            observation_comparison="bitexact original measured177",
            target_atol_rad=2e-5,
            qpos_atol_mixed=1e-3,
            qvel_atol_mixed=0.02,
            physics="native CPU MuJoCo unmodified",
            network="actual frozen GPU float32 actor177",
            native_solver_seeds_and_action_history_restored=True,
            no_threshold_widening=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    batch = None
    nn_rows = 0
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_physics_batch import NativePhysicsBatch
        from rosclaw_soccer.rsi.native_reference_observation import native_reference_observation177
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required for actual neural execution")

        def pinned(suffix):
            matches = [Path(k) for k in pins if k.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"one immutable input required: {suffix}")
            return matches[0]

        bp = pinned("/rsi-body-bound-precontact-curriculum-20261021/precontact-resets.npz")
        mp = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        config = pinned("/policy/velocity/v0/params/deploy.yaml")
        onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
        checkpoint = pinned("/ball-aware-update200.pt")
        params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
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
        # Original full scalar neural executions are independent source truth.
        original = {}
        for w in np.unique(bank["world"]):
            path = pinned(f"/rsi-body-bound-contact-episodes-20261021/world{int(w)}-repeat0.npz")
            with np.load(path, allow_pickle=False) as z:
                original[int(w)] = {
                    k: z[k].copy()
                    for k in (
                        "observation",
                        "target_rad",
                        "canonical_qpos_after",
                        "canonical_qvel_after",
                    )
                }
        model = mujoco.MjModel.from_binary_path(str(mp))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        torso = model.body("robot/torso_link").id
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
        obs, oracle_obs, oracle_target, oracle_q, oracle_v = [], [], [], [], []
        for i, data in enumerate(batch.data):
            frame = int(bank["reference_frame"][i])
            current, _ = native_reference_observation177(
                data=data,
                layout=layout,
                params=params,
                plan=plans[int(bank["source_id"][i])],
                frame=frame,
                origin=bank["origin"][i],
                goal=bank["goal_world"][i],
                last_action=bank["previous_action"][i],
                phase=np.float32(bank["phase_before"][i]),
                torso=torso,
            )
            source = original[int(bank["world"][i])]
            obs.append(current)
            oracle_obs.append(source["observation"][frame])
            oracle_target.append(source["target_rad"][frame])
            oracle_q.append(source["canonical_qpos_after"][frame])
            oracle_v.append(source["canonical_qvel_after"][frame])
        obs = np.array(obs)
        if not np.array_equal(obs, oracle_obs) or not np.array_equal(obs, bank["observation"]):
            np.savez_compressed(
                a.output / "failed-observations.npz",
                actual=obs,
                oracle=oracle_obs,
                bank=bank["observation"],
            )
            raise ValueError("native177 must exactly reconstruct every original observed state")
        actor = (
            VelocityPriorActor.from_onnx(onnx, expected_hash=pins[str(onnx)], reference_features=79)
            .cuda()
            .eval()
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        actor.load_state_dict(
            {k: state["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()},
            strict=True,
        )
        with torch.no_grad():
            action = actor(torch.from_numpy(obs).cuda()).cpu().numpy().copy()
        nn_rows += 256
        target = np.stack([target29(x, params) for x in action])
        ctrl = np.zeros((256, model.nu))
        ctrl[:, layout.ai] = target
        actual = batch.advance(ctrl, substeps=10)
        qi = np.r_[layout.rq : layout.rq + 7, layout.qi, layout.bq : layout.bq + 7]
        vi = np.r_[layout.rv : layout.rv + 6, layout.vi, layout.bv : layout.bv + 6]
        q = actual["qpos"][:, :, qi]
        v = actual["qvel"][:, :, vi]
        errors = dict(
            observation=float(abs(obs - oracle_obs).max()),
            target=float(abs(target - oracle_target).max()),
            qpos=float(abs(q - oracle_q).max()),
            qvel=float(abs(v - oracle_v).max()),
        )
        passed = errors["target"] <= 2e-5 and errors["qpos"] <= 1e-3 and errors["qvel"] <= 0.02
        raw = a.output / "native-gpu-policy-control.npz"
        np.savez_compressed(
            raw,
            observation=obs,
            action=action,
            target=target,
            canonical_qpos=q,
            canonical_qvel=v,
            oracle_target=oracle_target,
            oracle_qpos=oracle_q,
            oracle_qvel=oracle_v,
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_GPU_POLICY_ONE_CONTROL_QUALIFIED_NOT_LEARNER"
                if passed
                else "NATIVE_GPU_POLICY_ONE_CONTROL_REJECTED",
                input_file_hashes=pins,
                raw_hash=digest(raw),
                errors=errors,
                native_steps=batch.native_steps,
                gpu_nn_rows=nn_rows,
                optimizer_updates=0,
                full_training_env_qualified=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc),
            native_steps=batch.native_steps if batch is not None else 0,
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
