"""Recorded-target GPU drift across full contact/recovery; NOT training approval.

Compare against already qualified native TRAIN64 arrays. No native re-execution,
neural inference, gradients, threshold widening or automatic qualification.
"""

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path


def state_drift(reference, observed, *, robot_qpos, ball_qpos, joints):
    """Typed state errors: never describe quaternion/joint differences as metres."""
    import numpy as np

    if (
        reference.shape != observed.shape
        or reference.ndim != 3
        or reference.shape[-1] != 43
        or not np.isfinite(reference).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("finite aligned actual qpos trajectories required")
    result = {}
    for name, index in (("root", robot_qpos), ("ball", ball_qpos)):
        result[name + "_position_max_distance_m"] = float(
            np.linalg.norm(
                reference[..., index : index + 3] - observed[..., index : index + 3], axis=-1
            ).max()
        )
        a = reference[..., index + 3 : index + 7].astype(np.float64)
        b = observed[..., index + 3 : index + 7].astype(np.float64)
        na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        if (na < 1e-12).any() or (nb < 1e-12).any():
            raise ValueError("nonzero measured quaternion norms required")
        # Normalize copies for a rotation metric, never the physical state.
        dot = ((a / na[..., None]) * (b / nb[..., None])).sum(axis=-1)
        result[name + "_rotation_max_geodesic_rad"] = float(
            (2 * np.arccos(np.clip(np.abs(dot), 0, 1))).max()
        )
    result["hinge_position_max_absolute_rad"] = float(
        np.abs(reference[..., joints] - observed[..., joints]).max()
    )
    return result


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-root", type=Path, required=True)
    p.add_argument("--native-hash", required=True)
    p.add_argument("--controls", type=int, choices=(1, 84), default=84)
    p.add_argument(
        "--feedback", action="store_true", help="own GPU state, original177 neural policy"
    )
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    receipt = a.native_root / "complete.json"
    if (
        a.output.exists()
        or digest(receipt) != a.native_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("frozen source, exact native proof and unused output required")
    proof = json.loads(receipt.read_text())
    if (
        proof["status"] != "NATIVE_CONTACT24_RECOVERY60_PROBE_NOT_LEARNER"
        or proof["strict_replays_exact"] is not True
        or proof["repeats_exact"] is not True
    ):
        raise ValueError("actual complete native contact/recovery source required")
    rawpath = a.native_root / "option0.npz"
    pins = {
        **proof["input_file_hashes"],
        str(receipt): a.native_hash,
        str(rawpath): proof["raw_hashes"][0],
    }
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    spec = importlib.util.find_spec("mujoco_warp")
    if spec is None or not spec.origin:
        raise ValueError("installed pinned simulator source required")
    for f in Path(spec.origin).parent.rglob("*.py"):
        pins[str(f)] = digest(f)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            controls=a.controls,
            worlds=64,
            substeps_per_control=10,
            maximum_GPU_world_steps=64 * a.controls * 10,
            new_native_steps=0,
            maximum_neural_rows=64 * a.controls if a.feedback else 0,
            maximum_native_forward_observation_calls=64 * a.controls if a.feedback else 0,
            optimizer_updates=0,
            source_split="TRAIN diagnostic only, NOT disjoint agreement qualification",
            fixed_recorded_targets_not_neural_feedback=not a.feedback,
            feedback_from_own_GPU_state=a.feedback,
            CPU_forward_observation_not_integration=a.feedback,
            all_GPU_training_environment=False,
            no_MDP_reset=True,
            same_original_compiled_model_and_PD=True,
            old_strict_GPU_gate_still_rejected=True,
            no_threshold_widening_or_training_approval=True,
            continues_recorded_controls_if_body_falls=not a.feedback,
            whole_batch_aborts_on_body_guard_if_feedback=a.feedback,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    gpu = None
    feedback = None
    rows = []
    neural_trace = []
    try:
        import mujoco
        import numpy as np
        import warp

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from scripts.rsi_warp_recorded_batch import RecordedWarpBatch

        def pinned(suffix):
            found = [Path(f) for f in pins if f.endswith(suffix)]
            if len(found) != 1:
                raise ValueError("one exact input required: " + suffix)
            return found[0]

        def load(path):
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        bank = load(pinned("/rsi-body-bound-precontact-curriculum-20261021/precontact-resets.npz"))
        native = load(rawpath)
        if not native["head_valid"].all() or not native["tail_valid"].all():
            raise ValueError("source must have full64x84 recorded control execution")
        ids = native["bank_index"]
        mp = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        model = mujoco.MjModel.from_binary_path(str(mp))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        cfg = pinned("/policy/velocity/v0/params/deploy.yaml")
        params = VelocityPolicyParameters.load(cfg, expected_hash=pins[str(cfg)])
        roles = np.array(
            [
                2 if (model.body(i).name or "").startswith("robot/") else -1
                for i in range(model.nbody)
            ],
            dtype=np.int32,
        )
        for foot, name in enumerate(("left", "right")):
            roles[model.body("robot/" + name + "_ankle_roll_link").id] = foot
        controls = np.concatenate((native["head_target"], native["tail_target"]))[: a.controls]
        if controls.shape != (a.controls, 64, 29) or abs(model.opt.timestep - 0.002) > 1e-12:
            raise ValueError("full recorded native HG29 targets and original2ms timestep required")
        previous = np.zeros((64, model.nu))
        previous[:, layout.ai] = np.stack(
            [target29(x, params) for x in bank["previous_action"][ids]]
        )
        gpu = RecordedWarpBatch(model, worlds=64, ball=layout.ball, roles=roles)
        gpu.restore(
            qpos=bank["qpos"][ids],
            qvel=bank["qvel"][ids],
            warmstart=bank["qacc_warmstart"][ids],
            ctrl=previous,
            time=bank["reference_frame"][ids] * 0.02,
        )
        initial = {
            k: warp.to_torch(getattr(gpu.data, k)).cpu().numpy().copy()
            for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "time")
        }
        if a.feedback:
            import torch

            from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
            from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
            from scripts.rsi_warp_foundation_feedback import FoundationFeedback

            torch.set_num_threads(1)
            plans = [
                load(
                    pinned(
                        "/rsi-train-starts-body-bound-index-check-20261021/"
                        + name.removesuffix(".npz")
                        + "-plan.npz"
                    )
                )
                for name in TRAIN_MOTION_NAMES
            ]
            onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
            actor = VelocityPriorActor.from_onnx(
                onnx, expected_hash=pins[str(onnx)], reference_features=79
            ).eval()
            state = torch.load(
                pinned("/ball-aware-update200.pt"), map_location="cpu", weights_only=True
            )
            actor.load_state_dict(
                {k: state["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()},
                strict=True,
            )
            feedback = FoundationFeedback(
                model=model,
                layout=layout,
                params=params,
                bank=bank,
                indices=ids,
                plans=plans,
                actor=actor,
            )
        for control in range(a.controls):
            if feedback is not None:
                state = {
                    k: warp.to_torch(getattr(gpu.data, k)).cpu().numpy().copy()
                    for k in ("qpos", "qvel")
                }
                neural_trace.append(feedback.next(state["qpos"], state["qvel"], control))
                controls[control] = neural_trace[-1]["target"]
            ctrl = np.zeros((64, model.nu))
            ctrl[:, layout.ai] = controls[control]
            for _ in range(10):
                rows.append(gpu.step(ctrl))
            if feedback is not None:
                q = np.stack([row["qpos"] for row in rows[-10:]])
                if (
                    q[:, :, layout.rq + 2].min() < 0.65
                    or (1 - 2 * (q[:, :, layout.rq + 4] ** 2 + q[:, :, layout.rq + 5] ** 2)).min()
                    < 0.9
                ):
                    raise ValueError("whole GPU feedback diagnostic stopped at actual body failure")
            if control % 20 == 0:
                print(
                    json.dumps(
                        dict(control=control, completed_GPU_world_steps=gpu.completed_steps)
                    ),
                    flush=True,
                )
        arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        path = a.output / "gpu-recorded.npz"
        np.savez_compressed(
            path,
            bank_index=ids,
            processed_targets=controls,
            **{"initial_" + k: v for k, v in initial.items()},
            **{"neural_" + k: np.stack([row[k] for row in neural_trace]) for k in neural_trace[0]}
            if neural_trace
            else {},
            **arrays,
        )
        qids = [*range(layout.rq, layout.rq + 7), *layout.qi, *range(layout.bq, layout.bq + 7)]
        vids = [*range(layout.rv, layout.rv + 6), *layout.vi, *range(layout.bv, layout.bv + 6)]
        truth = {}
        for key in ("qpos", "qvel", "forces"):
            x = np.concatenate([native["head_" + key], native["tail_" + key]])[: a.controls]
            truth[key] = x.transpose(0, 2, 1, 3).reshape(a.controls * 10, 64, -1)
        outcomes = []
        for world in range(64):
            paired = {}
            for mode, data in (("native_recorded", truth), ("gpu_recorded", arrays)):
                q = data["qpos"][:, world][:, qids].astype(np.float64)
                v = data["qvel"][:, world][:, vids].astype(np.float64)
                force = data["forces"][:, world].astype(np.float64)
                paired[mode] = dict(
                    postcontact=native_contact_window(
                        force=force[:, :2],
                        canonical_qpos=q,
                        canonical_qvel=v,
                        goal=bank["goal_world"][ids[world]],
                        intended_foot=int(bank["expected_foot"][ids[world]]),
                    ),
                    forbidden=bool((force[:, 2] > 1).any()),
                    stable=bool(
                        q[:, 2].min() >= 0.65
                        and (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)).min() >= 0.9
                    ),
                )
            outcomes.append(dict(world=world, bank_index=int(ids[world]), **paired))
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "GPU_FOUNDATION_FEEDBACK_DIAGNOSTIC_NOT_TRAINING_QUALIFICATION"
                    if a.feedback
                    else "GPU_RECORDED_OPTION_DRIFT_DIAGNOSTIC_NOT_TRAINING_QUALIFICATION"
                ),
                input_file_hashes=pins,
                raw_hash=digest(path),
                outcomes=outcomes,
                max_qpos_difference=float(np.abs(arrays["qpos"] - truth["qpos"]).max()),
                max_qvel_difference=float(np.abs(arrays["qvel"] - truth["qvel"]).max()),
                typed_state_drift=state_drift(
                    truth["qpos"],
                    arrays["qpos"],
                    robot_qpos=layout.rq,
                    ball_qpos=layout.bq,
                    joints=layout.qi,
                ),
                correct_foot_label_agreement=sum(
                    x["native_recorded"]["postcontact"]["correct_foot_first"]
                    == x["gpu_recorded"]["postcontact"]["correct_foot_first"]
                    for x in outcomes
                ),
                nonfoot_label_agreement=sum(
                    x["native_recorded"]["forbidden"] == x["gpu_recorded"]["forbidden"]
                    for x in outcomes
                ),
                stable_label_agreement=sum(
                    x["native_recorded"]["stable"] == x["gpu_recorded"]["stable"] for x in outcomes
                ),
                gpu_world_steps=gpu.completed_steps,
                gpu_attempted_steps=gpu.attempted_steps,
                new_native_steps=0,
                neural_rows=feedback.neural_rows if feedback else 0,
                native_forward_observation_calls=feedback.native_forward_calls if feedback else 0,
                optimizer_updates=0,
                old_strict_GPU_gate_still_rejected=True,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        if neural_trace:
            np.savez_compressed(
                a.output / "failed-neural.npz",
                **{k: np.stack([row[k] for row in neural_trace]) for k in neural_trace[0]},
            )
        if rows:
            np.savez_compressed(
                a.output / "failed-partial.npz",
                **{k: np.stack([r[k] for r in rows]) for k in rows[0]},
            )
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                gpu_world_steps=gpu.completed_steps if gpu else 0,
                gpu_attempted_steps=gpu.attempted_steps if gpu else 0,
                new_native_steps=0,
                neural_rows=feedback.neural_rows if feedback else 0,
                native_forward_observation_calls=feedback.native_forward_calls if feedback else 0,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
