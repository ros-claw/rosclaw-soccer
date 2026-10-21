"""Zero-influence physical check and protected-domain test of a new SIM adapter.

This is an architecture test, NOT training, skill improvement or a candidate
reuse of the rejected full-actor learner. The immutable original parent is used.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("course-root", "reference-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("course-hash", "expected-head"):
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
        raise ValueError("new frozen adapter proof and unused output required")
    f = a.course_root / "complete.json"
    if digest(f) != a.course_hash:
        raise ValueError("immutable actual native course proof required")
    report = json.loads(f.read_text())
    if report.get("status") != "NATIVE_CONTACT24_COURSE_QUALIFIED_NOT_SKILL":
        raise ValueError("complete actual native course proof required")
    pins = {**report["input_file_hashes"], str(f): a.course_hash}
    for i, h in enumerate(report["raw_hashes"]):
        pins[str(a.course_root / f"episode{i}.npz")] = h
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            architecture="frozen foundation177 + zero-init128/64/29 bounded residual",
            max_normalized_residual=0.2,
            radial_gate_m=[0.35, 0.7],
            gate_input="CURRENT observed pelvis-to-ball body-frame planar distance",
            missing_ball9_disables_adapter=True,
            no_source_outcome_or_case_gate=True,
            random_modified_head_only_function_test_not_learned_candidate=True,
            maximum_native_steps=3840,
            maximum_rollout_forward_rows=384,
            static_input_rows=1520,
            static_forward_foundation_rows=4560,
            static_forward_adapter_rows=3040,
            paired_rollouts="8sources lead10 x24controls base and zero adapter whole raw exact",
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    env = None
    steps = rows = 0
    try:
        import copy

        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_contact_adapter import NativeResidualContactActor
        from rosclaw_soccer.rsi.native_contact_course import NativeContactCourse
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
        from scripts.rsi_native_course_inputs import load_body_references

        torch.set_num_threads(1)
        torch.manual_seed(7602)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")

        def pinned(suffix):
            matches = [Path(k) for k in pins if k.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError("one immutable input required: " + suffix)
            return matches[0]

        cfg, onnx, cp = (
            pinned(s)
            for s in (
                "/policy/velocity/v0/params/deploy.yaml",
                "/policy/velocity/v0/exported/policy.onnx",
                "/ball-aware-update200.pt",
            )
        )
        mp = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        bp = pinned("/rsi-body-bound-precontact-curriculum-20261021/precontact-resets.npz")
        sp = pinned("/rsi-train-starts-body-bound-index-check-20261021/train-starts.npz")
        rp = pinned("/native-rehearsal1200.npz")
        params = VelocityPolicyParameters.load(cfg, expected_hash=pins[str(cfg)])
        with np.load(bp, allow_pickle=False) as z:
            bank = {k: z[k].copy() for k in z.files}
        with np.load(sp, allow_pickle=False) as z:
            starts = z["observation"].copy()
        with np.load(rp, allow_pickle=False) as z:
            old = np.concatenate((z["observation"], np.zeros((1200, 79), dtype=np.float32)), axis=1)
        obs = torch.from_numpy(np.concatenate((bank["observation"], starts, old))).cuda()
        base = (
            VelocityPriorActor.from_onnx(onnx, expected_hash=pins[str(onnx)], reference_features=79)
            .cuda()
            .eval()
        )
        parent = torch.load(cp, map_location="cpu", weights_only=True)
        base.load_state_dict(
            {k: parent["algorithm"]["actor_state_dict"]["mlp." + k] for k in base.state_dict()}
        )
        adapter = NativeResidualContactActor(copy.deepcopy(base)).cuda().eval()
        zeros = {k: v.clone() for k, v in adapter.adapter.state_dict().items()}
        with torch.no_grad():
            foundation = base(obs)
            initial = adapter(obs)
            if not torch.equal(initial, foundation):
                raise ValueError("zero-init adapter changes actual foundation outputs")
            # Deliberate finite weight test, not optimizer output or a skill.
            adapter.adapter[-1].weight.normal_(0, 0.01)
            adapter.adapter[-1].bias.fill_(0.01)
            modified = adapter(obs)
            gate = adapter.measured_gate(obs)
            far = gate == 0
            if (
                not torch.equal(modified[far], foundation[far])
                or not torch.any(gate > 0)
                or torch.equal(modified[~far], foundation[~far])
                or float((modified - foundation).abs().max()) > 0.200001
            ):
                raise ValueError(
                    "modified residual violates protection/bounds or has no near effect"
                )
            if not torch.all(far[256:320]) or not torch.all(far[320:]):
                raise ValueError("actual frame0 and absent-perception rehearsal must be protected")
        np.savez_compressed(
            a.output / "function-test.npz",
            observation=obs.cpu().numpy(),
            foundation=foundation.cpu().numpy(),
            initial=initial.cpu().numpy(),
            modified=modified.cpu().numpy(),
            gate=gate.cpu().numpy(),
        )
        adapter.adapter.load_state_dict(zeros)
        model = mujoco.MjModel.from_binary_path(str(mp))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        plans = []
        for name in TRAIN_MOTION_NAMES:
            with np.load(
                pinned(
                    "/rsi-train-starts-body-bound-index-check-20261021/"
                    + name.removesuffix(".npz")
                    + "-plan.npz"
                ),
                allow_pickle=False,
            ) as z:
                plans.append({k: z[k].copy() for k in z.files})
        refs = load_body_references(a.reference_root, model, layout, plans)
        arrays, hashes = {}, {}
        for label, actor in (("base", base), ("zero-adapter", adapter)):
            env = NativeContactCourse(
                model=model,
                params=params,
                bank=bank,
                plans=plans,
                body_references=refs,
                worlds=8,
                lead=10,
            )
            env.reset()
            history = []
            while not env.ended.all():
                with torch.no_grad():
                    action = actor(torch.from_numpy(env.observation).cuda()).cpu().numpy().copy()
                rows += env.worlds
                history.append(env.step(action))
            arrays[label] = {k: np.stack([r[k] for r in history]) for k in history[0]}
            path = a.output / (label + ".npz")
            np.savez_compressed(path, **arrays[label])
            hashes[label] = digest(path)
            steps += env.native_steps
            env.close()
            env = None
        if any(
            not np.array_equal(arrays["base"][k], arrays["zero-adapter"][k]) for k in arrays["base"]
        ):
            raise ValueError("zero adapter changes actual full native/NN/reward trajectories")
        if any(
            not torch.equal(v, adapter.base.state_dict()[k]) for k, v in base.state_dict().items()
        ):
            raise ValueError("frozen foundation weights changed")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_BOUNDED_CONTACT_ADAPTER_ZERO_INFLUENCE_QUALIFIED_NOT_SKILL",
                input_file_hashes=pins,
                function_test_hash=digest(a.output / "function-test.npz"),
                paired_raw_hashes=hashes,
                native_steps=steps,
                rollout_forward_rows=rows,
                static_input_rows=1520,
                static_foundation_forward_rows=4560,
                static_adapter_forward_rows=3040,
                optimizer_updates=0,
                protected_rows=int(far.sum()),
                zero_influence_all_inputs=True,
                modified_finite_adapter_far_and_missing_perception_exact=True,
                full_paired_physics_exact=True,
                frozen_foundation_exact=True,
                trained_adapter=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc),
            native_steps=steps + (env.native_steps if env is not None else 0),
            rollout_forward_rows=rows,
            optimizer_updates=0,
            hardware_authorized=False,
        )
        print(json.dumps(failure), flush=True)
        write_once(a.output / "failed.json", failure)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
