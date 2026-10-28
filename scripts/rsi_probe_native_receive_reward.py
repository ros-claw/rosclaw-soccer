"""Compare kick and experimental receive credit on identical native physics.

This is a plumbing probe on an existing kick bank, not receiving training or a
task qualification. It changes no robot or ball state after episode reset.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("qualification-root", "reference-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("qualification-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if args.output.exists() or subprocess.check_output(["git", "status", "--porcelain"]):
        raise ValueError("unused output and clean pinned source required")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != args.expected_head
    ):
        raise ValueError("exact source head required")
    qualification = args.qualification_root / "complete.json"
    if digest(qualification) != args.qualification_hash:
        raise ValueError("exact source qualification receipt required")
    report = json.loads(qualification.read_text())
    if report.get("status") != "NATIVE_CONTACT24_COURSE_QUALIFIED_NOT_SKILL":
        raise ValueError("qualified kick-bank source required")
    pins = dict(report["input_file_hashes"])
    check_pins(pins)

    def pinned(suffix):
        found = [Path(k) for k in pins if k.endswith(suffix)]
        if len(found) != 1:
            raise ValueError("one pinned asset required: " + suffix)
        return found[0]

    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
    from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
    from rosclaw_soccer.rsi.native_contact_course import NativeContactCourse
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
    from scripts.rsi_native_course_inputs import load_body_references

    torch.set_num_threads(1)
    cfg = pinned("/policy/velocity/v0/params/deploy.yaml")
    onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
    checkpoint_path = pinned("/ball-aware-update200.pt")
    bank_path = pinned("/rsi-body-bound-precontact-curriculum-20261021/precontact-resets.npz")
    model_path = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
    params = VelocityPolicyParameters.load(cfg, expected_hash=pins[str(cfg)])
    with np.load(bank_path, allow_pickle=False) as data:
        bank = {k: data[k].copy() for k in data.files}
    plans = []
    for name in TRAIN_MOTION_NAMES:
        path = pinned(
            "/rsi-train-starts-body-bound-index-check-20261021/"
            + name.removesuffix(".npz")
            + "-plan.npz"
        )
        with np.load(path, allow_pickle=False) as data:
            plans.append({k: data[k].copy() for k in data.files})
    model = mujoco.MjModel.from_binary_path(str(model_path))
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    refs = load_body_references(args.reference_root, model, layout, plans)
    actor = VelocityPriorActor.from_onnx(
        onnx, expected_hash=pins[str(onnx)], reference_features=79
    ).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actor.load_state_dict(
        {k: checkpoint["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()},
        strict=True,
    )
    courses = [
        NativeContactCourse(
            model=model,
            params=params,
            bank=bank,
            plans=plans,
            body_references=refs,
            worlds=8,
            lead=10,
            threads=2,
            receive_reward=mode,
        )
        for mode in (False, True)
    ]
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_head=args.expected_head,
            qualification_hash=args.qualification_hash,
            experiment="receive-credit-physics-identity-v1",
            worlds=8,
            kick_bank_not_receive_training=True,
            controls=24,
            optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    arrays = {}
    try:
        observations = [course.reset() for course in courses]
        if not np.array_equal(*observations):
            raise ValueError("initial observations differ")
        rows = []
        while not courses[0].ended.all():
            with torch.no_grad():
                action = actor(torch.from_numpy(courses[0].observation)).numpy().copy()
            first, second = [course.step(action) for course in courses]
            for key in (
                "qpos",
                "qvel",
                "forces",
                "guards",
                "observation",
                "valid",
                "terminated",
                "truncated",
                "target",
                "tracking",
            ):
                if not np.array_equal(first[key], second[key]):
                    raise ValueError("physical identity failed: " + key)
            rows.append((first, second, action))
        if not courses[1].ended.all():
            raise ValueError("terminal status differs")
        for key in ("qpos", "qvel", "forces", "guards", "valid", "terminated", "truncated"):
            arrays[key] = np.stack([row[1][key] for row in rows])
        for key in (
            "first_clean_foot",
            "first_nonfoot",
            "sustained_capture",
            "foot_seen",
            "contaminated",
            "stable_controls",
        ):
            arrays["credit_" + key] = np.stack([row[1]["credit_" + key] for row in rows])
        arrays["kick_reward"] = np.stack([row[0]["reward"] for row in rows])
        arrays["receive_reward"] = np.stack([row[1]["reward"] for row in rows])
        arrays["action"] = np.stack([row[2] for row in rows])
        np.savez_compressed(args.output / "physical-pairs.npz", **arrays)
        check_pins(pins)
        if subprocess.check_output(["git", "status", "--porcelain"]):
            raise ValueError("source changed during run")
        write_once(
            args.output / "complete.json",
            dict(
                status="PAIRED_RECEIVE_REWARD_PHYSICS_IDENTITY_NOT_RECEIVE_SKILL",
                source_head=args.expected_head,
                qualification_hash=args.qualification_hash,
                raw_hash=digest(args.output / "physical-pairs.npz"),
                worlds=8,
                controls=len(rows),
                native_steps=[course.native_steps for course in courses],
                first_clean_foot=int(arrays["credit_first_clean_foot"].sum()),
                first_nonfoot=int(arrays["credit_first_nonfoot"].sum()),
                sustained_capture=int(arrays["credit_sustained_capture"].sum()),
                physics_identical=True,
                training=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    finally:
        for course in courses:
            course.close()


if __name__ == "__main__":
    main()
