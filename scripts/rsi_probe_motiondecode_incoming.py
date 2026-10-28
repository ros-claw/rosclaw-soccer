"""Execute a bounded incoming-ball probe with MotionDecode FK reference.

One source clip supplies correlated frames; no receiving labels are inferred
from that source and this is not a training or promotion result.
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
    a = p.parse_args()
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
    if (
        qual.get("status") != "NATIVE_CONTACT24_COURSE_QUALIFIED_NOT_SKILL"
        or reference.get("status") != "MOTIONDECODE_KINEMATIC_NATIVE_FK_REFERENCE_NOT_TRACKED_SKILL"
    ):
        raise ValueError("qualified actor assets and explicit nonphysical reference required")
    pins = {
        **qual["input_file_hashes"],
        str(qual_path): a.qualification_hash,
        str(ref_path): a.reference_hash,
        str(a.reference_root / "reference.npz"): reference["raw_hash"],
    }
    check_pins(pins)

    def pinned(suffix):
        found = [Path(k) for k in pins if k.endswith(suffix)]
        if len(found) != 1:
            raise ValueError("one exact asset required: " + suffix)
        return found[0]

    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
    from rosclaw_soccer.rsi.native_contact_course import NativeContactCourse
    from rosclaw_soccer.rsi.native_receive_bank import build_incoming_bank
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
    with np.load(a.reference_root / "reference.npz", allow_pickle=False) as z:
        plan = {k.removeprefix("plan_"): z[k].copy() for k in z.files if k.startswith("plan_")}
        body = {k.removeprefix("body_"): z[k].copy() for k in z.files if k.startswith("body_")}
    plans, refs = [plan] * 8, [body] * 8
    world = np.arange(256, dtype=np.int64)
    source = world % 8
    frame = 20 + source * 25 + (world // 8) * 5
    if int(frame.max()) + 40 >= len(plan["joint"]):
        raise ValueError("all declared forty-step references required")
    bank = dict(
        world=world,
        source_id=source,
        reference_frame=frame,
        lead_controls=np.array([10, 20, 40, 80], dtype=np.int64)[(world // 8) % 4],
        qpos=plan["native_qpos"][frame].copy(),
        qvel=plan["native_qvel"][frame].copy(),
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
            ROSTER[i % 8], 930000 + i, (1.25, 2.0)[(i // 8) % 2], (-0.08, 0.08)[(i // 16) % 2]
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
            source_clip_count=1,
            correlated_reference_frames=True,
            source_fps_assumption=reference["source_fps_assumption"],
            scene="single G1, one free ball",
            worlds=8,
            controls=40,
            no_mid_episode_resets=True,
            training=False,
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
        worlds=8,
        lead=10,
        threads=2,
        receive_reward=True,
        receive_window=40,
    )
    try:
        env.reset()
        rows = []
        while not env.ended.all():
            with torch.no_grad():
                action = actor(torch.from_numpy(env.observation)).numpy().copy()
            result = env.step(action)
            rows.append({**result, "action": action})
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
                reference_clip_count=1,
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
                independent_replay=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
