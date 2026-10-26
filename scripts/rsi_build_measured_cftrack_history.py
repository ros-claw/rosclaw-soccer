"""Build scratch CFtrack observation candidates from five real SIM boundaries."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
    p.add_argument("--layout-review", type=Path, required=True)
    p.add_argument("--layout-review-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.source / "complete.json"
    if (
        a.output.exists()
        or any(p.is_symlink() for p in (complete, a.review, a.layout_review))
        or digest(complete) != a.source_hash
        or digest(a.review) != a.review_hash
        or digest(a.layout_review) != a.layout_review_hash
        or any(n in sys.modules for n in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused output, exact pinned sources and isolated import required")
    source = json.loads(complete.read_text())
    review = json.loads(a.review.read_text())
    layout_review = json.loads(a.layout_review.read_text())
    if (
        source["status"] != "WARM_LEFT_SETTLING_MEASURED_NOT_KICK"
        or source["controls"] != 157
        or review["status"] != "WARM_LEFT_SETTLING_NN_AND_NATIVE_EXACT_NOT_KICK"
        or review["source_hash"] != a.source_hash
        or layout_review["status"] != "CFTRACK_STAGED_LATERAL_NN_NATIVE_AND_BOUNDARY_EXACT"
        or layout_review["history_proprioception_audited"] is not True
        or layout_review["new_neural_calls"] != 314
    ):
        raise ValueError("closed physical source and qualified history layout required")
    trace = a.source / source["artifact"]["path"]
    if (
        trace.is_symlink()
        or trace.parent != a.source
        or digest(trace) != source["artifact"]["hash"]
    ):
        raise ValueError("exact ordinary source trace required")
    pins = dict(source["input_file_hashes"])
    pins.update(
        {
            str(complete): a.source_hash,
            str(a.review): a.review_hash,
            str(a.layout_review): a.layout_review_hash,
            str(trace): source["artifact"]["hash"],
        }
    )
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    external_files = [Path(x) for x in pins if x.endswith("/policy/omnicontact/OmniContact.py")]
    if len(scenes) != 1 or len(external_files) != 1:
        raise ValueError("one pinned native scene and external scratch policy required")
    external = external_files[0].parents[2]
    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.cftrack_action_history import describe_previous_target_encoding
    from rosclaw_soccer.rsi.cftrack_history_observation import (
        assemble_current_ball_observation,
        flatten_five_frame_history,
    )
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if compiled_model_hash(model) != (
        "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact native body required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.OmniContact import OmniContact

    policy = OmniContact(
        StateAndCmd(29),
        PolicyOutput(29),
        onnx_path=str(external / "policy/omnicontact/model/kick_50k.onnx"),
    )
    if tuple(policy.kinematics.joint_names) != tuple(G1_DDS_JOINT_NAMES):
        raise ValueError("source FK must use canonical HG29 order")
    policy.task, policy.reference_source, policy.active_object_name = "kickball", "CFgen", "ball"
    policy.box_dims = policy.ball_dims = np.full(3, 0.11, np.float32)
    policy.bbox_scale = policy.box_dims * 2
    policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)
    with np.load(trace, allow_pickle=False) as f:
        raw = {k: f[k].copy() for k in f.files}
    if (
        raw["qpos"].shape != (1570, model.nq)
        or raw["qvel"].shape != (1570, model.nv)
        or raw["joint_targets"].shape != (157, 29)
    ):
        raise ValueError("complete native and executed target history required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.source_hash,
            review_hash=a.review_hash,
            layout_review_hash=a.layout_review_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            entry_frames=[57, 82],
            five_consecutive_measured_boundaries=True,
            previous_targets_are_not_original_policy_actions=True,
            cached_ball_from_penultimate_substep=True,
            no_history_installed=True,
            network_calls=0,
            native_steps=0,
            hardware_authorized=False,
        ),
    )
    results, arrays = [], {}
    for entry in (57, 82):
        data = mujoco.MjData(model)
        view = SimpleNamespace(
            world=SimpleNamespace(data=data, control_steps=0), policy=policy, layout=lo
        )
        observations, roundtrip, action_magnitudes = [], [], []
        for frame in range(entry - 4, entry + 1):
            data.qpos[:] = raw["qpos"][frame * 10 - 2]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            data.qpos[:] = raw["qpos"][frame * 10 - 1]
            data.qvel[:] = raw["qvel"][frame * 10 - 1]
            view.world.control_steps = frame
            SharedOmniContactPlayer._observe(view)
            policy._get_fk_info()
            encoded = describe_previous_target_encoding(
                raw["joint_targets"][frame - 1],
                policy.default_angles_lab,
                policy.action_scale_lab,
                policy.mj2lab,
                policy.lab2mj,
            )
            policy.action = encoded["implied_action_lab"].copy()
            observations.append(assemble_current_ball_observation(policy))
            roundtrip.append(encoded["maximum_target_roundtrip_error_rad"])
            action_magnitudes.append(encoded["maximum_absolute_implied_action"])
        history = np.stack(observations)
        arrays[f"entry{entry}_observations"] = history
        arrays[f"entry{entry}_flattened"] = flatten_five_frame_history(history[:4], history[4])
        results.append(
            dict(
                entry_control=entry,
                source_frames=list(range(entry - 4, entry + 1)),
                max_target_roundtrip_error_rad=max(roundtrip),
                max_implied_action_abs=max(action_magnitudes),
                scratch_data_time=float(data.time),
            )
        )
        if data.time != 0 or policy.counter_step != 0 or np.any(policy.obs_history_buffer):
            raise ValueError("scratch history construction must not infer, integrate or install")
    artifact = a.output / "history.npz"
    np.savez_compressed(artifact, **arrays)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="MEASURED_CFTRACK_HISTORY_CANDIDATES_NOT_INSTALLED",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            review_hash=a.review_hash,
            layout_review_hash=a.layout_review_hash,
            input_file_hashes=pins,
            records=results,
            artifact=dict(path=artifact.name, hash=digest(artifact)),
            previous_target_actions_are_implied=True,
            network_calls=0,
            native_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
