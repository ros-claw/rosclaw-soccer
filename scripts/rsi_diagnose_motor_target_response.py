"""Public TRAIN command/response audit, not a simulator rollout or regrading.

Distinguishes joint TARGET discontinuities from actual measured joint motion.
The +/-0.2 residual authority lower bound concerns targets only, never proves
that physical smoothness or task success is impossible.
"""

import argparse
import json
from pathlib import Path


def summarize(target, reference, measured, velocity, names, *, dt=0.02):
    import numpy as np

    if (
        target.ndim != 2
        or target.shape[1] != len(names)
        or len(target) < 2
        or len(set(names)) != len(names)
        or dt != 0.02
        or any(
            x.shape != target.shape or not np.isfinite(x).all()
            for x in (target, reference, measured, velocity)
        )
    ):
        raise ValueError("same finite actual20ms named-joint command/response arrays required")

    def peak(difference):
        frame, joint = np.unravel_index(int(np.argmax(abs(difference))), difference.shape)
        return dict(
            control=int(frame) + 1,
            joint=names[joint],
            signed_change_rad=float(difference[frame, joint]),
            absolute_change_rad=float(abs(difference[frame, joint])),
        )

    cmd, ref, motion = [np.diff(x, axis=0) for x in (target, reference, measured)]
    event = peak(cmd)
    t, j = event["control"], names.index(event["joint"])
    return dict(
        peak_target_change=event,
        peak_reference_target_change=peak(ref),
        peak_measured_control_end_change=peak(motion),
        measured_same_joint_change_at_peak_target_rad=float(motion[t - 1, j]),
        measured_same_joint_velocity_at_peak_target_rad_s=float(velocity[t, j]),
        maximum_measured_control_end_velocity_rad_s=float(abs(velocity).max()),
        maximum_correction_from_reference_rad=float(abs(target - reference).max()),
        unavoidable_target_jump_floor_under_pointwise_0p2rad_bound=float(
            max(0.0, float(abs(ref).max()) - 0.4)
        ),
        target_step_is_not_actual_motion=True,
        physical_smoothness_impossibility_proven=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("source-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--index", type=int, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    rp = a.root / "complete.json"
    if (
        a.output.exists()
        or rp.is_symlink()
        or digest(rp) != a.source_hash
        or (a.root / "failed.json").exists()
    ):
        raise ValueError("unused diagnostic and exact public source required")
    r = json.loads(rp.read_text())
    if (
        r["TRAIN_only"] is not True
        or r["Fresh_verified"] is not False
        or len(r["records"]) != 80
        or not 0 <= a.index < 80
    ):
        raise ValueError("only complete public consumedTRAIN80 permitted")
    row = r["records"][a.index]
    paths = {k: a.root / row["artifacts"][k]["path"] for k in ("A", "physical")}
    pins = {str(rp): a.source_hash}
    for k, path in paths.items():
        if path.is_symlink() or not path.resolve().is_relative_to(a.root.resolve()):
            raise ValueError("confined physical primitives required")
        pins[str(path)] = row["artifacts"][k]["hash"]
    scene = a.root / "scene.xml"
    pins[str(scene)] = r["scene_hash"]
    policies = [Path(x) for x in r["input_file_hashes"] if x.endswith("/ckp/policy_30000.onnx")]
    if len(policies) != 1:
        raise ValueError("exact qualified named29 source policy required")
    policy = policies[0]
    pins[str(policy)] = r["input_file_hashes"][str(policy)]
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    import onnx

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.paid_teacher_policy import strict_parameters

    graph = onnx.load(policy, load_external_data=False)
    names, params = strict_parameters({x.key: x.value for x in graph.metadata_props})
    order = np.array([names.index(n) for n in G1_DDS_JOINT_NAMES])
    model = mujoco.MjModel.from_xml_path(str(scene))
    qi = np.array([model.jnt_qposadr[model.joint("a/" + n).id] for n in G1_DDS_JOINT_NAMES])
    vi = np.array([model.jnt_dofadr[model.joint("a/" + n).id] for n in G1_DDS_JOINT_NAMES])
    with np.load(paths["A"], allow_pickle=False) as z:
        nn = {k: z[k].copy() for k in ("executed_action", "NN_reference_nominal")}
    with np.load(paths["physical"], allow_pickle=False) as z:
        raw = {k: z[k].copy() for k in ("target_a", "qpos", "qvel")}
    if raw["qpos"].shape[1:] != (10, model.nq) or raw["qvel"].shape[1:] != (10, model.nv):
        raise ValueError("actual compiled substep layout required")
    scale, offset = params["action_scale"][order], params["default_joint_pos"][order]
    target = offset + nn["executed_action"][:, order] * scale
    reference = offset + nn["NN_reference_nominal"][:, order] * scale
    if not np.array_equal(target, raw["target_a"]):
        raise ValueError("command must exactly reconstruct ACTUAL executed HG29 target")
    summary = summarize(
        target,
        reference,
        np.take(raw["qpos"][:, -1], qi, axis=1),
        np.take(raw["qvel"][:, -1], vi, axis=1),
        list(G1_DDS_JOINT_NAMES),
    )
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="PUBLIC_TRAIN_NAMED_TARGET_RESPONSE_DIAGNOSED_NOT_NEW_PHYSICS",
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_index=a.index,
            **summary,
            physical_steps=0,
            optimizer_updates=0,
            Fresh_cases_read=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
