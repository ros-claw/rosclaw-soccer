"""Independently audit all factorial memory-branch primitive evidence.

No new physics or neural calls: producer repeats are counted as producer work.
"""

import argparse
import ast
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hash", required=True)
    p.add_argument("--parent-root", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if (
        a.output.exists()
        or digest(rp) != a.hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact source/completed branch receipt required")
    r, d = json.loads(rp.read_text()), json.loads(dp.read_text())
    if r["status"] != "PAID_LIVE_MEMORY_FACTORIAL_EXAM_CLOSED" or len(r["results"]) != 64:
        raise ValueError("all64 two-repeat branch outcomes required")
    pins = {**r["input_file_hashes"], str(rp): a.hash, str(dp): digest(dp)}
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins, own_native_steps=0, own_neural_rows=0, hardware_authorized=False
        ),
    )
    import mujoco
    import numpy as np
    import onnx

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.paid_primitive_audit import PaidPrimitiveAuditor
    from rosclaw_soccer.rsi.paid_teacher_policy import strict_parameters

    policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
    constants = policy.parent.parent / "exp/mujoco_soccer/constants.py"
    if str(constants) not in pins:
        raise ValueError("source-planned body order must be pinned")
    body_names = next(
        ast.literal_eval(node.value)
        for node in ast.parse(constants.read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "BODY_NAMES" for target in node.targets
        )
    )
    torso_index = body_names.index("torso_link")
    xml = next(Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml"))
    graph = onnx.load(policy, load_external_data=False)
    names, params = strict_parameters({x.key: x.value for x in graph.metadata_props})
    order = np.array([names.index(n) for n in G1_DDS_JOINT_NAMES])
    effort = np.array(
        [
            139
            if ("hip_roll" in n or "knee" in n)
            else 88
            if ("hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint")
            else 50
            if ("ankle" in n or n in ("waist_roll_joint", "waist_pitch_joint"))
            else 5
            if ("wrist_pitch" in n or "wrist_yaw" in n)
            else 25
            for n in G1_DDS_JOINT_NAMES
        ]
    )
    model = mujoco.MjModel.from_xml_path(str(xml))
    layout = NativeVelocityLayout(model, robot_prefix="", ball_body_name="soccer_ball")
    auditor = PaidPrimitiveAuditor(
        layout=layout,
        parameters=params,
        order=order,
        effort=effort,
        goal_mocap_id=int(model.body_mocapid[model.body("soccer_goal_marker").id]),
    )
    actual_native = actual_neural = 0
    try:
        seen = set()
        for row in r["results"]:
            i, ai = row["case"], row["arm"]
            if (i, ai) in seen or i not in range(16) or ai not in range(4):
                raise ValueError("unique complete factorial branch bank required")
            seen.add((i, ai))
            arm = d["arms"][ai]
            parent_path = a.parent_root / f"parent{i}/parent.npz"
            first_path = a.parent_root / f"parent{i}/first.npz"
            if str(parent_path) not in pins or str(first_path) not in pins:
                raise ValueError("actual producer parent must be pinned before use")
            with np.load(parent_path, allow_pickle=False) as z:
                parent = {k: z[k].copy() for k in z.files}
            with np.load(first_path, allow_pickle=False) as z:
                memory = (
                    tuple(z["NN_" + k + "_after"][-1].copy() for k in ("h", "c"))
                    if arm["carry_h_c"]
                    else None
                )
            motion_path = policy.parent.parent / "motions/soccer-standard" / d["cases"][i]["source"]
            if str(motion_path) not in pins:
                raise ValueError("exact source motion must be qualified")
            with np.load(motion_path, allow_pickle=False) as z:
                horizon = len(z["joint_pos"])
                refpos = z["joint_pos"][-1] if arm["carry_reference"] else np.zeros(29, np.float32)
                refvel = z["joint_vel"][-1] if arm["carry_reference"] else np.zeros(29, np.float32)
                refang = (
                    z["body_ang_vel_w"][-1, torso_index]
                    if arm["carry_reference"]
                    else np.zeros(3, np.float32)
                )
                motion_q = z["joint_pos"].copy()
                motion_v = z["joint_vel"].copy()
                motion_w = z["body_ang_vel_w"][:, torso_index].copy()
            baseline = None
            for repeat in (0, 1):
                path = a.root / f"arm{ai}-case{i}-repeat{repeat}.npz"
                expected = row["raw_hash"] if repeat == 0 else row["independent_repeat_raw_hash"]
                if digest(path) != expected:
                    raise ValueError("exact independently executed branch hash required")
                pins[str(path)] = expected
                with np.load(path, allow_pickle=False) as z:
                    raw = {k: z[k].copy() for k in z.files}
                if any(
                    not np.array_equal(v, parent[k.removeprefix("initial_")])
                    for k, v in raw.items()
                    if k.startswith("initial_")
                ):
                    raise ValueError("physical parent was changed at skill entry")
                if (
                    row["source"] != d["cases"][i]["source"]
                    or not np.array_equal(raw["NN_obs"][0, :29], refpos)
                    or not np.array_equal(raw["NN_obs"][0, 29:58], refvel)
                    or not np.array_equal(raw["NN_obs"][0, 61:64], refang)
                    or not np.array_equal(
                        raw["NN_obs"][1:, :29], motion_q[: len(raw["NN_action"]) - 1]
                    )
                    or not np.array_equal(
                        raw["NN_obs"][1:, 29:58], motion_v[: len(raw["NN_action"]) - 1]
                    )
                    or not np.array_equal(
                        raw["NN_obs"][1:, 61:64], motion_w[: len(raw["NN_action"]) - 1]
                    )
                ):
                    raise ValueError("declared reference-memory factor differs")
                checked = auditor.audit(
                    raw,
                    row,
                    horizon=horizon,
                    previous_action=parent["previous_actual_action"],
                    initial_memory=memory,
                )
                if checked["quality"] != row["research_contact_quality"]:
                    raise ValueError("research quality summary differs")
                actual_native += checked["native_steps"]
                actual_neural += checked["neural_rows"]
                if repeat == 0:
                    baseline = raw
                elif set(raw) != set(baseline) or any(
                    not np.array_equal(raw[k], baseline[k]) for k in raw
                ):
                    raise ValueError("independent whole-neural/native branch repeat differs")
        summaries = [
            dict(
                arm=arm,
                quality=sum(x["research_contact_quality"] for x in r["results"] if x["arm"] == ai),
                stable=sum(not x["body_failed"] for x in r["results"] if x["arm"] == ai),
                forbidden=sum(x["forbidden"] for x in r["results"] if x["arm"] == ai),
            )
            for ai, arm in enumerate(d["arms"])
        ]
        if (
            summaries != r["summaries"]
            or actual_native != r["actual_native_steps"]
            or actual_neural != r["actual_neural_rows"]
        ):
            raise ValueError("all128 actual branch totals differ")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_MEMORY_PRIMITIVE_AUDIT_PASS",
                input_file_hashes=pins,
                summaries=summaries,
                raws=128,
                producer_native_steps=actual_native,
                producer_neural_rows=actual_neural,
                own_native_steps=0,
                own_neural_rows=0,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(a.output / "failed.json", dict(error=repr(error), hardware_authorized=False))
        raise


if __name__ == "__main__":
    main()
