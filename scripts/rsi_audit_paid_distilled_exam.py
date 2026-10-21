"""Independent primitive/torque/selection audit of every neural-only exam raw.

No new neural inference or physics. Independent whole neural+native repeats
already belong to the producer; this audit does not relabel them as its work.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hash", required=True)
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    path = a.root / "complete.json"
    if (
        a.output.exists()
        or digest(path) != a.hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact completed exam/frozen audit code required")
    r = json.loads(path.read_text())
    pins = {**r["input_file_hashes"], str(path): a.hash}
    teacher_declaration = a.teacher_root / "declaration.json"
    if (
        str(teacher_declaration) not in pins
        or r["status"] != "PAID_DISTILLED_CPU_NEURAL_ONLY_EXAM_CLOSED"
        or r["Cartesian_intervention"] is not False
        or r["teacher_mixture"] is not False
        or r["all_full_neural_native_repeats_exact"] is not True
    ):
        raise ValueError("actual neural-only completed same-body exam required")
    d = json.loads(teacher_declaration.read_text())
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            all288_raws_required=True,
            own_neural_rows=0,
            own_native_steps=0,
            hardware_authorized=False,
        ),
    )
    import mujoco
    import onnx

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.native_contact_window import native_contact_window
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.paid_teacher_policy import strict_parameters

    policies = [Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx")]
    xmls = [Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml")]
    if len(policies) != 1 or not xmls or len({digest(f) for f in xmls}) != 1:
        raise ValueError("one original actor and identical scene contracts required")
    graph = onnx.load(policies[0], load_external_data=False)
    names, params = strict_parameters({x.key: x.value for x in graph.metadata_props})
    order = np.array([names.index(n) for n in G1_DDS_JOINT_NAMES])
    offset, scale, kp, kd = [
        params[k][order]
        for k in ("default_joint_pos", "action_scale", "joint_stiffness", "joint_damping")
    ]
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
    model = mujoco.MjModel.from_xml_path(str(xmls[0]))
    layout = NativeVelocityLayout(model, robot_prefix="", ball_body_name="soccer_ball")
    mocap_id = model.body_mocapid[model.body("soccer_goal_marker").id]
    actual_native = actual_neural = 0
    raws = 0

    def quality(row):
        post = row["postcontact"]
        return (
            not row["body_failed"]
            and not row["forbidden"]
            and post["correct_foot_first"]
            and post["post_contact_quality_observed"]
            and post["goal_progress_1p2s_m"] >= 1
            and abs(post["goal_lateral_1p2s_m"]) <= 0.75
            and not post["other_foot_contact_in_window"]
            and post["maximum_pelvis_retreat_after_contact_m"] <= 0.05
        )

    def read(row, case, repeat=False):
        nonlocal actual_native, actual_neural, raws
        file = a.root / row["raw_path"]
        if not file.resolve().is_relative_to(a.root.resolve()) or digest(file) != row["raw_hash"]:
            raise ValueError("original confined raw hash differs")
        pins[str(file)] = row["raw_hash"]
        with np.load(file, allow_pickle=False) as z:
            z = {k: z[k].copy() for k in z.files}
        if (
            any(k.startswith("PLAN_") for k in z)
            or row["composite_teacher"]
            or any(not np.isfinite(x).all() for x in z.values())
        ):
            raise ValueError("finite raw and no Cartesian intervention required")
        n, controls = len(z["NN_action"]), len(z["target"])
        if (
            n != row["neural_rows"]
            or controls * 10 != row["native_steps"]
            or not np.array_equal(z["neural_control"], np.arange(controls) < n)
            or not np.array_equal(z["NN_action"], z["NN_nominal_action"])
        ):
            raise ValueError("actual neural output/execution/clock identity differs")
        if (
            not np.array_equal(z["NN_h_before"][1:], z["NN_h_after"][:-1])
            or not np.array_equal(z["NN_c_before"][1:], z["NN_c_after"][:-1])
            or z["NN_h_before"][0].any()
            or z["NN_c_before"][0].any()
            or z["NN_obs"][0, 125:154].any()
            or not np.array_equal(z["NN_obs"][1:, 125:154], z["NN_action"][:-1])
        ):
            raise ValueError("episode RNN/actual previous-action history differs")
        target = offset + z["NN_action"][:, order] * scale
        if not np.array_equal(target, z["target"][:n]) or not np.array_equal(
            z["target"][n:], np.broadcast_to(target[-1], z["target"][n:].shape)
        ):
            raise ValueError("actual pure-neural named PD targets/held tail differ")
        qb = np.concatenate(
            (z["native_qpos_before"][:, layout.qi][:, None], z["qpos"][:, :-1, 7:36]), axis=1
        )
        vb = np.concatenate(
            (z["native_qvel_before"][:, layout.vi][:, None], z["qvel"][:, :-1, 6:35]), axis=1
        )
        if not np.array_equal(
            np.clip(kp * (z["target"][:, None] - qb) - kd * vb, -effort, effort), z["torque"]
        ):
            raise ValueError("every2ms actual bounded torque law differs")
        q, v, f = [
            z[k].reshape(-1, width) for k, width in (("qpos", 43), ("qvel", 41), ("forces", 3))
        ]
        goal = z["initial_mocap_pos"][mocap_id]
        if "goal_world" in case and not np.array_equal(goal, np.array(case["goal_world"])):
            raise ValueError("old physical target changed")
        post = native_contact_window(
            force=f[:, :2],
            canonical_qpos=q,
            canonical_qvel=v,
            goal=goal,
            intended_foot=0 if "left" in row["source"] else 1,
        )
        bad = (
            (z["qpos"][:, :, 2] < 0.65)
            | (1 - 2 * (z["qpos"][:, :, 4] ** 2 + z["qpos"][:, :, 5] ** 2) < 0.9)
        ).any(axis=1)
        forbidden = bool((f[:, 2] > 1).any())
        impulse = float((f[:, 2] * 0.002).sum())
        reward = -12.0 if bad.any() else -8 - 6 * forbidden
        if not bad.any() and post["correct_foot_first"] and post["post_contact_quality_observed"]:
            distance = float(
                np.linalg.norm(q[int(post["first_foot_contact_substep"]) + 600, 36:39] - goal)
            )
            reward = (
                2 * (6 - min(12, distance))
                - 6 * forbidden
                - min(3, 2 * impulse)
                + 4 * (not forbidden)
                - 0.5 * post["maximum_pelvis_retreat_after_contact_m"]
            )
        if (
            not np.array_equal(bad, z["body_failed"])
            or bool(bad.any()) != row["body_failed"]
            or (bad.any() and not np.array_equal(np.flatnonzero(bad), [controls - 1]))
            or post != row["postcontact"]
            or forbidden != row["forbidden"]
            or impulse != row["nonfoot_impulse_Ns"]
            or reward != row["terminal_reward"]
        ):
            raise ValueError("independent primitive physical outcomes differ")
        actual_native += controls * 10
        actual_neural += n
        raws += 1
        if repeat:
            other = file.with_name(file.name.replace("-repeat0.npz", "-repeat1.npz"))
            if other == file or digest(other) != row["independent_repeat_raw_hash"]:
                raise ValueError("exact independently executed repeat required")
            pins[str(other)] = row["independent_repeat_raw_hash"]
            with np.load(other, allow_pickle=False) as second:
                if set(second.files) != set(z) or any(
                    not np.array_equal(z[k], second[k]) for k in z
                ):
                    raise ValueError("all raw independent NN/native repeat arrays differ")
            actual_native += controls * 10
            actual_neural += n
            raws += 1

    try:
        for key, cases in (("old16", d["train_cases"]), ("dev16", d["dev_cases"])):
            if len(r["baseline"][key]) != 16:
                raise ValueError("complete baseline required")
            for row, case in zip(r["baseline"][key], cases, strict=True):
                read(row, case)
        summaries = []
        if len(r["candidates"]) != 4:
            raise ValueError("all four candidates required")
        for candidate in r["candidates"]:
            scores = {}
            for key, cases in (("old16", d["train_cases"]), ("dev16", d["dev_cases"])):
                section = candidate["results"][key]
                rows = section["results"]
                if len(rows) != 16:
                    raise ValueError("all declared case outcomes required")
                for row, case in zip(rows, cases, strict=True):
                    read(row, case, True)
                scores[key] = dict(
                    stable=sum(not x["body_failed"] for x in rows),
                    quality=sum(quality(x) for x in rows),
                    clean=sum(
                        not x["body_failed"]
                        and not x["forbidden"]
                        and x["postcontact"]["correct_foot_first"]
                        for x in rows
                    ),
                    rescues=[
                        i
                        for i, (x, y) in enumerate(zip(r["baseline"][key], rows, strict=True))
                        if not quality(x) and quality(y)
                    ],
                    regressions=[
                        i
                        for i, (x, y) in enumerate(zip(r["baseline"][key], rows, strict=True))
                        if quality(x) and not quality(y)
                    ],
                )
                if any(section[k] != v for k, v in scores[key].items()):
                    raise ValueError("reported skill statistics differ")
            passed = (
                scores["old16"]["quality"] >= 14
                and scores["dev16"]["quality"] >= 12
                and all(x["stable"] == 16 and not x["regressions"] for x in scores.values())
            )
            expected = (
                "NEURAL_ONLY_DISTILLATION_SKILL_GATE_PASSED"
                if passed
                else "NEURAL_ONLY_DISTILLATION_SKILL_GATE_NOT_MET"
            )
            if candidate["status"] != expected:
                raise ValueError("research skill gate differs")
            summaries.append(
                dict(candidate=candidate["candidate_index"], status=expected, results=scores)
            )
        if (
            raws != 288
            or actual_native != r["native_steps"]
            or actual_neural != r["new_neural_rows"]
        ):
            raise ValueError("all real execution/neural counts differ")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_DISTILLED_ALL_RAW_PRIMITIVE_AUDIT_PASSED",
                input_file_hashes=pins,
                results=summaries,
                all_raws=raws,
                producer_native_steps=actual_native,
                producer_neural_rows=actual_neural,
                own_native_steps=0,
                own_neural_rows=0,
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
                raws_read=raws,
                own_native_steps=0,
                own_neural_rows=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
