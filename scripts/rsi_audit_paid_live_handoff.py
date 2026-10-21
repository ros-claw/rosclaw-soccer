"""Read-only physical continuity audit of every live-body practice world.

Checks boundary/drop/hold/native-step/action/torque/contact chains. Counts no
new inference or physics; independent producer repeats remain producer work.
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
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rp = a.root / "complete.json"
    if (
        a.output.exists()
        or digest(rp) != a.hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("exact completed source/receipt and unused output required")
    r = json.loads(rp.read_text())
    declaration = json.loads((a.root / "declaration.json").read_text())
    pins = {
        **r["input_file_hashes"],
        str(rp): a.hash,
        str(a.root / "declaration.json"): digest(a.root / "declaration.json"),
    }
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    if (
        r["status"] != "PAID_CONTINUOUS_BODY_PRACTICE_EXAM_CLOSED"
        or r["team_game"]
        or r["robot_state_reset_between_skills"]
        or len(r["results"]) != 16
        or len(declaration["matrix"]) != 16
    ):
        raise ValueError("all declared 16 practice sequences required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins, own_native_steps=0, own_neural_rows=0, hardware_authorized=False
        ),
    )
    import mujoco
    import onnx

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.native_contact_window import native_contact_window
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.paid_teacher_policy import strict_parameters

    policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
    xml = next(Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml"))
    graph = onnx.load(policy, load_external_data=False)
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
    model = mujoco.MjModel.from_xml_path(str(xml))
    layout = NativeVelocityLayout(model, robot_prefix="", ball_body_name="soccer_ball")
    mocap_id = model.body_mocapid[model.body("soccer_goal_marker").id]
    count_native = count_neural = boundaries = drops = 0

    def read(path, expected=None):
        if not path.resolve().is_relative_to(a.root.resolve()) or path.is_symlink():
            raise ValueError("confined ordinary raw artifact required")
        h = digest(path)
        if expected and h != expected:
            raise ValueError("sealed raw hash differs")
        pins[str(path)] = h
        with np.load(path, allow_pickle=False) as z:
            raw = {k: z[k].copy() for k in z.files}
        if any(not np.isfinite(v).all() for v in raw.values()):
            raise ValueError("all actual arrays must be finite")
        return raw

    def canonical(q, v):
        return (
            np.concatenate(
                (
                    q[..., layout.rq : layout.rq + 7],
                    q[..., layout.qi],
                    q[..., layout.bq : layout.bq + 7],
                ),
                axis=-1,
            ),
            np.concatenate(
                (
                    v[..., layout.rv : layout.rv + 6],
                    v[..., layout.vi],
                    v[..., layout.bv : layout.bv + 6],
                ),
                axis=-1,
            ),
        )

    def same(x, y):
        if set(x) != set(y) or any(not np.array_equal(x[k], y[k]) for k in x):
            raise ValueError("full native boundary integration-state continuity differs")

    def state(raw, prefix):
        return {k.removeprefix(prefix): v for k, v in raw.items() if k.startswith(prefix)}

    def quality(row):
        x = row["postcontact"]
        return bool(
            not row["body_failed"]
            and not row["forbidden"]
            and x["correct_foot_first"]
            and x["post_contact_quality_observed"]
            and x["goal_progress_1p2s_m"] >= 1
            and abs(x["goal_lateral_1p2s_m"]) <= 0.75
            and not x["other_foot_contact_in_window"]
            and x["maximum_pelvis_retreat_after_contact_m"] <= 0.05
        )

    def block(z, entry, previous, *, arm, row=None):
        nonlocal count_native, count_neural
        n = len(z["target"])
        qb, vb = canonical(z["native_qpos_before"], z["native_qvel_before"])
        eq, ev = canonical(entry["qpos"], entry["qvel"])
        if (
            not np.array_equal(qb[0], eq)
            or not np.array_equal(vb[0], ev)
            or not np.array_equal(qb[1:], z["qpos"][:-1, -1])
            or not np.array_equal(vb[1:], z["qvel"][:-1, -1])
            or not np.isclose(z["time_before"][0], entry["time"], rtol=0, atol=1e-11)
            or not np.allclose(np.diff(z["time_before"]), 0.02, rtol=0, atol=1e-11)
        ):
            raise ValueError("every actual control boundary/time must be continuous")
        jq = np.concatenate((qb[:, None, 7:36], z["qpos"][:, :-1, 7:36]), axis=1)
        jv = np.concatenate((vb[:, None, 6:35], z["qvel"][:, :-1, 6:35]), axis=1)
        if not np.array_equal(
            np.clip(kp * (z["target"][:, None] - jq) - kd * jv, -effort, effort), z["torque"]
        ):
            raise ValueError("actual bounded native PD torque differs")
        bad = (
            (z["qpos"][:, :, 2] < 0.65)
            | (1 - 2 * (z["qpos"][:, :, 4] ** 2 + z["qpos"][:, :, 5] ** 2) < 0.9)
        ).any(axis=1)
        if not np.array_equal(bad, z["body_failed"]) or (bad[:-1].any()):
            raise ValueError("body failure must stop at its first control boundary")
        count_native += n * 10
        if row is None:
            if z["neural_control"].any() or not z["post_reference_hold"].all():
                raise ValueError("practice settling must not pad neural/reference frames")
            return previous
        nn = len(z["NN_action"])
        count_neural += nn
        if (
            not np.array_equal(z["neural_control"], np.arange(n) < nn)
            or not np.array_equal(z["NN_obs"][0, 125:154], previous)
            or not np.array_equal(z["NN_obs"][1:, 125:154], z["NN_action"][:-1])
            or z["NN_h_before"][0].any()
            or z["NN_c_before"][0].any()
            or not np.array_equal(z["NN_h_before"][1:], z["NN_h_after"][:-1])
            or not np.array_equal(z["NN_c_before"][1:], z["NN_c_after"][:-1])
        ):
            raise ValueError("explicit new RNN boundary and actual previous action must agree")
        expected = z["NN_nominal_action"].copy()
        for i in range(min(nn, arm)):
            u = (i + 1) / arm
            w = 10 * u**3 - 15 * u**4 + 6 * u**5
            expected[i] = ((1 - w) * previous + w * expected[i]).astype(np.float32)
        targets = offset + z["NN_action"][:, order] * scale
        if (
            not np.array_equal(expected, z["NN_action"])
            or not np.array_equal(targets, z["target"][:nn])
            or not np.array_equal(
                z["target"][nn:], np.broadcast_to(targets[-1], z["target"][nn:].shape)
            )
        ):
            raise ValueError("declared neural/transition target law differs")
        q, v, f = [
            z[k].reshape(-1, width) for k, width in (("qpos", 43), ("qvel", 41), ("forces", 3))
        ]
        post = native_contact_window(
            force=f[:, :2],
            canonical_qpos=q,
            canonical_qvel=v,
            goal=entry["mocap_pos"][mocap_id],
            intended_foot=0 if "left" in row["source"] else 1,
        )
        if (
            post != row["postcontact"]
            or bool(bad.any()) != row["body_failed"]
            or bool((f[:, 2] > 1).any()) != row["forbidden"]
            or quality(row) != row["research_contact_quality"]
            or nn != row["neural_rows"]
            or n * 10 != row["native_steps"]
        ):
            raise ValueError("independent primitive contact/quality/body outcome differs")
        return z["NN_action"][-1].copy()

    try:
        for index, sealed in enumerate(r["results"]):
            paired_artifacts = []
            for repeat in (0, 1):
                folder = a.root / f"world{index}-repeat{repeat}"
                wp = folder / "complete.json"
                pins[str(wp)] = digest(wp)
                world = json.loads(wp.read_text())
                if repeat == 0 and world != sealed:
                    raise ValueError("sealed full world receipt differs")
                if world["spec"] != declaration["matrix"][index] or world["body_resets"] != 1:
                    raise ValueError("predeclared world and one initial reset required")
                previous = np.zeros(29, np.float32)
                terminal = entry = None
                phase_rows = {s["phase"]: s for s in world["segments"] if "postcontact" in s}
                blocks = []
                for artifact in world["artifacts"]:
                    path = a.root / artifact["path"]
                    z = read(path, artifact["hash"])
                    blocks.append(z)
                    name = path.stem
                    if name.startswith("skill"):
                        phase = int(name.removeprefix("skill"))
                        initial = state(z, "initial_")
                        if entry is not None:
                            # initial snapshot is the subset historically stored by the episode.
                            same(initial, {k: entry[k] for k in initial})
                        entry = initial
                        previous = block(
                            z,
                            entry,
                            previous,
                            arm=world["spec"]["transition_controls"] if phase else 0,
                            row=phase_rows[phase],
                        )
                        terminal = z
                    elif name.startswith("terminal"):
                        if terminal is None:
                            raise ValueError("terminal requires preceding physical controls")
                        q, v = canonical(z["qpos"], z["qvel"])
                        if (
                            not np.array_equal(q, terminal["qpos"][-1, -1])
                            or not np.array_equal(v, terminal["qvel"][-1, -1])
                            or not np.array_equal(z["ctrl"][layout.ai], terminal["torque"][-1, -1])
                            or not np.isclose(
                                z["time"], terminal["time_before"][-1] + 0.02, rtol=0, atol=1e-11
                            )
                        ):
                            raise ValueError("actual terminal state/time differs")
                        entry = z
                    elif name.startswith("drop"):
                        before, after = state(z, "before_"), state(z, "after_")
                        same(entry, before)
                        q, _ = canonical(before["qpos"], before["qvel"])
                        performed = bool(z["performed"])
                        if performed != bool(np.linalg.norm(q[36:38] - q[:2]) >= 2):
                            raise ValueError("ball return violated declared out-of-zone rule")
                        if performed:
                            direction = before["mocap_pos"][mocap_id, :2] - q[:2]
                            direction /= np.linalg.norm(direction)
                            foot = world["spec"]["sources"][int(name.removeprefix("drop"))]
                            sign = 1 if "left" in foot else -1
                            point = (
                                q[:2]
                                + 0.8 * direction
                                + sign * 0.15 * np.array([-direction[1], direction[0]])
                            )
                            expected_ball = np.r_[point, 0.4, 1.0, 0, 0, 0]
                            if (
                                not np.array_equal(
                                    after["qpos"][layout.bq : layout.bq + 7], expected_ball
                                )
                                or after["qvel"][layout.bv : layout.bv + 6].any()
                            ):
                                raise ValueError(
                                    "declared gravity-drop point or zero release velocity differs"
                                )
                            modified = {k: x.copy() for k, x in after.items()}
                            modified["qpos"][layout.bq : layout.bq + 7] = before["qpos"][
                                layout.bq : layout.bq + 7
                            ]
                            modified["qvel"][layout.bv : layout.bv + 6] = before["qvel"][
                                layout.bv : layout.bv + 6
                            ]
                            same(before, modified)
                            entry = after
                            drops += 1
                    elif name.startswith("hold"):
                        block(z, entry, previous, arm=0)
                        terminal = z
                    elif name.startswith("boundary"):
                        before, after = state(z, "before_"), state(z, "after_")
                        same(before, after)
                        if terminal is not None:
                            q, v = canonical(before["qpos"], before["qvel"])
                            if not np.array_equal(
                                q, terminal["qpos"][-1, -1]
                            ) or not np.array_equal(v, terminal["qvel"][-1, -1]):
                                raise ValueError("pre-skill body state was reset")
                        entry = after
                        boundaries += 1
                    elif name.startswith("rejected"):
                        same(entry, z)
                    else:
                        raise ValueError("unknown physical artifact")
                paired_artifacts.append(blocks)
            if len(paired_artifacts[0]) != len(paired_artifacts[1]):
                raise ValueError("repeat artifact count differs")
            for left, right in zip(*paired_artifacts, strict=True):
                same(left, right)
        if count_native != r["actual_native_steps"] or count_neural != r["actual_neural_rows"]:
            raise ValueError("actual work ledger differs from all 32 world raw chains")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="CONTINUOUS_BODY_PRACTICE_PRIMITIVE_AUDIT_PASS",
                input_file_hashes=pins,
                producer_native_steps=count_native,
                producer_neural_rows=count_neural,
                checked_skill_boundaries=boundaries,
                checked_ball_drops=drops,
                own_native_steps=0,
                own_neural_rows=0,
                goal_complete=False,
                team_game=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(a.output / "failed.json", dict(error=repr(error), hardware_authorized=False))
        raise


if __name__ == "__main__":
    main()
