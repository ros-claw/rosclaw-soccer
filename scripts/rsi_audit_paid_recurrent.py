"""Independently replay every recurrent learning/evaluation physical episode.

Reconstruct rewards/GAE, whole-prefix behavior likelihood and safe checkpoint
Adam counters. No policy promotion or publicity rights are inferred.
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
    receipt = a.root / "complete.json"
    if (
        a.output.exists()
        or digest(receipt) != a.hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("exact completed training and unused frozen audit source required")
    r = json.loads(receipt.read_text())
    if r["status"] != "PAID_RECURRENT_PPO_RESEARCH_CLOSED_NOT_PROMOTED":
        raise ValueError("full recurrent training receipt required")
    declaration = a.root / "declaration.json"
    d = json.loads(declaration.read_text())
    pins = {**r["input_file_hashes"], str(receipt): a.hash, str(declaration): digest(declaration)}
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            all_raw_native_episodes=True,
            whole_prefix_behavior_likelihood=True,
            likelihood_absolute_tolerance=0.001,
            state_absolute_tolerance=0.0002,
            safe_Adam_steps=True,
            optimizer_gradient_replay=False,
            commercial_publicity_authorized=False,
            hardware_authorized=False,
        ),
    )
    native = neural = critic_rows = 0
    try:
        import mujoco
        import torch
        from torch import nn

        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH, PaidTrainableActor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policies = [Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx")]
        if len(policies) != 1:
            raise ValueError("one pinned research full actor required")
        contract = PaidResearchPolicy(policies[0], expected_hash=PUBLISHED_HASH, horizon=1)
        actor = PaidTrainableActor(policies[0], expected_hash=PUBLISHED_HASH).eval()
        reference = {k: v.clone() for k, v in actor.state_dict().items()}
        critic = nn.Sequential(
            nn.Linear(160, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1)
        )
        order = contract.hg_from_policy
        offset, scale, kp, kd = [
            contract.parameters[k][order]
            for k in ("default_joint_pos", "action_scale", "joint_stiffness", "joint_damping")
        ]
        from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES

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
        xmls = [Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml")]
        # Several receipts may pin the same recipe through different roots;
        # all must be exactly the same XML, not a selected favorable model.
        if not xmls or len({digest(f) for f in xmls}) != 1:
            raise ValueError("identical standard compiled scene recipes required")
        model = mujoco.MjModel.from_xml_path(str(xmls[0]))
        if mujoco.get_mjcb_control() is not None:
            raise ValueError("no external control callback")
        layout = NativeVelocityLayout(model, robot_prefix="", ball_body_name="soccer_ball")
        feet = [model.body(s + "_ankle_roll_link").id for s in ("left", "right")]
        robot = set(range(1, model.nbody)) - {layout.ball, model.body("soccer_goal_marker").id}

        def forces(data):
            f = np.zeros(3)
            for ci in range(data.ncon):
                c = data.contact[ci]
                x, y = (int(model.geom_bodyid[g]) for g in c.geom)
                other = y if x == layout.ball else x if y == layout.ball else -1
                if other in robot:
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, wrench)
                    role = feet.index(other) if other in feet else 2
                    f[role] = max(f[role], np.linalg.norm(wrench[:3]))
            return f

        checkpoints = []
        for iteration in range(r["config"]["iterations"] + 1):
            path = a.root / f"recurrent-iteration{iteration}.pt"
            h = (
                r["initial_checkpoint_hash"]
                if iteration == 0
                else r["history"][iteration - 1]["checkpoint_hash"]
            )
            if digest(path) != h:
                raise ValueError("saved checkpoint hash differs")
            pins[str(path)] = h
            cp = torch.load(path, weights_only=True, map_location="cpu")
            if (
                cp["schema"] != "soccer.paid_recurrent_research.v1"
                or cp["iteration"] != iteration
                or cp["declaration_hash"] != digest(declaration)
                or cp["config"] != r["config"]
            ):
                raise ValueError("exact safe recurrent checkpoint required")
            if set(cp["anchor"]) != set(reference) or any(
                not torch.equal(v, cp["anchor"][k]) for k, v in reference.items()
            ):
                raise ValueError("immutable original full teacher anchor changed")
            if any(not torch.equal(cp["actor"][k], reference[k]) for k in ("mean", "denominator")):
                raise ValueError("normalizer changed")
            actor_steps, all_steps = cp["actor_optimizer_updates"], cp["optimizer_updates"]
            groups = cp["optimizer"]["param_groups"]
            ids = groups[0]["params"]
            if len(groups) != 1 or len(ids) != 23:
                raise ValueError("16actor+6critic+1std explicit Adam parameter group required")
            for i, pid in enumerate(ids):
                expected = all_steps if 16 <= i < 22 else actor_steps
                state = cp["optimizer"]["state"].get(pid)
                if expected == 0:
                    if state is not None:
                        raise ValueError("frozen optimizer slot unexpectedly exists")
                elif (
                    state is None
                    or float(state["step"]) != expected
                    or not all(torch.isfinite(state[n]).all() for n in ("exp_avg", "exp_avg_sq"))
                ):
                    raise ValueError("actual safe Adam step count/moments differ")
            checkpoints.append(cp)
        plans = [("initial-dev", r["initial_dev"], 0, False, d["dev_cases"])]
        plans += [
            (f"iteration{i + 1}", row["results"], i, True, d["train_cases"])
            for i, row in enumerate(r["history"])
        ]
        plans += [
            ("final-dev", r["final_dev"], len(checkpoints) - 1, False, d["dev_cases"]),
            ("final-old16", r["final_old16"], len(checkpoints) - 1, False, d["old_cases"]),
        ]
        results = []
        rollout_actor = rollout_critic = 0
        for folder, rows, cp_index, stochastic, cases in plans:
            if len(rows) != len(cases):
                raise ValueError("all declared episodes required")
            cp = checkpoints[cp_index]
            actor.load_state_dict(cp["actor"], strict=True)
            critic.load_state_dict(cp["critic"], strict=True)
            for index, row in enumerate(rows):
                path = a.root / folder / f"case{index}.npz"
                if digest(path) != row["raw_hash"]:
                    raise ValueError("actual training/evaluation raw changed")
                pins[str(path)] = row["raw_hash"]
                with np.load(path, allow_pickle=False) as z:
                    z = {k: z[k].copy() for k in z.files}
                n, controls = len(z["NN_obs"]), len(z["target"])
                if (
                    row["neural_rows"] != n
                    or row["native_steps"] != controls * 10
                    or not 0 <= controls - n <= 60
                    or not np.array_equal(z["neural_control"], np.arange(controls) < n)
                ):
                    raise ValueError("actual neural/source/tail clocks differ")
                rollout_actor += n
                rollout_critic += n if stochastic else 0
                if (
                    z["NN_h_before"][0].any()
                    or z["NN_c_before"][0].any()
                    or not np.array_equal(z["NN_h_before"][1:], z["NN_h_after"][:-1])
                    or not np.array_equal(z["NN_c_before"][1:], z["NN_c_after"][:-1])
                    or z["NN_obs"][0, 125:154].any()
                    or not np.array_equal(z["NN_obs"][1:, 125:154], z["NN_action"][:-1])
                ):
                    raise ValueError("actual per-episode h/c or previous action history differs")
                obs = torch.from_numpy(z["NN_obs"])
                with torch.no_grad():
                    mean, (h, c) = actor(
                        obs[:, None], (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
                    )
                    neural += n
                    if not np.allclose(
                        h.numpy(), z["NN_h_after"][-1], atol=0.0002, rtol=0
                    ) or not np.allclose(c.numpy(), z["NN_c_after"][-1], atol=0.0002, rtol=0):
                        raise ValueError("free whole recurrent history differs")
                    if stochastic:
                        dist = torch.distributions.Normal(
                            mean[:, 0], cp["log_std"].clamp(np.log(0.01), np.log(0.05)).exp()
                        )
                        logp = dist.log_prob(torch.from_numpy(z["NN_action"])).sum(-1).numpy()
                        value = critic((obs - actor.mean) / actor.denominator).squeeze(-1).numpy()
                        critic_rows += n
                        if not np.allclose(
                            logp, z["NN_logp"], atol=0.001, rtol=0
                        ) or not np.allclose(value, z["NN_value"], atol=0.00002, rtol=0):
                            raise ValueError(
                                "actual sampled likelihood/value differs from complete prefix"
                            )
                    elif not np.allclose(mean[:, 0].numpy(), z["NN_action"], atol=0.0002, rtol=0):
                        raise ValueError("deterministic evaluation actor differs")
                target = offset + z["NN_action"][:, order] * scale
                if not np.array_equal(target, z["target"][:n]) or (
                    controls > n
                    and not np.array_equal(
                        z["target"][n:], np.broadcast_to(target[-1], z["target"][n:].shape)
                    )
                ):
                    raise ValueError("actual named PD targets/held tail differ")
                q = np.concatenate(
                    (
                        z["native_qpos_before"][:, layout.qi, None].transpose(0, 2, 1),
                        z["qpos"][:, :-1, 7:36],
                    ),
                    axis=1,
                )
                v = np.concatenate(
                    (
                        z["native_qvel_before"][:, layout.vi, None].transpose(0, 2, 1),
                        z["qvel"][:, :-1, 6:35],
                    ),
                    axis=1,
                )
                if not np.array_equal(
                    np.clip(kp * (z["target"][:, None] - q) - kd * v, -effort, effort), z["torque"]
                ):
                    raise ValueError("all2ms clipped torque law differs")
                data = mujoco.MjData(model)
                for field in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
                    getattr(data, field)[:] = z["initial_" + field]
                data.time = float(z["initial_time"])
                mujoco.mj_forward(model, data)
                data.qacc_warmstart[:] = z["initial_qacc_warmstart"]
                for frame in range(controls):
                    mujoco.mj_forward(model, data)
                    for sub in range(10):
                        data.ctrl[layout.ai] = z["torque"][frame, sub]
                        mujoco.mj_step(model, data)
                        native += 1
                        cq, cv = layout.canonical(data)
                        if (
                            not np.array_equal(cq, z["qpos"][frame, sub])
                            or not np.array_equal(cv, z["qvel"][frame, sub])
                            or not np.array_equal(forces(data), z["forces"][frame, sub])
                        ):
                            raise ValueError("strict native all2ms q/v/contact replay differs")
                q, v, f = [
                    z[k].reshape(-1, width)
                    for k, width in (("qpos", 43), ("qvel", 41), ("forces", 3))
                ]
                goal = np.array(cases[index].get("goal", []))
                if goal.shape != (3,):
                    # Rotated target marker is part of the recorded native
                    # reset; no future observation or unpinned inferred target.
                    goal = z["initial_mocap_pos"][
                        model.body_mocapid[model.body("soccer_goal_marker").id]
                    ]
                post = native_contact_window(
                    force=f[:, :2],
                    canonical_qpos=q,
                    canonical_qvel=v,
                    goal=goal,
                    intended_foot=0 if "left" in row["source"] else 1,
                )
                forbidden = bool((f[:, 2] > 1).any())
                impulse = float((f[:, 2] * 0.002).sum())
                bad = (
                    (z["qpos"][:, :, 2] < 0.65)
                    | (1 - 2 * (z["qpos"][:, :, 4] ** 2 + z["qpos"][:, :, 5] ** 2) < 0.9)
                ).any(axis=1)
                if not np.array_equal(bad, z["body_failed"]) or (
                    bad.any() and not np.array_equal(np.flatnonzero(bad), [controls - 1])
                ):
                    raise ValueError("body failure/stop mask differs")
                reward = -12.0 if bad.any() else (-8.0 - 6 * forbidden)
                if (
                    not bad.any()
                    and post["post_contact_quality_observed"]
                    and post["correct_foot_first"]
                ):
                    distance = float(
                        np.linalg.norm(
                            q[int(post["first_foot_contact_substep"]) + 600, 36:39] - goal
                        )
                    )
                    reward = (
                        2 * (6 - min(12, distance))
                        - 6 * forbidden
                        - min(3, 2 * impulse)
                        + 4 * (not forbidden)
                        - 0.5 * post["maximum_pelvis_retreat_after_contact_m"]
                    )
                if (
                    post != row["postcontact"]
                    or forbidden != row["forbidden"]
                    or impulse != row["nonfoot_impulse_Ns"]
                    or reward != row["terminal_reward"]
                    or bool(bad.any()) != row["body_failed"]
                ):
                    raise ValueError("raw physical reward/outcome differs")
                value = z["NN_value"].astype(np.float32)
                reward_t = np.zeros(n, np.float32)
                reward_t[-1] = reward * 0.995 ** (controls - n)
                advantage = np.zeros(n, np.float32)
                carry = nv = 0.0
                for t in range(n - 1, -1, -1):
                    carry = float(reward_t[t]) + 0.995 * nv - float(value[t]) + 0.995 * 0.95 * carry
                    advantage[t] = carry
                    nv = float(value[t])
                if not np.array_equal(advantage, z["NN_advantage"]) or not np.array_equal(
                    (advantage + value).astype(np.float32), z["NN_returns"]
                ):
                    raise ValueError("independent finite-option GAE/returns differ")
                results.append(
                    dict(folder=folder, case=index, neural_rows=n, native_steps=controls * 10)
                )
            print(
                json.dumps(dict(folder=folder, audited_native_steps=native, neural_rows=neural)),
                flush=True,
            )
        qualification_steps = json.loads((a.root / "environment-qualified.json").read_text())[
            "native_steps"
        ]
        if (
            native + qualification_steps != r["native_steps"]
            or rollout_actor != r["counts"]["rollout_actor_rows"]
            or rollout_critic != r["counts"]["rollout_critic_rows"]
        ):
            raise ValueError("complete actual native/rollout neural accounting differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_RECURRENT_ALL_RAW_NATIVE_LIKELIHOOD_REWARD_GAE_AUDIT_PASSED",
                input_file_hashes=pins,
                results=results,
                native_steps=native,
                new_actor_rows=neural,
                new_critic_rows=critic_rows,
                optimizer_updates=0,
                optimizer_gradient_replay=False,
                strict_native_all_substeps=True,
                goal_complete=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                native_steps=native,
                new_actor_rows=neural,
                new_critic_rows=critic_rows,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
