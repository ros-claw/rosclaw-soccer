"""Audit all CEM raw outcomes/selection; closed-loop repeat selected and baselines.

Nonselected search trajectories receive raw primitive/outcome audits, NOT new
physics replay. Selected TRAIN and both DEV arms plus old baselines are fully
recomputed through neural inference, Cartesian IK and actual native physics.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_search_paid_foot_contact import contact_rank

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hash", required=True)
    p.add_argument("--producer-source", type=Path, required=True)
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
        raise ValueError("unused output/exact closed search/frozen auditor required")
    r = json.loads(receipt.read_text())
    d = json.loads((a.root / "declaration.json").read_text())
    if (
        r["status"] != "PAID_CARTESIAN_CONTACT_TEACHER_RESEARCH_NOT_PROMOTED"
        or subprocess.check_output(
            ["git", "-C", str(a.producer_source), "rev-parse", "HEAD"], text=True
        ).strip()
        != d["source_head"]
    ):
        raise ValueError("actual original producer checkout required")
    pins = {
        **r["input_file_hashes"],
        str(receipt): a.hash,
        str(a.root / "declaration.json"): digest(a.root / "declaration.json"),
    }
    for folder in ("src", "scripts"):
        for old in (a.producer_source / folder).rglob("*.py"):
            new = Path.cwd() / old.relative_to(a.producer_source)
            if digest(old) != digest(new):
                raise ValueError("audit cannot change existing actor/IK/physics helpers")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            all_search_raw_outcome_audits=True,
            selected_and_baselines_closed_loop_repeat=True,
            nonselected_new_physics_replay=False,
            no_gradient_updates=True,
            hardware_authorized=False,
            commercial_publicity_authorized=False,
        ),
    )
    bench = None
    try:
        import sys

        import torch

        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent, phase_intent
        from rosclaw_soccer.rsi.paid_research_bench import PaidResearchBench
        from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH, PaidTrainableActor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policies = [Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx")]
        if len(policies) != 1:
            raise ValueError("one actual pinned external actor required")
        ref = policies[0].parent.parent
        sys.path.insert(0, str(ref))
        from exp.mujoco_soccer.constants import BODY_NAMES, JOINT_EFFORT_LIMIT_ARRAY
        from exp.mujoco_soccer.math_utils import quat_apply
        from exp.mujoco_soccer.motion import load_motion_clips
        from exp.mujoco_soccer.observations import build_observation
        from exp.mujoco_soccer.scene import MujocoSoccer

        names = list(dict.fromkeys(x["source"] for x in d["train_cases"]))
        xmls = [Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml")]
        if not xmls or len({digest(f) for f in xmls}) != 1:
            raise ValueError("one identical standard physics recipe required")
        contract = PaidResearchPolicy(policies[0], expected_hash=PUBLISHED_HASH, horizon=1)
        actor = PaidTrainableActor(policies[0], expected_hash=PUBLISHED_HASH).eval()
        clips = {n: load_motion_clips(ref / "motions/soccer-standard" / n)[0] for n in names}
        bench = PaidResearchBench(
            xml=xmls[0],
            contract=contract,
            clips=clips,
            scene_factory=MujocoSoccer,
            build_observation=build_observation,
            quat_apply=quat_apply,
            effort=JOINT_EFFORT_LIMIT_ARRAY,
            torso_index=BODY_NAMES.index("torso_link"),
        )
        audited = {}

        def audit_raw(row, case, latent, repeat=False):
            path = a.root / row["raw_path"]
            if (
                path.resolve().parent.is_relative_to(a.root.resolve()) is False
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("exact confined original raw required")
            pins[str(path)] = row["raw_hash"]
            with np.load(path, allow_pickle=False) as z:
                z = {k: z[k].copy() for k in z.files}
            controls, n = len(z["target"]), len(z["NN_action"])
            if any(not np.isfinite(x).all() for x in z.values()):
                raise ValueError("all raw physical/neural/planner values must be finite")
            if n != min(controls, clips[case["source"]].length):
                raise ValueError("actual source reference horizon differs")
            if (
                row["source"] != case["source"]
                or row["native_steps"] != controls * 10
                or row["neural_rows"] != n
                or not np.array_equal(z["neural_control"], np.arange(controls) < n)
                or not 0 <= controls - n <= 60
            ):
                raise ValueError("actual case/control/tail identity differs")
            if (
                z["NN_h_before"][0].any()
                or z["NN_c_before"][0].any()
                or not np.array_equal(z["NN_h_before"][1:], z["NN_h_after"][:-1])
                or not np.array_equal(z["NN_c_before"][1:], z["NN_c_after"][:-1])
                or z["NN_obs"][0, 125:154].any()
                or not np.array_equal(z["NN_obs"][1:, 125:154], z["NN_action"][:-1])
            ):
                raise ValueError("per-episode recurrent/executed action history differs")
            expected = np.stack(
                [phase_intent(t, d["centers"][case["source"]], latent) for t in range(n)]
            )
            if (
                not np.array_equal(expected, z["PLAN_pose_offset"])
                or np.abs(z["PLAN_joint_delta_HG"]).max() > 0.2 + 1e-12
            ):
                raise ValueError("actual declared Cartesian context/bounded IK intent differs")
            target = (
                contract.parameters["default_joint_pos"][contract.hg_from_policy]
                + z["NN_action"][:, contract.hg_from_policy]
                * contract.parameters["action_scale"][contract.hg_from_policy]
            )
            if not np.array_equal(target, z["target"][:n]) or (
                controls > n
                and not np.array_equal(
                    z["target"][n:], np.broadcast_to(target[-1], z["target"][n:].shape)
                )
            ):
                raise ValueError("actual composite commands/held targets differ")
            q, v, f = [
                z[k].reshape(-1, width) for k, width in (("qpos", 43), ("qvel", 41), ("forces", 3))
            ]
            goal = z["initial_mocap_pos"][bench.template.goal_mocap_id]
            post = native_contact_window(
                force=f[:, :2],
                canonical_qpos=q,
                canonical_qvel=v,
                goal=goal,
                intended_foot=0 if "left" in case["source"] else 1,
            )
            bad = (
                (z["qpos"][:, :, 2] < 0.65)
                | (1 - 2 * (z["qpos"][:, :, 4] ** 2 + z["qpos"][:, :, 5] ** 2) < 0.9)
            ).any(axis=1)
            if not bad.any() and controls != clips[case["source"]].length + 60:
                raise ValueError("nonterminal source/tail trajectory was censored")
            forbidden = bool((f[:, 2] > 1).any())
            impulse = float((f[:, 2] * 0.002).sum())
            reward = -12.0 if bad.any() else -8 - 6 * forbidden
            if (
                not bad.any()
                and post["correct_foot_first"]
                and post["post_contact_quality_observed"]
            ):
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
                post != row["postcontact"]
                or not np.array_equal(bad, z["body_failed"])
                or bool(bad.any()) != row["body_failed"]
                or forbidden != row["forbidden"]
                or reward != row["terminal_reward"]
                or impulse != row["nonfoot_impulse_Ns"]
                or (bad.any() and not np.array_equal(np.flatnonzero(bad), [controls - 1]))
            ):
                raise ValueError("primitive body/contact/reward/outcome differs")
            audited[row["raw_path"]] = controls * 10
            if repeat:
                intent = FootTaskSpaceIntent(
                    bench.template.model,
                    foot="left" if "left" in case["source"] else "right",
                    center=d["centers"][case["source"]],
                    latent=latent,
                )
                repeated, summary = bench.collect(case, actor, transform=intent)
                if set(repeated) != set(z) or any(not np.array_equal(repeated[k], z[k]) for k in z):
                    raise ValueError(
                        "full independent serial actor/IK/native repeat differs: " + str(path)
                    )
                if any(summary[k] != row[k] for k in summary):
                    raise ValueError("independent full closed-loop summary differs")

        # All32 paired baseline executions are repeated with zero foot intent.
        for row, case in zip(r["baseline"], d["train_cases"], strict=True):
            audit_raw(row, case, np.zeros((2, 6)), True)
        for row, case in zip(r["baseline_dev"], d["dev_cases"], strict=True):
            audit_raw(row, case, np.zeros((2, 6)), True)
        for source_index, name in enumerate(names):
            rng = np.random.default_rng(20261021 + source_index)
            pair = [x for x in d["train_cases"] if x["source"] == name]
            best_z, mean, std = np.zeros((2, 6)), np.zeros((2, 6)), np.full((2, 6), 0.5)
            best_rows = [row for row in r["baseline"] if row["source"] == name]
            rank = contact_rank(best_rows)
            for generation in range(d["generations"]):
                file = a.root / f"source{source_index}/generation{generation}/complete.json"
                pins[str(file)] = digest(file)
                record = json.loads(file.read_text())
                zs = [best_z.copy()] + list(rng.normal(mean, std, size=(11, 2, 6)))
                if len(record["candidates"]) != 12:
                    raise ValueError("all predeclared candidates required")
                ranks = []
                for candidate, z in zip(record["candidates"], zs, strict=True):
                    latent = np.tanh(z)
                    if not np.array_equal(np.array(candidate["z"]), z) or not np.array_equal(
                        np.array(candidate["latent"]), latent
                    ):
                        raise ValueError("actual source-seeded bounded CEM proposal differs")
                    for row, case in zip(candidate["results"], pair, strict=True):
                        audit_raw(row, case, latent)
                    score = contact_rank(candidate["results"])
                    ranks.append(score)
                    if list(score) != candidate["rank"]:
                        raise ValueError("raw primitive selection rank differs")
                    if score > rank:
                        best_z, best_rows, rank = z.copy(), candidate["results"], score
                elite = np.stack(
                    [zs[i] for i in sorted(range(12), key=lambda i: ranks[i], reverse=True)[:4]]
                )
                mean = 0.5 * mean + 0.5 * elite.mean(axis=0)
                std = np.clip(0.5 * std + 0.5 * elite.std(axis=0), 0.15, 1.2)
                if (
                    list(rank) != record["best_rank"]
                    or not np.array_equal(mean, np.array(record["next_z_mean"]))
                    or not np.array_equal(std, np.array(record["next_z_std"]))
                ):
                    raise ValueError("CEM distribution update/elite retention differs")
            selected = r["selected"][source_index]
            if (
                selected["source"] != name
                or not np.array_equal(np.array(selected["selected_z"]), best_z)
                or not np.array_equal(np.array(selected["selected_latent"]), np.tanh(best_z))
                or selected["results"] != best_rows
                or selected["rank"] != list(rank)
            ):
                raise ValueError("selected teacher not bound to raw source/CEM intent")
            for row, case in zip(best_rows, pair, strict=True):
                audit_raw(row, case, np.tanh(best_z), True)
            print(
                json.dumps(
                    dict(
                        source=name,
                        rank=list(rank),
                        audited_raws=len(audited),
                        new_native_steps=bench.native_steps,
                    )
                ),
                flush=True,
            )
        latent_by_source = {x["source"]: np.array(x["selected_latent"]) for x in r["selected"]}
        for row, case in zip(r["final_dev"], d["dev_cases"], strict=True):
            audit_raw(row, case, latent_by_source[case["source"]], True)
        if len(audited) != d["max_episodes"] or sum(audited.values()) != r["native_steps"]:
            raise ValueError("complete declared physical execution accounting differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CARTESIAN_RAW_SELECTION_AND_CLOSED_LOOP_AUDIT_PASSED",
                input_file_hashes=pins,
                all_search_raws=len(audited),
                native_steps=bench.native_steps,
                new_neural_rows=bench.neural_rows,
                neural_optimizer_updates=0,
                all_selected_and_baseline_closedloop_exact=True,
                nonselected_new_physics_replay=False,
                raw_selection_and_CEM_updates_reconstructed=True,
                original_receipt_did_not_seal_nonselected_generation_artifacts=True,
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
                native_steps=0 if bench is None else bench.native_steps,
                new_neural_rows=0 if bench is None else bench.neural_rows,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
