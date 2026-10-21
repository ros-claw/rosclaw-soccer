"""Bounded coherent Cartesian-foot teacher CEM, not another PPO round.

Each source shares one intent across two TRAIN ball conditions; DEV is never
searched. No collider/gain changes or best-highlight-only reporting. Research
CC-BY-NC foundation remains external and grants no publicity authorization.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def contact_rank(rows: list[dict[str, Any]]) -> tuple[int, int, int, float]:
    if len(rows) != 2:
        raise ValueError("two complete declared TRAIN conditions required")
    feasible = all(
        not r["body_failed"]
        and r["postcontact"]["correct_foot_first"]
        and r["postcontact"]["post_contact_quality_observed"]
        for r in rows
    )
    clean = sum(
        not r["body_failed"]
        and not r["forbidden"]
        and r["postcontact"]["correct_foot_first"]
        and r["postcontact"]["post_contact_quality_observed"]
        for r in rows
    )
    good = sum(
        not r["body_failed"]
        and not r["forbidden"]
        and r["postcontact"]["correct_foot_first"]
        and r["postcontact"]["post_contact_quality_observed"]
        and r["postcontact"]["goal_progress_1p2s_m"] >= 1
        and abs(r["postcontact"]["goal_lateral_1p2s_m"]) <= 0.75
        and not r["postcontact"]["other_foot_contact_in_window"]
        and r["postcontact"]["maximum_pelvis_retreat_after_contact_m"] <= 0.05
        for r in rows
    )
    return (int(feasible), good, clean, sum(r["terminal_reward"] for r in rows))


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--teacher-hash", required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--course-root", type=Path, required=True)
    p.add_argument("--course-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 4, 8), default=8)
    p.add_argument("--generations", type=int, choices=(2, 3), default=3)
    a = p.parse_args()
    teacher_file, course_file = a.teacher_root / "complete.json", a.course_root / "declaration.json"
    if (
        a.output.exists()
        or digest(teacher_file) != a.teacher_hash
        or digest(course_file) != a.course_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact frozen research inputs/code required")
    teacher, course = json.loads(teacher_file.read_text()), json.loads(course_file.read_text())
    if (
        teacher["status"] != "PAID_RESEARCH_NATIVE_TORCH_PROBE_NOT_PROMOTED"
        or not teacher["strict_replays_exact"]
        or not teacher["whole_repeats_exact"]
    ):
        raise ValueError("completed physical standard-ball teacher required")
    cases = [
        r for r in teacher["results"] if r["profile"] == "cached-standard" and r["repeat"] == 0
    ]
    names = list(dict.fromkeys(r["source"] for r in cases))
    dev = course["dev_cases"]
    if len(cases) != 16 or len(names) != 8 or len(dev) != 16:
        raise ValueError("all16 TRAIN and16 disjoint DEV source-balanced cases required")
    centers = {
        name: sum(
            r["postcontact"]["first_foot_contact_substep"] // 10
            for r in cases
            if r["source"] == name
        )
        // 2
        for name in names
    }
    pins = {
        **teacher["input_file_hashes"],
        str(teacher_file): a.teacher_hash,
        str(course_file): a.course_hash,
    }
    for relative, h in teacher["raw_hashes"].items():
        pins[str(a.teacher_root / relative)] = h
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            train_cases=cases,
            dev_cases=dev,
            generations=a.generations,
            population=12,
            elite_count=4,
            workers=a.workers,
            intent_dimensions=[2, 6],
            pose_scale=[0.04, 0.035, 0.04, 0.15, 0.2, 0.15],
            centers=centers,
            phase_centers_relative=[-12, 8],
            bump_width_controls=30,
            ik_iterations=4,
            ik_damping=0.02,
            rotation_weight_m_per_rad=0.15,
            max_joint_increment_rad=0.08,
            max_total_joint_delta_rad=0.2,
            proposal=(
                "tanh normal(z_mean,z_std), seed20261021+source_index; slot0 retains currentbest"
            ),
            common_intent_per_source_both_TRAIN_conditions=True,
            DEV_not_searched=True,
            quality_profile=(
                "research-only:body stable,footfirst,fullwindow,no nonfoot>1N,"
                "progress>=1m,abs lateral<=.75m,otherfoot false,retreat<=.05m"
            ),
            rank_order=[
                "both TRAIN cases body/foot/window feasible",
                "quality_count",
                "clean_count",
                "sum existing task reward",
            ],
            max_episodes=48 + 8 * a.generations * 12 * 2,
            max_native_steps=(48 + 8 * a.generations * 12 * 2) * 3600,
            shared_model_per_episode_data=True,
            no_neural_optimizer_updates=True,
            cem_distribution_updates_max=8 * a.generations,
            original_formal52_unchanged=True,
            commercial_publicity_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
            method_sources=[
                "https://mujoco.readthedocs.io/en/stable/programming/simulation.html",
                "https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac",
            ],
        ),
    )
    bench = None
    cem_updates = 0
    try:
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
        from rosclaw_soccer.rsi.paid_research_bench import PaidResearchBench
        from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH, PaidTrainableActor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(a.reference_root))
        from exp.mujoco_soccer.constants import BODY_NAMES, JOINT_EFFORT_LIMIT_ARRAY
        from exp.mujoco_soccer.math_utils import quat_apply
        from exp.mujoco_soccer.motion import load_motion_clips
        from exp.mujoco_soccer.observations import build_observation
        from exp.mujoco_soccer.scene import MujocoSoccer

        policy_path = a.reference_root / "ckp/policy_30000.onnx"
        if pins.get(str(policy_path)) != PUBLISHED_HASH:
            raise ValueError("exact external original research actor required")
        contract = PaidResearchPolicy(policy_path, expected_hash=PUBLISHED_HASH, horizon=1)
        actor = PaidTrainableActor(policy_path, expected_hash=PUBLISHED_HASH).eval()
        original_weights = {k: v.clone() for k, v in actor.state_dict().items()}
        clips = {
            n: load_motion_clips(a.reference_root / "motions/soccer-standard" / n)[0] for n in names
        }
        xml = a.teacher_root / "cached-standard/declared-scene.xml"
        if str(xml) not in pins:
            raise ValueError("qualified identical standard scene required")
        bench = PaidResearchBench(
            xml=xml,
            contract=contract,
            clips=clips,
            scene_factory=MujocoSoccer,
            build_observation=build_observation,
            quat_apply=quat_apply,
            effort=JOINT_EFFORT_LIMIT_ARRAY,
            torso_index=BODY_NAMES.index("torso_link"),
        )
        model = bench.template.model
        critical = {
            k: getattr(model, k).copy()
            for k in (
                "body_mass",
                "body_inertia",
                "body_pos",
                "body_quat",
                "geom_size",
                "geom_pos",
                "geom_quat",
                "geom_friction",
                "geom_contype",
                "geom_conaffinity",
                "geom_solref",
                "geom_solimp",
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_gear",
                "dof_armature",
                "dof_damping",
                "dof_frictionloss",
                "jnt_range",
            )
        }
        for case in cases + dev:
            bench.make(case)

        def run(case, latent, path):
            intent = FootTaskSpaceIntent(
                model,
                foot="left" if "left" in case["source"] else "right",
                center=centers[case["source"]],
                latent=latent,
            )
            raw, row = bench.collect(case, actor, transform=intent)
            np.savez_compressed(path, **raw)
            row.update(raw_hash=digest(path), raw_path=str(path.relative_to(a.output)))
            return raw, row

        # Zero intent must reproduce ALL16 prior complete trajectories exactly,
        # not merely similar metrics, before a nonzero Cartesian intent runs.
        baseline = []
        folder = a.output / "baseline-old16"
        folder.mkdir()
        for i, case in enumerate(cases):
            raw, row = run(case, np.zeros((2, 6)), folder / f"case{i}.npz")
            with np.load(
                a.teacher_root / f"cached-standard/case{case['case']}-repeat0.npz",
                allow_pickle=False,
            ) as old:
                for k in (
                    "qpos",
                    "qvel",
                    "forces",
                    "torque",
                    "target",
                    "native_qpos_before",
                    "native_qvel_before",
                    "neural_control",
                ):
                    if not np.array_equal(raw[k], old[k]):
                        raise ValueError(
                            "shared-model/zero-intent full native regression differs: " + k
                        )
                for new, prior in (
                    ("NN_obs", "NN_observation"),
                    ("NN_action", "NN_action"),
                    ("NN_h_before", "NN_h_before"),
                    ("NN_c_before", "NN_c_before"),
                    ("NN_h_after", "NN_h_after"),
                    ("NN_c_after", "NN_c_after"),
                ):
                    if not np.array_equal(raw[new], old[prior]):
                        raise ValueError("zero-intent causal LSTM history differs: " + new)
            baseline.append(row)
        write_once(
            folder / "complete.json", dict(results=baseline, all16_full_native_and_NN_exact=True)
        )
        baseline_dev = []
        folder = a.output / "baseline-dev16"
        folder.mkdir()
        for i, case in enumerate(dev):
            baseline_dev.append(run(case, np.zeros((2, 6)), folder / f"case{i}.npz")[1])
        write_once(folder / "complete.json", dict(results=baseline_dev))

        def search(source_index):
            nonlocal cem_updates
            name = names[source_index]
            rng = np.random.default_rng(20261021 + source_index)
            pair = [case for case in cases if case["source"] == name]
            best_z, mean, std = np.zeros((2, 6)), np.zeros((2, 6)), np.full((2, 6), 0.5)
            best_rows = [row for row in baseline if row["source"] == name]
            best_rank = contact_rank(best_rows)
            folder = a.output / f"source{source_index}"
            folder.mkdir()
            for generation in range(a.generations):
                generation_folder = folder / f"generation{generation}"
                generation_folder.mkdir()
                zs = [best_z.copy()] + list(rng.normal(mean, std, size=(11, 2, 6)))
                candidates = []
                for ci, z in enumerate(zs):
                    latent = np.tanh(z)
                    rows = [
                        run(case, latent, generation_folder / f"candidate{ci}-case{k}.npz")[1]
                        for k, case in enumerate(pair)
                    ]
                    rank = contact_rank(rows)
                    candidates.append(
                        dict(z=z.tolist(), latent=latent.tolist(), rank=list(rank), results=rows)
                    )
                    if rank > best_rank:
                        best_z, best_rows, best_rank = z.copy(), rows, rank
                ordered = sorted(
                    range(12), key=lambda i: tuple(candidates[i]["rank"]), reverse=True
                )
                elite = np.stack([zs[i] for i in ordered[:4]])
                mean = 0.5 * mean + 0.5 * elite.mean(axis=0)
                std = np.clip(0.5 * std + 0.5 * elite.std(axis=0), 0.15, 1.2)
                write_once(
                    generation_folder / "complete.json",
                    dict(
                        candidates=candidates,
                        best_rank=list(best_rank),
                        next_z_mean=mean.tolist(),
                        next_z_std=std.tolist(),
                    ),
                )
                with bench.lock:
                    cem_updates += 1
                print(
                    json.dumps(dict(source=name, generation=generation, best_rank=list(best_rank))),
                    flush=True,
                )
            selected = dict(
                source=name,
                selected_z=best_z.tolist(),
                selected_latent=np.tanh(best_z).tolist(),
                rank=list(best_rank),
                results=best_rows,
            )
            write_once(folder / "selected.json", selected)
            return selected

        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            selected = list(pool.map(search, range(8)))
        by_source = {row["source"]: np.array(row["selected_latent"]) for row in selected}
        final_dev = []
        folder = a.output / "final-dev16"
        folder.mkdir()
        for i, case in enumerate(dev):
            final_dev.append(run(case, by_source[case["source"]], folder / f"case{i}.npz")[1])
        write_once(folder / "complete.json", dict(results=final_dev))
        if any(not np.array_equal(getattr(model, k), v) for k, v in critical.items()) or any(
            not torch.equal(actor.state_dict()[k], v) for k, v in original_weights.items()
        ):
            raise ValueError("physics constants or frozen full actor changed")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CARTESIAN_CONTACT_TEACHER_RESEARCH_NOT_PROMOTED",
                input_file_hashes=pins,
                baseline=baseline,
                baseline_dev=baseline_dev,
                selected=selected,
                final_dev=final_dev,
                native_steps=bench.native_steps,
                new_neural_rows=bench.neural_rows,
                optimizer_updates=0,
                optimizer_counter_scope="neural_gradients_only",
                cem_distribution_updates_committed=cem_updates,
                all16_zero_intent_full_native_NN_exact=True,
                all_declared_sources_and_cases_retained=True,
                final_rollouts_not_yet_independently_replayed=True,
                goal_complete=False,
                commercial_publicity_authorized=False,
                promotion_authorized=False,
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
                optimizer_updates=0,
                cem_distribution_updates_committed=cem_updates,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
