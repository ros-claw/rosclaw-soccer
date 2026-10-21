"""Factorial h/c and cached-reference carry ablation on actual live parents.

Loads explicit MuJoCo integration checkpoints from completed per-parent
artifacts. This is a checkpoint-branch experiment, not fresh full matches.
Unmodified zero/zero arms must match all prior continuous-body baseline arrays.
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-root", type=Path, required=True)
    p.add_argument("--parent-declaration-hash", required=True)
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    dp, tp = a.parent_root / "declaration.json", a.training_root / "complete.json"
    if (
        a.output.exists()
        or digest(dp) != a.parent_declaration_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("pinned source and completed parent bank, unused output required")
    d = json.loads(dp.read_text())
    t = json.loads(tp.read_text())
    cases = d["TRAIN_cases"]
    if len(cases) != 16 or not d["parents_from_actual_first_kick_and_explicit_gravity_drop"]:
        raise ValueError("all sixteen real post-kick parents required")
    if d["input_file_hashes"].get(str(tp)) != digest(tp):
        raise ValueError("same previously qualified learned v0 required")
    pins = {**d["input_file_hashes"], str(dp): a.parent_declaration_hash}
    for i in range(16):
        for name in ("complete.json", "parent.npz", "first.npz", "drop.npz"):
            path = a.parent_root / f"parent{i}" / name
            pins[str(path)] = digest(path)
        path = a.parent_root / f"parent{i}/hold.npz"
        if path.exists():
            pins[str(path)] = digest(path)
        path = a.parent_root / f"baseline/case{i}.npz"
        pins[str(path)] = digest(path)
    bp = a.parent_root / "baseline/complete.json"
    pins[str(bp)] = digest(bp)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    arms = [dict(carry_h_c=h, carry_reference=r) for h in (False, True) for r in (False, True)]
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            cases=cases,
            arms=arms,
            first_kicks_borrowed_not_recounted=True,
            actual_branch_repeats=2,
            parent_artifacts_completed_but_overall_CEM_may_still_be_running=True,
            parent_checkpoints_pinned_before_use=True,
            motion_reference_still_required=True,
            body_state_reset_to_motion_initial=False,
            simulation_checkpoint_branches=True,
            actual_previous_action_always_carried=True,
            transition_blending=False,
            IK_executed=False,
            weight_updates=0,
            fresh_exam=False,
            team_game=False,
            hardware_authorized=False,
            promotion_authorized=False,
            commercial_publicity_authorized=False,
        ),
    )
    bench = None
    try:
        from concurrent.futures import ThreadPoolExecutor

        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_distilled_actor import load_distilled_actor
        from rosclaw_soccer.rsi.paid_native_episode import PaidNativeEpisode
        from rosclaw_soccer.rsi.paid_research_bench import PaidResearchBench
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
        from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        xml = next(Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml"))
        contract = PaidResearchPolicy(policy, expected_hash=PUBLISHED_HASH, horizon=1)
        actor, _ = load_distilled_actor(
            policy,
            checkpoint=a.training_root / "distilled-epoch256.pt",
            checkpoint_hash=t["final_checkpoint_hash"],
            training_receipt=tp,
            training_receipt_hash=digest(tp),
            scene_hash=digest(xml),
        )
        sys.path.insert(0, str(policy.parent.parent))
        from exp.mujoco_soccer.constants import BODY_NAMES, JOINT_EFFORT_LIMIT_ARRAY
        from exp.mujoco_soccer.math_utils import quat_apply
        from exp.mujoco_soccer.motion import load_motion_clips
        from exp.mujoco_soccer.observations import build_observation
        from exp.mujoco_soccer.scene import MujocoSoccer

        names = list(dict.fromkeys(c["source"] for c in cases))
        clips = {
            n: load_motion_clips(policy.parent.parent / "motions/soccer-standard" / n)[0]
            for n in names
        }
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
        parents = []
        for i, case in enumerate(cases):
            folder = a.parent_root / f"parent{i}"
            receipt = json.loads((folder / "complete.json").read_text())
            if (
                receipt["first"]["raw_hash"] != digest(folder / "first.npz")
                or receipt["first"]["body_failed"]
                or receipt["first"]["source"] != case["source"]
            ):
                raise ValueError("qualified actual first-kick proof required")
            with np.load(folder / "parent.npz", allow_pickle=False) as z:
                parent = {k: z[k].copy() for k in z.files}
            with np.load(folder / "first.npz", allow_pickle=False) as z:
                memory = tuple(
                    torch.from_numpy(z["NN_" + k + "_after"][-1].copy()) for k in ("h", "c")
                )
                if not np.array_equal(z["NN_action"][-1], parent["previous_actual_action"]):
                    raise ValueError("parent previous action is not the actually executed action")
            parents.append((parent, memory))

        def restore(i, arm):
            parent, memory = parents[i]
            env = copy.copy(bench.template)
            env.data = mujoco.MjData(env.model)
            mujoco.mj_setState(
                env.model,
                env.data,
                parent["full_integration_state"],
                mujoco.mjtState.mjSTATE_INTEGRATION,
            )
            mujoco.mj_forward(env.model, env.data)
            actual = snapshot(env.data)
            if any(not np.array_equal(actual[k], parent[k]) for k in actual):
                raise ValueError("native integration-checkpoint restore is not bit-exact")
            ep = PaidNativeEpisode(
                env,
                clips[cases[i]["source"]],
                contract.parameters,
                contract.hg_from_policy,
                JOINT_EFFORT_LIMIT_ARRAY,
                build_observation,
                BODY_NAMES.index("torso_link"),
            )
            ep.last_action = parent["previous_actual_action"].copy()
            ep.ref_pos, ep.ref_vel = ep.clip.joint_pos[-1].copy(), ep.clip.joint_vel[-1].copy()
            ep.ref_ang = ep.clip.body_ang_vel_w[-1, BODY_NAMES.index("torso_link")].copy()
            ep.warning = tuple(int(x.number) for x in env.data.warning)
            ep.termination_reason = "REFERENCE_COMPLETE"
            goal = env.data.mocap_pos[env.goal_mocap_id].copy()
            ep.begin_current_state(ep.clip, goal=goal, preserve_reference=arm["carry_reference"])
            if any(not np.array_equal(actual[k], v) for k, v in snapshot(env.data).items()):
                raise ValueError("reference/memory carry may not modify the physical parent")
            with bench.lock:
                bench.counters.append(ep.work)
            return ep, memory if arm["carry_h_c"] else None

        # Admit every exact restore before any branch advances physics.
        for i in range(16):
            restore(i, arms[0])

        def execute(i, ai, repeat):
            ep, memory = restore(i, arms[ai])
            raw, row = bench.collect_current(ep, actor, initial_state=memory)
            if not np.array_equal(raw["NN_action"], raw["NN_nominal_action"]):
                raise ValueError("all ablation arms require pure neural output, no mixing")
            if ai == 0:
                with np.load(a.parent_root / f"baseline/case{i}.npz", allow_pickle=False) as z:
                    if any(not np.array_equal(raw[k], z[k]) for k in raw):
                        raise ValueError(
                            "restored branch failed full continuous baseline re-admission"
                        )
            path = a.output / f"arm{ai}-case{i}-repeat{repeat}.npz"
            np.savez_compressed(path, **raw)
            row.update(
                case=i,
                arm=ai,
                repeat=repeat,
                raw_path=path.name,
                raw_hash=digest(path),
                research_contact_quality=contact_rank([row, row])[1] > 0,
            )
            return raw, row

        def paired(item):
            i, ai = item
            first, row = execute(i, ai, 0)
            second, other = execute(i, ai, 1)
            if set(first) != set(second) or any(
                not np.array_equal(first[k], second[k]) for k in first
            ):
                raise ValueError("whole neural and native branch repeats differ")
            row["independent_repeat_raw_hash"] = other["raw_hash"]
            print(
                json.dumps(
                    dict(
                        case=i,
                        arm=ai,
                        quality=row["research_contact_quality"],
                        body_failed=row["body_failed"],
                    )
                ),
                flush=True,
            )
            return row

        with ThreadPoolExecutor(max_workers=2) as pool:
            # Complete zero/zero regression before evaluating carry alternatives.
            baseline = list(pool.map(paired, [(i, 0) for i in range(16)]))
            alternatives = list(pool.map(paired, [(i, ai) for ai in (1, 2, 3) for i in range(16)]))
        results = baseline + alternatives
        check_pins(pins)
        summaries = [
            dict(
                arm=arm,
                quality=sum(x["research_contact_quality"] for x in results if x["arm"] == ai),
                stable=sum(not x["body_failed"] for x in results if x["arm"] == ai),
                forbidden=sum(x["forbidden"] for x in results if x["arm"] == ai),
            )
            for ai, arm in enumerate(arms)
        ]
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_LIVE_MEMORY_FACTORIAL_EXAM_CLOSED",
                input_file_hashes=pins,
                results=results,
                summaries=summaries,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                optimizer_updates=0,
                all_baselines_reproduced=True,
                all_branch_repeats_exact=True,
                parent_first_kicks_recounted_as_new_work=False,
                team_game=False,
                fresh_exam=False,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                actual_native_steps=0 if bench is None else bench.native_steps,
                actual_neural_rows=0 if bench is None else bench.neural_rows,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
