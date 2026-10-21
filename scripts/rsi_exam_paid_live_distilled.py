"""CPU neural-only old/DEV and full two-kick world exams of all four models.

By default the candidate controls BOTH kicks. A separate, explicit contextual
mode preserves the qualified initial-skill actor and uses the learned actor
only for live-successor skills. This is not a case-ID winner lookup. Neither
mode resets the body at handoff. Ball returns are practice, not a team game.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def select_entry_actor(mode: str, candidate: Any, prior: Any) -> Any:
    """Declared skill-context routing; no case/foot/goal/outcome is an input."""
    if mode == "candidate":
        return candidate
    if mode == "qualified-prior":
        return prior
    raise ValueError("unknown initial motor context")


def successor_memory(raw, mode, *, same_actor):
    """Private measured neural memory, never a translation between two experts."""
    if mode == "reset":
        return None
    if mode != "carry" or not same_actor:
        raise ValueError("trained carry requires the same actor on both skills")
    import numpy as np
    import torch

    arrays = [raw[k][-1] for k in ("NN_h_after", "NN_c_after")]
    if any(
        x.shape != (2, 1, 128) or x.dtype != np.float32 or not np.isfinite(x).all() for x in arrays
    ):
        raise ValueError("actual complete finite per-player recurrent memory required")
    return tuple(torch.from_numpy(x.copy()) for x in arrays)


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior-exam-root", type=Path, required=True)
    p.add_argument("--prior-exam-hash", required=True)
    p.add_argument("--live-teacher-root", type=Path, required=True)
    p.add_argument("--live-teacher-hash", required=True)
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--candidate", nargs=2, action="append", required=True)
    p.add_argument(
        "--entry-controller", choices=("candidate", "qualified-prior"), default="candidate"
    )
    p.add_argument("--candidate-index", type=int, choices=range(4))
    p.add_argument("--sequence-bank-root", type=Path)
    p.add_argument("--sequence-bank-hash")
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    xp, tp, bp = [
        root / "complete.json" for root in (a.prior_exam_root, a.live_teacher_root, a.base_root)
    ]
    if (
        a.output.exists()
        or len(a.candidate) != 4
        or digest(xp) != a.prior_exam_hash
        or digest(tp) != a.live_teacher_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError(
            "four candidates, completed baselines and exact frozen new source required"
        )
    prior, teacher, base = [json.loads(f.read_text()) for f in (xp, tp, bp)]
    declaration = json.loads((a.prior_exam_root / "declaration.json").read_text())
    sequences = a.sequence_bank_root is not None
    if sequences != (a.sequence_bank_hash is not None) or (
        sequences and a.entry_controller != "candidate"
    ):
        raise ValueError("paired sequence bank identity and one actor for both skills required")
    prior_results = (
        prior["candidates"][0]["isolated"] if sequences else prior["candidates"][0]["results"]
    )
    pins = {
        **teacher["input_file_hashes"],
        str(tp): a.live_teacher_hash,
        str(xp): a.prior_exam_hash,
        str(bp): digest(bp),
        str(a.prior_exam_root / "declaration.json"): digest(a.prior_exam_root / "declaration.json"),
    }
    if sequences:
        sp = a.sequence_bank_root / "complete.json"
        if digest(sp) != a.sequence_bank_hash:
            raise ValueError("exact completed complete-world training bank required")
        bank = json.loads(sp.read_text())
        if (
            bank["bootstrap_exam_root"] != str(a.prior_exam_root)
            or bank["base_root"] != str(a.base_root)
            or bank["base_checkpoint_hash"] != base["final_checkpoint_hash"]
            or prior["entry_controller"] != "candidate"
        ):
            raise ValueError("sequence initialization must match the qualified new baseline")
        pins.update(bank["input_file_hashes"])
        pins[str(sp)] = a.sequence_bank_hash
        for world in prior["candidates"][0]["continuous"]["results"]:
            for key in ("first", "second"):
                row = world[key]
                pins[str(a.prior_exam_root / row["raw_path"])] = row["raw_hash"]
    for key in ("old16", "dev16"):
        for row in prior_results[key]["results"]:
            pins[str(a.prior_exam_root / row["raw_path"])] = row["raw_hash"]
    for row in teacher["baseline"]:
        pins[str(a.live_teacher_root / row["raw_path"])] = row["raw_hash"]
    candidates = []
    for index, (root, h) in enumerate(a.candidate):
        root = Path(root)
        rp = root / "complete.json"
        if digest(rp) != h:
            raise ValueError("final neural training receipt hash differs")
        r = json.loads(rp.read_text())
        if r["config"]["variant"] != index:
            raise ValueError("all three predeclared seeds and rehearsal ablation required")
        pins.update(r["input_file_hashes"])
        pins[str(rp)] = h
        pins[str(root / "declaration.json")] = digest(root / "declaration.json")
        stem = "successor" if sequences else "live-distilled"
        pins[str(root / f"{stem}-epoch256.pt")] = r["final_checkpoint_hash"]
        pins[str(root / f"{stem}-epoch0.pt")] = r["initial_checkpoint_hash"]
        if sequences and r["config"]["memory_handoff"] != ("carry" if index < 3 else "reset"):
            raise ValueError("exact trained memory protocol required for each candidate")
        candidates.append((root, h, r))
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            old_cases=declaration["old_cases"],
            DEV_cases=declaration["DEV_cases" if sequences else "dev_cases"],
            sequence_protocol=sequences,
            candidate_memory_handoff=[
                x[2]["config"]["memory_handoff"] if sequences else "reset" for x in candidates
            ],
            candidates=[str(x[0]) for x in candidates],
            independent_repeats=2,
            candidate_owns_both_kicks=a.entry_controller == "candidate",
            single_recurrent_actor=a.entry_controller == "candidate",
            learned_expert_context_switch=a.entry_controller == "qualified-prior",
            entry_controller=a.entry_controller,
            isolated_controller=a.entry_controller,
            exam_candidate_indices=list(range(4))
            if a.candidate_index is None
            else [a.candidate_index],
            context_routing=(
                "one learned candidate for all skills"
                if a.entry_controller == "candidate"
                else "frozen initial skill; learned live successor; no case-ID routing"
            ),
            robot_resets_per_world=1,
            reference_and_h_c_reset_explicit=not sequences,
            reference_clock_reset=True,
            memory_constant_during_hold_and_drop=True,
            previous_executed_action_always_carried=True,
            IK_executed=False,
            teacher_mixture=False,
            practice_ball_return_only_if_distance_m=2,
            goal_world_fixed=True,
            minimum_research_second_kick_quality=8,
            full_skill_second_kick_quality=12,
            old_and_DEV_minimum_quality_each=14,
            old16_minimum_quality=16 if sequences else 14,
            require_no_old_or_DEV_quality_regression=True,
            require_no_continuous_quality_or_stability_regression=True,
            full_continuous_skill_requires_all16_worlds_stable=True,
            motion_reference_still_required=True,
            team_game=False,
            fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
            commercial_publicity_authorized=False,
        ),
    )
    bench = None
    try:
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_distilled_actor import load_distilled_actor
        from rosclaw_soccer.rsi.paid_live_distilled_actor import load_live_distilled_actor
        from rosclaw_soccer.rsi.paid_research_bench import PaidResearchBench
        from rosclaw_soccer.rsi.paid_skill_handoff import practice_ball_drop, snapshot
        from rosclaw_soccer.rsi.paid_successor_actor import load_successor_actor
        from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        xmls = [Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml")]
        if not xmls or len({digest(f) for f in xmls}) != 1:
            raise ValueError("same qualified standard-ball physical body required")
        contract = PaidResearchPolicy(policy, expected_hash=PUBLISHED_HASH, horizon=1)
        baseline_loader = load_live_distilled_actor if sequences else load_distilled_actor
        baseline_actor, _ = baseline_loader(
            policy,
            checkpoint=a.base_root
            / ("live-distilled-epoch256.pt" if sequences else "distilled-epoch256.pt"),
            checkpoint_hash=base["final_checkpoint_hash"],
            training_receipt=bp,
            training_receipt_hash=digest(bp),
            scene_hash=digest(xmls[0]),
        )
        candidate_loader = load_successor_actor if sequences else load_live_distilled_actor
        actors = [
            candidate_loader(
                policy,
                checkpoint=root
                / ("successor-epoch256.pt" if sequences else "live-distilled-epoch256.pt"),
                checkpoint_hash=r["final_checkpoint_hash"],
                training_receipt=root / "complete.json",
                training_receipt_hash=h,
                scene_hash=digest(xmls[0]),
            )[0]
            for root, h, r in candidates
        ]
        sys.path.insert(0, str(policy.parent.parent))
        from exp.mujoco_soccer.constants import BODY_NAMES, JOINT_EFFORT_LIMIT_ARRAY
        from exp.mujoco_soccer.math_utils import quat_apply
        from exp.mujoco_soccer.motion import load_motion_clips
        from exp.mujoco_soccer.observations import build_observation
        from exp.mujoco_soccer.scene import MujocoSoccer

        cases, dev = (
            declaration["old_cases"],
            declaration["DEV_cases" if sequences else "dev_cases"],
        )
        names = list(dict.fromkeys(c["source"] for c in cases))
        clips = {
            n: load_motion_clips(policy.parent.parent / "motions/soccer-standard" / n)[0]
            for n in names
        }
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
        for case in cases + dev:
            bench.make(case)

        def quality(row):
            return bool(row is not None and contact_rank([row, row])[1] > 0)

        def save(path, raw):
            np.savez_compressed(path, **raw)
            return dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))

        def verify_neural(raw):
            if any(k.startswith("PLAN_") for k in raw) or not np.array_equal(
                raw["NN_action"], raw["NN_nominal_action"]
            ):
                raise ValueError("final neural inference cannot execute IK/teacher mixtures")

        def same(left, right):
            if set(left) != set(right) or any(not np.array_equal(left[k], right[k]) for k in left):
                raise ValueError("independent whole neural/native repeat or baseline differs")

        def isolated(case, actor, path):
            raw, row = bench.collect(case, actor)
            verify_neural(raw)
            row.update(save(path, raw))
            return raw, row

        def world(i, actor, folder, baseline=False, memory_handoff="reset"):
            folder.mkdir()
            ep = bench.make(cases[i])
            entry_actor = select_entry_actor(a.entry_controller, actor, baseline_actor)
            raw, first = bench.collect_current(ep, entry_actor)
            memory = successor_memory(
                raw, memory_handoff, same_actor=a.entry_controller == "candidate"
            )
            verify_neural(raw)
            artifacts = [save(folder / "first.npz", raw)]
            first.update(artifacts[-1])
            if baseline or a.entry_controller == "qualified-prior":
                reference = prior_results["old16"]["results"][i]
                with np.load(a.prior_exam_root / reference["raw_path"], allow_pickle=False) as z:
                    same(raw, {k: z[k] for k in z.files})
            artifacts.append(save(folder / "terminal-first.npz", snapshot(ep.env.data)))
            status = "FIRST_BODY_FAILURE" if first["body_failed"] else "FIRST_COMPLETE"
            second = None
            if not first["body_failed"]:
                try:
                    drop = practice_ball_drop(
                        ep, foot="left" if "left" in cases[i]["source"] else "right"
                    )
                except ValueError as error:
                    # A rejected declared fixture is retained, never resampled.
                    write_once(folder / "fixture-rejection.json", dict(error=str(error)))
                    status = "FIXTURE_REJECTED"
                else:
                    artifacts.append(save(folder / "drop.npz", drop))
                    holds = []
                    settled = True
                    if drop["performed"]:
                        for _ in range(50):
                            holds.append(ep.advance_hold())
                            q, v = ep.layout.canonical(ep.env.data)
                            if holds[-1]["body_failed"] or (q[38] <= 0.12 and abs(v[37]) <= 0.1):
                                break
                        artifacts.append(
                            save(
                                folder / "hold.npz",
                                {k: np.array([r[k] for r in holds]) for k in holds[0]},
                            )
                        )
                        settled = bool(q[38] <= 0.12 and abs(v[37]) <= 0.1)
                    if ep.termination_reason != "REFERENCE_COMPLETE":
                        status = "HOLD_BODY_FAILURE"
                    elif not settled:
                        status = "BALL_NOT_SETTLED"
                    else:
                        before = snapshot(ep.env.data)
                        previous = ep.last_action.copy()
                        ep.begin_current_state(clips[cases[i]["source"]], goal=ep.goal.copy())
                        after = snapshot(ep.env.data)
                        same(before, after)
                        artifacts.append(
                            save(
                                folder / "boundary.npz",
                                {
                                    **{"before_" + k: v for k, v in before.items()},
                                    **{"after_" + k: v for k, v in after.items()},
                                    "previous_actual_action": previous,
                                },
                            )
                        )
                        raw, second = bench.collect_current(ep, actor, initial_state=memory)
                        verify_neural(raw)
                        artifacts.append(save(folder / "second.npz", raw))
                        second.update(artifacts[-1])
                        artifacts.append(
                            save(folder / "terminal-second.npz", snapshot(ep.env.data))
                        )
                        status = (
                            "SECOND_BODY_FAILURE"
                            if second["body_failed"]
                            else "TWO_SKILLS_EXECUTED"
                        )
                        if baseline:
                            with np.load(
                                (
                                    a.prior_exam_root
                                    / prior["candidates"][0]["continuous"]["results"][i]["second"][
                                        "raw_path"
                                    ]
                                    if sequences
                                    else a.live_teacher_root / teacher["baseline"][i]["raw_path"]
                                ),
                                allow_pickle=False,
                            ) as z:
                                if any(not np.array_equal(raw[k], z[k]) for k in raw):
                                    raise ValueError("whole v0 continuous baseline differs")
            result = dict(
                case=i,
                status=status,
                first=first,
                second=second,
                artifacts=artifacts,
                native_steps=ep.native_steps,
                final_sim_time=float(ep.env.data.time),
                first_quality=quality(first),
                second_quality=quality(second),
                world_stable=status == "TWO_SKILLS_EXECUTED",
                robot_resets=1,
                memory_handoff=memory_handoff,
            )
            write_once(folder / "complete.json", result)
            return result

        old_baselines = {}
        for key, bank in (("old16", cases), ("dev16", dev)):
            folder = a.output / f"baseline-{key}"
            folder.mkdir()
            rows = []
            for i, case in enumerate(bank):
                raw, row = isolated(case, baseline_actor, folder / f"case{i}.npz")
                old = prior_results[key]["results"][i]
                with np.load(a.prior_exam_root / old["raw_path"], allow_pickle=False) as z:
                    same(raw, {k: z[k] for k in z.files})
                rows.append(row)
            old_baselines[key] = rows
        baseline_folder = a.output / "baseline-continuous"
        baseline_folder.mkdir()
        continuous_baseline = [
            world(i, baseline_actor, baseline_folder / f"case{i}", True) for i in range(16)
        ]
        write_once(
            a.output / "baseline-complete.json",
            dict(
                isolated=old_baselines,
                continuous=continuous_baseline,
                all_full_v0_arrays_reproduced=True,
            ),
        )

        def exam(index):
            actor = actors[index]
            initial_actor = select_entry_actor(a.entry_controller, actor, baseline_actor)
            folder = a.output / f"candidate{index}"
            folder.mkdir()
            isolated_results = {}
            for key, bank in (("old16", cases), ("dev16", dev)):
                target = folder / key
                target.mkdir()
                rows = []
                for i, case in enumerate(bank):
                    raw, row = isolated(case, initial_actor, target / f"case{i}-repeat0.npz")
                    other, again = isolated(case, initial_actor, target / f"case{i}-repeat1.npz")
                    same(raw, other)
                    row["independent_repeat_raw_hash"] = again["raw_hash"]
                    rows.append(row)
                isolated_results[key] = dict(
                    results=rows,
                    quality=sum(quality(r) for r in rows),
                    stable=sum(not r["body_failed"] for r in rows),
                    regressions=[
                        i
                        for i, (old, new) in enumerate(zip(old_baselines[key], rows, strict=True))
                        if quality(old) and not quality(new)
                    ],
                )
            target = folder / "continuous"
            target.mkdir()
            worlds = []
            memory_handoff = (
                candidates[index][2]["config"]["memory_handoff"] if sequences else "reset"
            )
            for i in range(16):
                first = world(i, actor, target / f"case{i}-repeat0", memory_handoff=memory_handoff)
                other = world(i, actor, target / f"case{i}-repeat1", memory_handoff=memory_handoff)
                if first["status"] != other["status"] or len(first["artifacts"]) != len(
                    other["artifacts"]
                ):
                    raise ValueError("independent complete world state machine differs")
                for x, y in zip(first["artifacts"], other["artifacts"], strict=True):
                    with (
                        np.load(a.output / x["raw_path"], allow_pickle=False) as left,
                        np.load(a.output / y["raw_path"], allow_pickle=False) as right,
                    ):
                        same({k: left[k] for k in left.files}, {k: right[k] for k in right.files})
                first["repeat_artifacts"] = other["artifacts"]
                worlds.append(first)
            second_quality = sum(w["second_quality"] for w in worlds)
            stable = sum(w["world_stable"] for w in worlds)
            regressions = [
                i
                for i, (old, new) in enumerate(zip(continuous_baseline, worlds, strict=True))
                if old["second_quality"] and not new["second_quality"]
            ]
            stability_regressions = [
                i
                for i, (old, new) in enumerate(zip(continuous_baseline, worlds, strict=True))
                if old["world_stable"] and not new["world_stable"]
            ]
            old_preserved = all(
                r["quality"] >= (16 if sequences and key == "old16" else 14)
                and r["stable"] == 16
                and not r["regressions"]
                for key, r in isolated_results.items()
            )
            research_gain = (
                old_preserved
                and second_quality >= 8
                and not regressions
                and not stability_regressions
            )
            full_skill = research_gain and second_quality >= 12 and stable == 16
            output = dict(
                candidate=index,
                memory_handoff=memory_handoff,
                entry_controller=a.entry_controller,
                isolated_controller=a.entry_controller,
                live_successor_controller="candidate",
                entry_checkpoint_hash=(
                    base["final_checkpoint_hash"]
                    if a.entry_controller == "qualified-prior"
                    else candidates[index][2]["final_checkpoint_hash"]
                ),
                single_recurrent_actor=a.entry_controller == "candidate",
                training_receipt_hash=candidates[index][1],
                checkpoint_hash=candidates[index][2]["final_checkpoint_hash"],
                isolated=isolated_results,
                continuous=dict(
                    results=worlds,
                    second_quality=second_quality,
                    stable=stable,
                    quality_regressions=regressions,
                    stability_regressions=stability_regressions,
                ),
                research_contact_gain=research_gain,
                full_continuous_skill_gate=full_skill,
                all_whole_neural_native_repeats_exact=True,
                IK_executed=False,
                teacher_mixture=False,
                goal_complete=False,
                team_game=False,
                fresh_exam=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
                hardware_authorized=False,
            )
            write_once(folder / "complete.json", output)
            print(
                json.dumps(
                    dict(
                        candidate=index,
                        old=isolated_results["old16"]["quality"],
                        DEV=isolated_results["dev16"]["quality"],
                        second_quality=second_quality,
                        stable=stable,
                        research_contact_gain=research_gain,
                        full_continuous_skill_gate=full_skill,
                    )
                ),
                flush=True,
            )
            return output

        with ThreadPoolExecutor(max_workers=4) as pool:
            indices = list(range(4)) if a.candidate_index is None else [a.candidate_index]
            results = list(pool.map(exam, indices))
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_LIVE_DISTILLED_CPU_FULL_WORLD_EXAM_CLOSED",
                sequence_protocol=sequences,
                input_file_hashes=pins,
                baseline_isolated=old_baselines,
                baseline_continuous=continuous_baseline,
                candidates=results,
                entry_controller=a.entry_controller,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                optimizer_updates=0,
                all_whole_neural_native_repeats_exact=True,
                IK_executed=False,
                goal_complete=False,
                team_game=False,
                fresh_exam=False,
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
                actual_native_steps=0 if bench is None else bench.native_steps,
                actual_neural_rows=0 if bench is None else bench.neural_rows,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
