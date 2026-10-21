"""SIM research: consecutive neural kicks with a continuous physical robot.

Three skills per world, with explicit trainer ball drops when >2m from the
robot. No body/clock reset after the first skill. Not a team-game claim.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exam-root", type=Path, required=True)
    p.add_argument("--exam-hash", required=True)
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    xp, tp = a.exam_root / "complete.json", a.training_root / "complete.json"
    if (
        a.output.exists()
        or digest(xp) != a.exam_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output, pinned completed exam and clean frozen source required")
    exam = json.loads(xp.read_text())
    declaration = json.loads((a.exam_root / "declaration.json").read_text())
    selected = exam["candidates"][0]
    if (
        selected["candidate_index"] != 0
        or selected["training_receipt_hash"] != digest(tp)
        or selected["status"] != "NEURAL_ONLY_DISTILLATION_SKILL_GATE_PASSED"
    ):
        raise ValueError("previously qualified v0 neural actor required")
    pins = {
        **exam["input_file_hashes"],
        str(xp): a.exam_hash,
        str(tp): digest(tp),
        str(a.exam_root / "declaration.json"): digest(a.exam_root / "declaration.json"),
    }
    cp = a.training_root / "distilled-epoch256.pt"
    pins[str(cp)] = selected["checkpoint_hash"]
    cases = declaration["old_cases"][::2]
    if len(cases) != 8 or len({c["source"] for c in cases}) != 8:
        raise ValueError("all eight declared static anchors required")
    reference_paths = []
    for i in range(8):
        row = selected["results"]["old16"]["results"][i * 2]
        path = a.exam_root / "candidate0" / "old16" / f"case{i * 2}-repeat0.npz"
        pins[str(path)] = row["raw_hash"]
        reference_paths.append(path)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    opposite = {
        "left": next(c["source"] for c in cases if "right" in c["source"]),
        "right": next(c["source"] for c in cases if "left" in c["source"]),
    }
    matrix = [
        dict(
            initial=case,
            sources=[
                case["source"],
                case["source"],
                opposite["left" if "left" in case["source"] else "right"],
            ],
            transition_controls=arm,
        )
        for case in cases
        for arm in (0, 12)
    ]
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            matrix=matrix,
            whole_world_repeats=2,
            robot_resets_per_world=1,
            h_c_reset_at_explicit_skill_boundary=True,
            previous_actual_action_carried=True,
            practice_ball_drop_only_if_distance_at_least_m=2,
            practice_drop_forward_m=0.8,
            practice_drop_lateral_m=0.15,
            practice_drop_height_m=0.4,
            maximum_settling_controls=50,
            never_force_settle=True,
            goal_world_unchanged=True,
            motion_reference_still_required=True,
            team_game=False,
            commercial_publicity_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    bench = None
    try:
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_distilled_actor import load_distilled_actor
        from rosclaw_soccer.rsi.paid_research_bench import PaidResearchBench
        from rosclaw_soccer.rsi.paid_skill_handoff import practice_ball_drop, snapshot
        from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        xmls = [Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml")]
        if not xmls or len({digest(f) for f in xmls}) != 1:
            raise ValueError("identical pinned standard-ball scene required")
        ref = policy.parent.parent
        contract = PaidResearchPolicy(policy, expected_hash=PUBLISHED_HASH, horizon=1)
        actor, _ = load_distilled_actor(
            policy,
            checkpoint=cp,
            checkpoint_hash=selected["checkpoint_hash"],
            training_receipt=tp,
            training_receipt_hash=digest(tp),
            scene_hash=digest(xmls[0]),
        )
        sys.path.insert(0, str(ref))
        from exp.mujoco_soccer.constants import BODY_NAMES, JOINT_EFFORT_LIMIT_ARRAY
        from exp.mujoco_soccer.math_utils import quat_apply
        from exp.mujoco_soccer.motion import load_motion_clips
        from exp.mujoco_soccer.observations import build_observation
        from exp.mujoco_soccer.scene import MujocoSoccer

        clips = {
            c["source"]: load_motion_clips(ref / "motions/soccer-standard" / c["source"])[0]
            for c in cases
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
        for case in cases:
            bench.make(case)  # Admit every initial course before worker allocation.

        def save(path, raw):
            np.savez_compressed(path, **raw)
            return dict(path=str(path.relative_to(a.output)), hash=digest(path))

        def world(index, spec, repeat):
            folder = a.output / f"world{index}-repeat{repeat}"
            folder.mkdir()
            ep = bench.make(spec["initial"])
            artifacts, segments = [], []
            status = "THREE_SKILLS_EXECUTED"
            for phase, source in enumerate(spec["sources"]):
                if phase:
                    before = snapshot(ep.env.data)
                    try:
                        fixture = practice_ball_drop(
                            ep, foot="left" if "left" in source else "right"
                        )
                    except ValueError as error:
                        status = "PRACTICE_FIXTURE_REJECTED"
                        segments.append(dict(phase=phase, error=str(error)))
                        artifacts.append(save(folder / f"rejected{phase}.npz", before))
                        break
                    artifacts.append(save(folder / f"drop{phase}.npz", fixture))
                    holds = []
                    if fixture["performed"]:
                        for _ in range(50):
                            holds.append(ep.advance_hold())
                            q, v = ep.layout.canonical(ep.env.data)
                            if holds[-1]["body_failed"] or (q[38] <= 0.12 and abs(v[37]) <= 0.1):
                                break
                        artifacts.append(
                            save(
                                folder / f"hold{phase}.npz",
                                {k: np.array([r[k] for r in holds]) for k in holds[0]},
                            )
                        )
                        if holds[-1]["body_failed"]:
                            status = "BODY_FAILED_DURING_DROP_HOLD"
                            break
                        if not (q[38] <= 0.12 and abs(v[37]) <= 0.1):
                            status = "BALL_DID_NOT_SETTLE_IN_DECLARED_WINDOW"
                            break
                    old = snapshot(ep.env.data)
                    ep.begin_current_state(clips[source], goal=ep.goal.copy())
                    new = snapshot(ep.env.data)
                    if any(not np.array_equal(old[k], new[k]) for k in old):
                        raise ValueError("skill boundary changed physical integration state")
                    artifacts.append(
                        save(
                            folder / f"boundary{phase}.npz",
                            {
                                **{"before_" + k: v for k, v in old.items()},
                                **{"after_" + k: v for k, v in new.items()},
                            },
                        )
                    )
                raw, row = bench.collect_current(
                    ep,
                    actor,
                    transition_controls=spec["transition_controls"] if phase else 0,
                )
                if phase == 0:
                    with np.load(reference_paths[index // 2], allow_pickle=False) as z:
                        if set(raw) != set(z.files) or any(
                            not np.array_equal(raw[k], z[k]) for k in raw
                        ):
                            raise ValueError(
                                "first unmodified neural skill failed whole-array re-admission"
                            )
                row.update(
                    phase=phase,
                    research_contact_quality=contact_rank([row, row])[1] > 0,
                    termination_reason=ep.termination_reason,
                )
                segments.append(row)
                artifacts.append(save(folder / f"skill{phase}.npz", raw))
                # Save terminal for chain verification, including actual final time.
                artifacts.append(save(folder / f"terminal{phase}.npz", snapshot(ep.env.data)))
                if row["body_failed"]:
                    status = "BODY_FAILED_DURING_SKILL"
                    break
            result = dict(
                index=index,
                repeat=repeat,
                status=status,
                spec=spec,
                segments=segments,
                artifacts=artifacts,
                native_steps=ep.native_steps,
                final_sim_time=float(ep.env.data.time),
                body_resets=1,
            )
            write_once(folder / "complete.json", result)
            print(
                json.dumps(
                    dict(
                        world=index,
                        repeat=repeat,
                        status=status,
                        qualities=[s.get("research_contact_quality") for s in segments],
                    )
                ),
                flush=True,
            )
            return result

        def paired(item):
            i, spec = item
            first, second = world(i, spec, 0), world(i, spec, 1)
            if first["status"] != second["status"] or first["segments"] != second["segments"]:
                raise ValueError("whole-world repeat outcome differs")
            if len(first["artifacts"]) != len(second["artifacts"]):
                raise ValueError("whole-world repeat artifact count differs")
            for x, y in zip(first["artifacts"], second["artifacts"], strict=True):
                with (
                    np.load(a.output / x["path"], allow_pickle=False) as left,
                    np.load(a.output / y["path"], allow_pickle=False) as right,
                ):
                    if set(left.files) != set(right.files) or any(
                        not np.array_equal(left[k], right[k]) for k in left.files
                    ):
                        raise ValueError("whole-world independent repeat arrays differ")
            return first

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(paired, enumerate(matrix)))
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CONTINUOUS_BODY_PRACTICE_EXAM_CLOSED",
                input_file_hashes=pins,
                results=results,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                optimizer_updates=0,
                all_independent_whole_world_repeats_exact=True,
                robot_state_reset_between_skills=False,
                team_game=False,
                motion_reference_still_required=True,
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
                neural_rows=0 if bench is None else bench.neural_rows,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
