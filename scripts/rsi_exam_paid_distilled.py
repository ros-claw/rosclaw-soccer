"""Actual CPU neural-only exams of all four predeclared distilled candidates.

No Cartesian intent, IK, CEM lookup, teacher correction or mixture executes in
candidate rollouts. Full independent repeats must match all raw arrays exactly.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_search_paid_foot_contact import contact_rank

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--teacher-hash", required=True)
    p.add_argument("--candidate", nargs=2, action="append", required=True, metavar=("ROOT", "HASH"))
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    tp = a.teacher_root / "complete.json"
    if (
        a.output.exists()
        or len(a.candidate) != 4
        or digest(tp) != a.teacher_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError(
            "all four fixed candidates, unused output and exact frozen sources required"
        )
    t = json.loads(tp.read_text())
    dp = a.teacher_root / "declaration.json"
    d = json.loads(dp.read_text())
    if t["status"] != "PAID_CARTESIAN_CONTACT_TEACHER_RESEARCH_NOT_PROMOTED":
        raise ValueError("actual common teacher/body cohort required")
    pins = {**t["input_file_hashes"], str(tp): a.teacher_hash, str(dp): digest(dp)}
    roots = []
    for root, h in a.candidate:
        root = Path(root)
        rp = root / "complete.json"
        if digest(rp) != h:
            raise ValueError("exact completed training receipt required")
        r = json.loads(rp.read_text())
        pins.update(r["input_file_hashes"])
        pins[str(rp)] = h
        pins[str(root / "declaration.json")] = digest(root / "declaration.json")
        cp = root / f"distilled-epoch{r['config']['epochs']}.pt"
        pins[str(cp)] = r["final_checkpoint_hash"]
        roots.append((root, h, r, cp))
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    for row in t["baseline"] + t["baseline_dev"]:
        pins[str(a.teacher_root / row["raw_path"])] = row["raw_hash"]
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            candidate_order=[str(x[0]) for x in roots],
            old_cases=d["train_cases"],
            dev_cases=d["dev_cases"],
            Cartesian_intervention=False,
            IK_executed=False,
            teacher_mixture=False,
            motion_reference_still_required=True,
            complete_neural_native_repeats=2,
            minimum_skill_gate_old16_quality=14,
            minimum_skill_gate_DEV16_quality=12,
            require_no_quality_regressions=True,
            require_all32_bodies_stable=True,
            commercial_publicity_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    try:
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_distilled_actor import load_distilled_actor
        from rosclaw_soccer.rsi.paid_research_bench import PaidResearchBench
        from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH, PaidTrainableActor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policies = [Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx")]
        xmls = [Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml")]
        if len(policies) != 1 or not xmls or len({digest(x) for x in xmls}) != 1:
            raise ValueError("exact common research actor/standard physics required")
        ref = policies[0].parent.parent
        contract = PaidResearchPolicy(policies[0], expected_hash=PUBLISHED_HASH, horizon=1)
        prior = PaidTrainableActor(policies[0], expected_hash=PUBLISHED_HASH).eval()
        actors = [
            load_distilled_actor(
                policies[0],
                checkpoint=cp,
                checkpoint_hash=r["final_checkpoint_hash"],
                training_receipt=root / "complete.json",
                training_receipt_hash=h,
                scene_hash=digest(xmls[0]),
            )[0]
            for root, h, r, cp in roots
        ]
        sys.path.insert(0, str(ref))
        from exp.mujoco_soccer.constants import BODY_NAMES, JOINT_EFFORT_LIMIT_ARRAY
        from exp.mujoco_soccer.math_utils import quat_apply
        from exp.mujoco_soccer.motion import load_motion_clips
        from exp.mujoco_soccer.observations import build_observation
        from exp.mujoco_soccer.scene import MujocoSoccer

        names = list(dict.fromkeys(c["source"] for c in d["train_cases"]))
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
        for case in d["train_cases"] + d["dev_cases"]:
            bench.make(case)

        def execute(case, actor, path):
            raw, row = bench.collect(case, actor)
            if any(k.startswith("PLAN_") for k in raw) or row["composite_teacher"]:
                raise ValueError("neural-only exam forbids planner traces/intervention")
            if not np.array_equal(raw["NN_action"], raw["NN_nominal_action"]):
                raise ValueError("actual executed action must equal this neural actor output")
            np.savez_compressed(path, **raw)
            row.update(raw_hash=digest(path), raw_path=str(path.relative_to(a.output)))
            return raw, row

        baseline = {}
        for key, cases, reference in (
            ("old16", d["train_cases"], t["baseline"]),
            ("dev16", d["dev_cases"], t["baseline_dev"]),
        ):
            folder = a.output / f"baseline-{key}"
            folder.mkdir()
            rows = []
            for i, (case, old) in enumerate(zip(cases, reference, strict=True)):
                raw, row = execute(case, prior, folder / f"case{i}.npz")
                with np.load(a.teacher_root / old["raw_path"], allow_pickle=False) as z:
                    for k in raw:
                        if not np.array_equal(raw[k], z[k]):
                            raise ValueError("full original prior re-admission differs: " + k)
                rows.append(row)
            baseline[key] = rows
            write_once(
                folder / "complete.json", dict(results=rows, full_original_prior_reproduced=True)
            )

        def quality(row):
            return contact_rank([row, row])[1] > 0

        def exam(index):
            actor = actors[index]
            folder = a.output / f"candidate{index}"
            folder.mkdir()
            result = {}
            for key, cases in (("old16", d["train_cases"]), ("dev16", d["dev_cases"])):
                target = folder / key
                target.mkdir()
                rows = []
                for i, case in enumerate(cases):
                    raw, row = execute(case, actor, target / f"case{i}-repeat0.npz")
                    second, again = execute(case, actor, target / f"case{i}-repeat1.npz")
                    if set(raw) != set(second) or any(
                        not np.array_equal(raw[k], second[k]) for k in raw
                    ):
                        raise ValueError("whole independent neural/native repeat differs")
                    row["independent_repeat_raw_hash"] = again["raw_hash"]
                    rows.append(row)
                result[key] = dict(
                    results=rows,
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
                        for i, (x, y) in enumerate(zip(baseline[key], rows, strict=True))
                        if not quality(x) and quality(y)
                    ],
                    regressions=[
                        i
                        for i, (x, y) in enumerate(zip(baseline[key], rows, strict=True))
                        if quality(x) and not quality(y)
                    ],
                )
            passed = (
                result["old16"]["quality"] >= 14
                and result["dev16"]["quality"] >= 12
                and all(x["stable"] == 16 and not x["regressions"] for x in result.values())
            )
            output = dict(
                candidate_index=index,
                training_receipt_hash=roots[index][1],
                checkpoint_hash=roots[index][2]["final_checkpoint_hash"],
                results=result,
                status="NEURAL_ONLY_DISTILLATION_SKILL_GATE_PASSED"
                if passed
                else "NEURAL_ONLY_DISTILLATION_SKILL_GATE_NOT_MET",
                all_neural_native_repeats_exact=True,
                Cartesian_intervention=False,
                motion_reference_still_required=True,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
            write_once(folder / "complete.json", output)
            print(
                json.dumps(
                    dict(
                        candidate=index,
                        status=output["status"],
                        old_quality=result["old16"]["quality"],
                        DEV_quality=result["dev16"]["quality"],
                    )
                ),
                flush=True,
            )
            return output

        with ThreadPoolExecutor(max_workers=4) as pool:
            candidates = list(pool.map(exam, range(4)))
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_DISTILLED_CPU_NEURAL_ONLY_EXAM_CLOSED",
                input_file_hashes=pins,
                baseline=baseline,
                candidates=candidates,
                native_steps=bench.native_steps,
                new_neural_rows=bench.neural_rows,
                optimizer_updates=0,
                Cartesian_intervention=False,
                teacher_mixture=False,
                motion_reference_still_required=True,
                all_full_neural_native_repeats_exact=True,
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
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
