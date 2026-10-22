"""After exact candidate exam, collect AND independently audit one fresh AR generation.

Research use of an unpromoted candidate is explicit. No old rollout is reused
as a new generation, no training or promotion is performed by this wrapper.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("exam-stage-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("exam-stage-head", "training-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--generation-index", type=int, choices=range(1, 100), required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen successor generation required")

    clean()
    sp = a.exam_stage_root / "complete.json"
    ap = a.exam_stage_root / "native-audit" / "complete.json"
    ep = a.exam_stage_root / "native-exam" / "complete.json"
    s, ar, er = [json.loads(x.read_text()) for x in (sp, ap, ep)]
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (sp, ap, ep))
        or s["status"] != "AR1_ACTOR_CRITIC_EXAM_CHAIN_CLOSED_NOT_GOAL_COMPLETE"
        or ar["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
        or ar["source_hash"] != digest(ep)
        or ar["worlds"] != 44
        or ar["training_hash"] != a.training_hash
        or er["training_hash"] != a.training_hash
        or s["training_hash"] != a.training_hash
        or ar.get("checkpoint_kind") != "ar1-actor-critic"
        or any(
            v["source_head"] != a.exam_stage_head
            or v["hardware_authorized"] is not False
            or v["promotion_authorized"] is not False
            for v in (s, ar, er)
        )
    ):
        raise ValueError("exact fully examined AR candidate required for research successor")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.fresh_followup_AR_generation.v1",
            source_head=a.expected_head,
            exam_stage_hash=digest(sp),
            native_audit_hash=digest(ap),
            training_hash=a.training_hash,
            generation_index=a.generation_index,
            research_use_of_unpromoted_candidate=True,
            automatic_retry=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    completed = []
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    try:
        experience = a.output / "experience"
        for module, name, options, status in (
            (
                "rsi_collect_dual_ar_experience",
                "experience",
                [
                    "--exam-root",
                    str(ep.parent),
                    "--audit-root",
                    str(ap.parent),
                    "--audit-hash",
                    digest(ap),
                    "--generation-index",
                    str(a.generation_index),
                ],
                "DUAL_AR1_ACTUAL_CALIBRATION_CLOSED_PENDING_NATIVE_AND_LAW_AUDIT",
            ),
            (
                "rsi_audit_dual_ar_experience",
                "audit",
                None,
                "DUAL_AR1_ALL8_OWN_NATIVE_AND_SAMPLING_LAW_AUDIT_PASS_NOT_TRAINED",
            ),
        ):
            clean()
            if options is None:
                options = [
                    "--root",
                    str(experience),
                    "--source-hash",
                    digest(experience / "complete.json"),
                ]
            argv = [
                sys.executable,
                "-m",
                "scripts." + module,
                "--expected-head",
                a.expected_head,
                "--output",
                str(a.output / name),
            ] + options
            write_once(a.output / (name + "-started.json"), dict(argv=argv))
            subprocess.run(argv, env=env, check=True)
            rp = a.output / name / "complete.json"
            r = json.loads(rp.read_text())
            if (
                r["source_head"] != a.expected_head
                or r["status"] != status
                or r["hardware_authorized"] is not False
                or r["promotion_authorized"] is not False
            ):
                raise ValueError("exact closed fresh generation evidence required")
            completed.append(dict(stage=name, hash=digest(rp)))
            write_once(a.output / (name + "-closed.json"), completed[-1])
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="FRESH_AR_GENERATION_COLLECTED_AND_AUDITED_NOT_TRAINED",
                source_head=a.expected_head,
                completed=completed,
                training_hash=a.training_hash,
                actor_updates=0,
                critic_updates=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                completed=completed,
                automatic_retry=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
