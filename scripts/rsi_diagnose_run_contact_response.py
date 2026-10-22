"""Fit TRAIN intervention response only after all-trial whole native audit PASS."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "audit-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("source-hash", "auditor-head", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("frozen response diagnostic source required")

    clean()
    sp, ap, dp = (
        a.root / "complete.json",
        a.audit_root / "complete.json",
        a.root / "declaration.json",
    )
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (sp, ap, dp))
        or (a.audit_root / "failed.json").exists()
        or digest(sp) != a.source_hash
    ):
        raise ValueError("unused diagnostic and exact successful source required")
    r, ar, d = [json.loads(x.read_text()) for x in (sp, ap, dp)]
    if (
        ar["status"] != "INHERITED_RUN_EXPERT_ALL_TRIAL_WHOLE_NATIVE_OWN_NN_IK_AUDIT_PASS"
        or ar["source_hash"] != a.source_hash
        or ar["source_head"] != a.auditor_head
        or ar["worlds"] != 134
        or not ar["independent_whole_world_replay"]
        or ar["counts"]
        != dict(
            native=r["actual_native_steps"],
            neural=r["actual_neural_calls"],
            kinematic_intent=r["actual_kinematic_intent_calls"],
        )
        or d["protocol"] != "paid.TRAIN.inherited_run_actual_state_expert.contextual_v2"
        or r["declaration_hash"] != digest(dp)
        or any(ar[k] is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError("fully audited fixed context-v2 actual physics required")
    pins = dict(ar["input_file_hashes"])
    for x in (sp, ap, dp):
        pins[str(x)] = digest(x)
    for x in (
        Path(__file__),
        Path("src/rosclaw_soccer/rsi/run_contact_response.py"),
        Path("src/rosclaw_soccer/rsi/run_teacher_objective.py"),
    ):
        pins[str(x.resolve())] = digest(x)
    check_pins(pins)
    import numpy as np

    from rosclaw_soccer.rsi.run_contact_response import bounded_proposals, fit_response
    from rosclaw_soccer.rsi.run_teacher_objective import run_teacher_rank

    reports = []
    for ci in range(2):
        rows = [
            x
            for x in r["records"]
            if x["course"] == ci and (x["tag"].startswith("g") or x["tag"] == f"baseline-c{ci}")
        ]
        if len(rows) != 65:
            raise ValueError("complete unique TRAIN context required")
        safe = [x for x in rows if run_teacher_rank(x)[1] and not x["result"]["forbidden"]]
        x = np.array([np.array(v["latent"]).reshape(12) for v in safe], np.float64)
        y = np.array(
            [
                [
                    v["result"]["postcontact"]["goal_lateral_1p2s_m"],
                    v["result"]["postcontact"]["goal_progress_1p2s_m"],
                ]
                for v in safe
            ],
            np.float64,
        )
        model = fit_response(x, y)
        best = max(rows, key=run_teacher_rank)
        base = np.array(best["latent"], np.float64).reshape(12)
        grad = np.array(model["latent_gradient"])[:, 0]
        proposals = bounded_proposals(
            base, best["result"]["postcontact"]["goal_lateral_1p2s_m"], grad
        )
        reports.append(
            dict(
                course=ci,
                fit_tags=[v["tag"] for v in safe],
                excluded_rows=len(rows) - len(safe),
                response=model,
                actual_base_tag=best["tag"],
                actual_base_quality=best["healthy_quality"],
                actual_base_rank=run_teacher_rank(best),
                center=d["centers"][ci],
                proposed_latents=[np.array(v).reshape(2, 6).tolist() for v in proposals],
                all_proposals_require_new_whole_physics=True,
            )
        )
    check_pins(pins)
    clean()
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="TRAIN_RUN_CONTACT_RESPONSE_DIAGNOSED_NOT_MOTOR_POLICY",
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_hash=a.source_hash,
            audit_hash=digest(ap),
            contexts=reports,
            physical_worlds_executed=0,
            neural_motor_updates=0,
            learned_actor_claim=False,
            predicted_success_is_evidence=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    print(
        json.dumps(
            dict(
                status="RESPONSE_DIAGNOSIS_CLOSED",
                contexts=[
                    dict(
                        course=x["course"],
                        rows=x["response"]["rows"],
                        loo_rmse=x["response"]["leave_one_out_rmse"],
                    )
                    for x in reports
                ],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
