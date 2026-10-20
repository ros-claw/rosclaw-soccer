"""Automatic read-only full40 successor; never promotes a candidate policy."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "core-source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-inputs-hash", "expected-exam-source"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--campaign", action="append", nargs=2, required=True, metavar=("UNIT", "ROOT"))
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    _, bd, pins, cases = admit(a)
    if len(a.campaign) != 4 or len({u for u, _ in a.campaign}) != 4:
        raise ValueError("four unique original40 shards required")
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            campaigns=a.campaign,
            expected_exam_source=a.expected_exam_source,
            no_new_physics_or_optimization=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        rows, candidate_hash = {}, None
        for unit, path in a.campaign:
            wait_for_terminal_collection(unit, wait=True)
            root = Path(path)
            d = json.loads((root / "declaration.json").read_text())
            c = json.loads((root / "complete.json").read_text())
            if (
                d["source_head"] != a.expected_exam_source
                or c["source_head"] != a.expected_exam_source
                or c["status"] != "KINEMATIC_COLUMN_NATIVE_EPISODES_AND_REPLAY_VERIFIED"
                or c["new_native_steps"] != 6000 * len(d["indices"])
                or c["optimizer_steps"] != 0
                or c["candidate_hash"] != d["candidate_hash"]
                or candidate_hash not in (None, c["candidate_hash"])
            ):
                raise ValueError("closed same-candidate native exam differs")
            candidate_hash = c["candidate_hash"]
            for k, v in d["input_file_hashes"].items():
                if k in pins and pins[k] != v:
                    raise ValueError("conflicting frozen inputs")
                pins[k] = v
            for name in ("declaration.json", "complete.json"):
                pins[str(root / name)] = digest(root / name)
            result = c["results"]
            if (
                [r["course_index"] for r in result["results"]] != d["indices"]
                or result["status"] != "ALL_DECLARED_EPISODES_AND_NOISE_REPLAY_VERIFIED"
                or result["new_native_steps"] != c["new_native_steps"]
            ):
                raise ValueError("inner complete horizon differs")
            for r in result["results"]:
                i = r["course_index"]
                if i not in range(40) or i in rows or r["split"] != bd["rows"][i]["split"]:
                    raise ValueError("unique original40 split-bound courses required")
                folder = root / f"course{i:02d}"
                if json.loads((folder / "complete.json").read_text()) != r:
                    raise ValueError("per-course terminal receipt differs")
                for name, expected in (
                    ("native.npz", r["native_hash"]),
                    ("sampling.npz", r["sampling_hash"]),
                    ("complete.json", digest(folder / "complete.json")),
                ):
                    pins[str(folder / name)] = expected
                proof = r["replay"]
                if (
                    r["native_steps"] != 6000
                    or proof["additional_native_steps"] != 3000
                    or proof["student_neural_and_muscle_native_bit_exact"] is not True
                    or not r["regenerated_noise_and_prequant_likelihood_bit_exact"]
                    or proof["intention_network_calls"] != 600
                    or proof["muscle_decoder_calls"] != 280
                    or proof["teacher_encoder_calls"] != 0
                    or proof["teacher_planner_calls"] != 0
                ):
                    raise ValueError("independent geometry/NN/muscle/native replay required")
                rows[i] = r
        if set(rows) != set(range(40)):
            raise ValueError("whole40 required; never drop failing cases")
        passed = {
            i
            for i, r in rows.items()
            if r["score"]["qualified"]
            and r["prefix_diagnostics"]["prefix_physical_safe"]
            and r["prefix_diagnostics"]["post_boundary_control_can_cause_first_touch"]
        }
        baseline = {i for i, (r, _) in cases[0].items() if r["score"]["qualified"]}
        if baseline != {0, 10}:
            raise ValueError("original frozen baseline identity differs")
        balance = sum(
            r["score"]["minimum_pelvis_z_m"] >= 0.65 and r["score"]["minimum_upright"] >= 0.9
            for r in rows.values()
        )
        nonfoot = sum(r["score"]["maximum_nonfoot_force_n"] > 1 for r in rows.values())
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="KINEMATIC_COLUMN_ALL40_AUTHENTICATED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                candidate_hash=candidate_hash,
                courses=40,
                baseline_success=2,
                success=len(passed),
                success_indices=sorted(passed),
                rescues=sorted(passed - baseline),
                regressions=sorted(baseline - passed),
                validation_success=sum(rows[i]["split"] == "validation" for i in passed),
                moving_success=sum(bd["rows"][i]["course"]["course"][2] != 0 for i in passed),
                body_balance=balance,
                nonfoot_cases=nonfoot,
                development_necessary_gate_pass=bool(
                    len(passed) > 2 and baseline <= passed and balance == 40 and nonfoot <= 14
                ),
                development_gate_is_not_promotion=True,
                fresh_exam_pass=False,
                reviewed_native_steps=240000,
                new_native_steps=0,
                optimizer_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        print(
            "GEOMETRY_ALL40_REVIEWED",
            sorted(passed),
            "regressions",
            sorted(baseline - passed),
            flush=True,
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                new_native_steps=0,
                optimizer_steps=0,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
