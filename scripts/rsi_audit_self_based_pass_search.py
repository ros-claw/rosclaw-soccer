"""Independently recompute ALL68 new search worlds and the full CEM decision.

Manual native clock, own observations/private neural state, no saved actions,
no producer integrator. Inherited4 calibrations come from their whole40 proof.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("producer-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_self_teacher_search_common import admit, open_setup, verify_search

    paths = [a.root / x for x in ("complete.json", "declaration.json", "scene.xml")]
    if a.output.exists() or any(p.is_symlink() for p in paths) or (a.root / "failed.json").exists():
        raise ValueError("whole closed source and unused own-native namespace required")
    s, d = [json.loads(p.read_text()) for p in paths[:2]]
    if (
        s["status"] != "WHOLE_LEARNED_SELF_TEACHER_SEARCH_COLLECTED_PENDING_ALL68_OWN"
        or d["protocol"] != "soccer.whole_learned_self_teacher_CEM68.v1"
        or s["source_head"] != a.producer_head
        or d["source_head"] != a.producer_head
        or s["declaration_hash"] != digest(paths[1])
        or s["scene_hash"] != digest(paths[2])
        or d["preflight_only"] is not False
        or d["state_id"] != s["state_id"]
        or d["new_worlds"] != 68
        or d["maximum_native_steps"] != 272000
        or d["maximum_actor_calls"] != 54400
        or d["rng_seed_base"] != 146023 + s["state_id"] * 101
        or d["generations_per_goal"] != 2
        or d["proposals_per_generation"] != 16
        or d["selected_repeats_per_goal"] != 2
        or any(
            v[k] is not False
            for v in (s, d)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
        or any(v[k] != 0 for v in (s, d) for k in ("optimizer_updates", "Fresh_cases_read"))
    ):
        raise ValueError("exact complete bounded search identity required")
    cal_root, cal_own, review = [
        Path(d[k]) for k in ("calibration_root", "calibration_own_root", "review_root")
    ]
    cal, scope, pf, pins, bank = admit(cal_root, cal_own, review, d["calibration_head"])
    if [digest(r / "complete.json") for r in (cal_root, cal_own, review)] != [
        d[k] for k in ("calibration_hash", "calibration_own_hash", "review_hash")
    ]:
        raise ValueError("same inherited whole calibrations required")
    for k, h in s["input_file_hashes"].items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting evidence ancestry")
        pins[k] = h
    for path in paths:
        pins[str(path)] = digest(path)
    verify_search(s, cal)
    for i, row in enumerate(s["records"]):
        if set(row["artifacts"]) != {"initial", "physical", "A", "B"}:
            raise ValueError("all original primitives required")
        for art in row["artifacts"].values():
            f = a.root / art["path"]
            if f.is_symlink() or f.resolve().parent != (a.root / f"trial{i}").resolve():
                raise ValueError("confined exact source artifacts required")
            pins[str(f)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_hash=digest(paths[0]),
            input_file_hashes=pins,
            manual_native_clock=True,
            source_actions_executed=False,
            borrowed_hidden_state=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    counts = dict(native_steps=0, actor_calls=0)
    records = []
    try:
        from scripts.rsi_pass_transfer_replay_trial import replay_trial

        setup, actor = open_setup(scope, pf, pins, bank, cal_root / "scene.xml")
        if setup.xml != paths[2].read_text():
            raise ValueError("same actual two-body scene required")
        prior = None
        for i, row in enumerate(s["records"]):
            extras = {k: row[k] for k in ("tag", "search_task", "prefix_identity_fields")}
            claim = {k: v for k, v in row.items() if k not in extras}
            own, prior = replay_trial(
                setup,
                actor,
                row["course"],
                claim,
                raw_root=a.root,
                output=a.output,
                index=i,
                counts=counts,
                prior_initial=prior,
                emit_course_rows=False,
            )
            records.append(own | extras)
        if counts != s["counts"]:
            raise ValueError("all actual native/NN accounting required")
        verify_search(s | dict(records=records), cal)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="WHOLE_LEARNED_SELF_TEACHER_ALL68_OWN_NATIVE_MATCH",
                source_head=a.expected_head,
                producer_hash=digest(paths[0]),
                input_file_hashes=pins,
                state_id=s["state_id"],
                counts=counts,
                records=records,
                generations=s["generations"],
                selections=s["selections"],
                manual_native_clock=True,
                source_actions_executed=False,
                borrowed_hidden_state=False,
                all_CEM_decisions_reconstructed=True,
                optimizer_updates=0,
                Fresh_cases_read=0,
                correction_bank_ready=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as e:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(e).__name__,
                error=str(e),
                counts=counts,
                records=records,
                automatic_retry=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
