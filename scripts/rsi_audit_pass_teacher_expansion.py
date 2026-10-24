"""Replay ALL6 transfer trials on private NN histories and a manual native clock."""

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
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_output_capacity import require_output_capacity
    from scripts.rsi_pass_teacher_expansion_common import admit_pilot, verify_expansion
    from scripts.rsi_self_teacher_search_common import open_setup

    paths = [a.root / n for n in ("complete.json", "declaration.json", "scene.xml")]
    if (
        a.output.exists()
        or a.root.is_symlink()
        or any(p.is_symlink() for p in paths)
        or (a.root / "failed.json").exists()
    ):
        raise ValueError("complete source and unused whole audit required")
    require_output_capacity(a.output, 32 * 1024**2)
    s, d = [json.loads(p.read_text()) for p in paths[:2]]
    if (
        s["status"] != "PASS_PROPOSAL_TRANSFER6_COLLECTED_PENDING_WHOLE_OWN"
        or d["protocol"] != "soccer.safe_pass_proposal_transfer6.v1"
        or any(v["source_head"] != a.producer_head for v in (s, d))
        or s["declaration_hash"] != digest(paths[1])
        or s["scene_hash"] != digest(paths[2])
        or s["state_id"] != d["state_id"]
        or d["preflight_only"] is not False
        or d["maximum_worlds"] != 6
        or d["maximum_native_steps"] != 24000
        or d["maximum_actor_calls"] != 4800
        or s["pilot_hash"] != d["pilot_hash"]
        or any(
            v[k] is not False
            for v in (s, d)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("exact completed bounded expansion source required")
    _, plans, cal, scope, pf, pins, bank, cr = admit_pilot(
        Path(d["pilot_root"]),
        d["pilot_hash"],
        d["pilot_head"],
        Path(d["model_root"]),
        d["model_hash"],
        d["model_head"],
    )
    if d["plans"] != plans:
        raise ValueError("whole predeclared nearest-proposal plan differs")
    verify_expansion(s, plans, cal, s["state_id"])
    verified = dict(pins)
    for key, h in s["input_file_hashes"].items():
        if key in pins and pins[key] != h:
            raise ValueError("conflicting expansion ancestry")
        pins[key] = h
    for path in paths:
        pins[str(path)] = digest(path)
    for i, row in enumerate(s["records"]):
        if set(row["artifacts"]) != {"initial", "physical", "A", "B"}:
            raise ValueError("all neural and physical primitives required")
        for art in row["artifacts"].values():
            f = a.root / art["path"]
            if f.is_symlink() or f.resolve().parent != (a.root / f"trial{i}").resolve():
                raise ValueError("confined source primitives required")
            pins[str(f)] = art["hash"]
    check_pins({k: h for k, h in pins.items() if verified.get(k) != h})
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

        setup, actor = open_setup(scope, pf, pins, bank, cr / "scene.xml")
        if setup.xml != paths[2].read_text():
            raise ValueError("unchanged whole native scene required")
        prior = None
        for i, row in enumerate(s["records"]):
            extra = {
                k: row[k]
                for k in (
                    "task",
                    "tag",
                    "role",
                    "proposal_donor_state",
                    "proposal_donor_controlled",
                    "prefix_identity_fields",
                )
            }
            claim = {k: v for k, v in row.items() if k not in extra}
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
            records.append(own | extra)
        if counts != s["counts"]:
            raise ValueError("whole source/own accounting differs")
        verify_expansion(s | dict(records=records), plans, cal, s["state_id"])
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PASS_PROPOSAL_TRANSFER_ALL6_OWN_NATIVE_MATCH",
                source_head=a.expected_head,
                input_file_hashes=pins,
                producer_hash=digest(paths[0]),
                state_id=s["state_id"],
                counts=counts,
                records=records,
                selections=s["selections"],
                manual_native_clock=True,
                source_actions_executed=False,
                borrowed_hidden_state=False,
                fixed_proposal_and_selection_reconstructed=True,
                student_optimizer_updates=0,
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
                records=records,
                counts=counts,
                automatic_retry=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
