"""Strictly integrate post-first standing/settle/successor-kick SIM phases.

Every replay begins at its own recorded complete ten-field snapshot. This is
not an extra independent course bank, a whole-match run, or new neural training.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "audit-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("hash", "audit-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or digest(a.root / "complete.json") != a.hash
        or digest(a.audit_root / "complete.json") != a.audit_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("exact closed input and unused immutable replay output required")
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate

    r, d = [
        json.loads((a.root / name).read_text()) for name in ("complete.json", "declaration.json")
    ]
    standing_gate(a.root, a.audit_root, variant=d["variant"], bank=d["course_bank"])
    review = json.loads((a.audit_root / "complete.json").read_text())
    pins = dict(review["input_file_hashes"])
    pins.update(
        {str(a.root / "complete.json"): a.hash, str(a.audit_root / "complete.json"): a.audit_hash}
    )
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            all_post_first_phases_required=True,
            first_kick_replayed=False,
            independent_whole_world_replay=False,
            hardware_authorized=False,
        ),
    )
    replay = None
    try:
        import numpy as np

        from rosclaw_soccer.rsi.paid_continuity_audit import snapshot_fields
        from rosclaw_soccer.rsi.paid_native_phase_replay import PaidNativePhaseReplay
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor

        auditor, _ = open_paid_auditor(pins, d["cases"])
        replay = PaidNativePhaseReplay(auditor)
        results = []

        def read(row):
            path = a.root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(a.root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("exact confined phase input required")
            pins[str(path)] = row["raw_hash"]
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        for i, world in enumerate(r["worlds"]):
            for repeat, artifacts in enumerate((world["artifacts"], world["repeat_artifacts"])):
                links = {Path(row["raw_path"]).stem: row for row in artifacts}
                phases = []
                for key, row in links.items():
                    if key.startswith("stand") or key.endswith("-settle"):
                        raw = read(row)
                        entry, end = (
                            snapshot_fields(raw, prefix) for prefix in ("initial_", "terminal_")
                        )
                    elif key.startswith("kick") and key[4:].isdigit() and key != "kick0":
                        k = int(key[4:])
                        raw = read(row)
                        boundary = read(links[f"between{k - 1}-{k}-boundary"])
                        entry = snapshot_fields(boundary, "after_")
                        end = read(links[f"kick-terminal{k}"])
                    else:
                        continue
                    result = replay.replay(raw, entry=entry, terminal=end)
                    phase = dict(
                        case=i, repeat=repeat, phase=key, raw_hash=row["raw_hash"], **result
                    )
                    write_once(a.output / f"case{i}-repeat{repeat}-{key}.json", phase)
                    phases.append(phase)
                results.extend(phases)
            print(json.dumps(dict(case=i, phases=len(phases))), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_POST_FIRST_ALL_PHASES_STRICT_NATIVE_REPLAY_PASS",
                input_file_hashes=pins,
                source_head=a.expected_head,
                results=results,
                own_native_steps=replay.native_steps,
                phases=replay.phases,
                all_q_v_forces_terminal_exact=True,
                first_kick_replayed=False,
                independent_whole_world_replay=False,
                optimizer_updates=0,
                own_neural_rows=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                own_native_steps=replay.native_steps if replay is not None else 0,
                phases=replay.phases if replay is not None else 0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
