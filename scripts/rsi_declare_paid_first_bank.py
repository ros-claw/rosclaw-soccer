"""Preflight exact original16 TRAIN/DEV fixtures for full first-enabled neural exams."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("source-root", "source-audit-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--bank", choices=("train", "dev"), required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused original-bank declaration and frozen source required")
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate

    standing_gate(a.source_root, a.source_audit_root, variant=1, bank=a.bank)
    source = json.loads((a.source_root / "complete.json").read_text())
    sd = json.loads((a.source_root / "declaration.json").read_text())
    audit = json.loads((a.source_audit_root / "complete.json").read_text())
    pins = dict(audit["input_file_hashes"])
    for f in (
        a.source_root / "complete.json",
        a.source_root / "declaration.json",
        a.source_audit_root / "complete.json",
    ):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    try:
        import numpy as np

        from rosclaw_soccer.rsi.paid_continuity_audit import identical, snapshot_fields
        from rosclaw_soccer.rsi.paid_first_bank import (
            SCHEMA,
            STATUS,
            original_case_with_measured_velocity,
        )
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

        bench, _, scene = open_paid_bench(pins, sd["cases"])
        cases, entries = [], []
        keys = {"qpos", "qvel", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "time"}
        for i, c in enumerate(sd["cases"]):
            link = source["worlds"][i]["kicks"][0]["result"]
            f = a.source_root / link["raw_path"]
            if (
                f.is_symlink()
                or not f.resolve().is_relative_to(a.source_root.resolve())
                or digest(f) != link["raw_hash"]
            ):
                raise ValueError("exact original initial-state witness required")
            pins[str(f)] = link["raw_hash"]
            with np.load(f, allow_pickle=False) as z:
                initial = snapshot_fields({k: z[k].copy() for k in z.files}, "initial_")
            ep = bench.make(c)
            before = snapshot(ep.env.data)
            if set(initial) != keys:
                raise ValueError("actual legacy seven-field schema required")
            identical(initial, {k: before[k] for k in keys})
            q, v = ep.layout.canonical(ep.env.data)
            clean = original_case_with_measured_velocity(
                c, i, v[35:37].tolist(), float(ep.env.model.body_mass[ep.layout.ball])
            )
            if not np.array_equal(q[36:39], clean["ball_world_initial"]):
                raise ValueError("original compiled position differs")
            explicit = bench.make(clean)
            identical(before, snapshot(explicit.env.data))
            cases.append(clean)
            entries.append(
                dict(
                    case=i,
                    original_raw_path=str(f),
                    original_raw_hash=link["raw_hash"],
                    position=q[36:39].tolist(),
                    velocity=v[35:37].tolist(),
                )
            )
        if bench.native_steps or bench.neural_rows or len(cases) != 16:
            raise ValueError("fixture declaration must execute neither physics nor neural policy")
        declaration = dict(
            schema=SCHEMA,
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_root=str(a.source_root),
            source_audit_root=str(a.source_audit_root),
            source_hash=digest(a.source_root / "complete.json"),
            source_audit_hash=digest(a.source_audit_root / "complete.json"),
            bank=a.bank,
            cases=cases,
            cases_predeclared=16,
            body_scene_hash=scene,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        write_once(a.output / "declaration.json", declaration)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=STATUS,
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                bank=a.bank,
                body_scene_hash=scene,
                cases_preflighted=16,
                entries=entries,
                original_seven_entry_fields_exact=True,
                explicit_velocity_preserves_full_native_entry=True,
                own_native_steps=0,
                own_neural_rows=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                own_native_steps=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
