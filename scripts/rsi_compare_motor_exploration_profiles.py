"""Read both fully audited paired amplitude interventions, never allocate physics."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("control-root", "treatment-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.motor_exploration_comparison import compare_motor_profiles
    from rosclaw_soccer.rsi.motor_exploration_curriculum import declared_motor_curriculum
    from rosclaw_soccer.rsi.motor_exploration_profile import declared_exploration_sigma
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean pinned readout source required")

    clean()
    if a.output.exists():
        raise ValueError("unused paired readout output required")
    roots = (a.control_root, a.treatment_root)
    records, declarations, proofs = [], [], []
    for root, profile in zip(roots, ("micro-v1", "contact-v1"), strict=True):
        paths = [
            root / "complete.json",
            root / "experience/complete.json",
            root / "audit/complete.json",
        ]
        stage, experience, audit = [json.loads(path.read_text()) for path in paths]
        declaration_path = root / "experience/declaration.json"
        d = json.loads(declaration_path.read_text())
        c = declared_motor_curriculum(d)
        declared_exploration_sigma(d)
        if (
            any(path.is_symlink() for path in paths + [declaration_path])
            or any((path.parent / "failed.json").exists() for path in paths)
            or stage["status"] != "FRESH_AR_GENERATION_COLLECTED_AND_AUDITED_NOT_TRAINED"
            or audit["status"] != c.audit_status
            or c.name != "retention-run64"
            or any(r["source_head"] != a.producer_head for r in (stage, experience, audit, d))
            or any(
                r["hardware_authorized"] is not False or r["promotion_authorized"] is not False
                for r in (stage, experience, audit, d)
            )
            or experience["declaration_hash"] != digest(declaration_path)
            or audit["source_hash"] != digest(paths[1])
            or audit["training_hash"] != experience["training_hash"]
            or stage["training_hash"] != experience["training_hash"]
            or stage["completed"]
            != [
                dict(stage="experience", hash=digest(paths[1])),
                dict(stage="audit", hash=digest(paths[2])),
            ]
            or audit["counts"]["native"] != experience["actual_native_steps"]
            or audit["counts"]["neural"] != experience["actual_neural_decisions"]
            or d.get("exploration_profile") != profile
            or d["sampling_law"] != "box-truncated-ar1"
            or len(audit["results"]) != c.worlds
        ):
            raise ValueError("exact closed own-native audited paired intervention required")
        for original, own in zip(experience["records"], audit["results"], strict=True):
            if (original["context"], original["seed"], original["healthy_quality"], False) != (
                own["context"],
                own["seed"],
                own["quality"],
                own["rejected"],
            ):
                raise ValueError("every own-world audit must agree with its original result")
        check_pins(audit["input_file_hashes"])
        declarations.append(d)
        records.append(experience["records"])
        proofs.append(
            dict(root=str(root), stage_hash=digest(paths[0]), audit_hash=digest(paths[2]))
        )
    for key in (
        "generation_index",
        "contexts",
        "seeds",
        "training_hash",
        "source_exam_hash",
        "rho",
    ):
        if declarations[0][key] != declarations[1][key]:
            raise ValueError("identical candidate, course, seeds and autoregressive law required")
    result = compare_motor_profiles(*records, generation=declarations[0]["generation_index"])
    clean()
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            schema="soccer.dual_motor.audited_amplitude_comparison.v1",
            source_head=a.expected_head,
            proofs=proofs,
            result=result,
            actual_new_physics_steps=0,
            actual_new_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
