"""Recover causal entry features/actual teacher plans; no physics or training."""

import argparse
import contextlib
import io
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("nominal-audit", "nominal-root", "perturbed-audit", "perturbed-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("nominal-hash", "perturbed-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("new memory output required")
    from scripts.rsi_verified_teacher_sources import admit_teacher_sources

    pins, sources = admit_teacher_sources(
        (
            ("nominal", a.nominal_audit, a.nominal_hash, a.nominal_root),
            ("perturbed", a.perturbed_audit, a.perturbed_hash, a.perturbed_root),
        )
    )
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique qualified source asset required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo

    from rosclaw_soccer.rsi.kick_entry_reconstruction import KickEntryReconstructor
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.sim.contracts import hash_json

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    state, output, policy_type, mapping = _load_robonaldo(
        one("/policy/robonaldo/FreeKick.py").parents[2]
    )
    with contextlib.redirect_stdout(io.StringIO()):
        policy = policy_type(state(29), output(29))
    reconstruct = KickEntryReconstructor(model, layout, policy, mapping)
    features, coefficients, active, expert_valid, nominal = [], [], [], [], []
    records = []
    for kind, root, producer, audit in sources:
        for i, result in enumerate(producer["results"]):
            ref = result["arms"]["C"]
            path = root / ref["record"]
            if (
                not path.resolve().is_relative_to(root.resolve())
                or digest(path) != ref["hash"]
                or pins.get(str(path)) != ref["hash"]
            ):
                raise ValueError("exact actual teacher record required")
            row = json.loads(path.read_text())
            if hash_json(row["outcome"]) != hash_json(audit["results"][i]["C"]):
                raise ValueError("teacher record outcome differs from independent audit")
            if row["entry_control"] is None:
                records.append(
                    dict(
                        kind=kind,
                        course_index=i,
                        entry_admitted=False,
                        feature_row=None,
                        source_record_hash=ref["hash"],
                    )
                )
                continue
            artifact = path.parent / row["artifact"]["path"]
            if pins.get(str(artifact)) != digest(artifact):
                raise ValueError("exact actual teacher native trace required")
            with np.load(artifact, allow_pickle=False) as raw:
                feature = reconstruct.reconstruct(row, dict(raw))
            # At the first real NN call both temporal lobes have EXACTLY zero
            # position/velocity offset, so this recovers the unmodified causal
            # 547 observation+current reference phase+actual remaining task time.
            changed = bool(row.get("changed_controls", 0))
            target = np.zeros((2, 6), np.float64)
            if changed:
                target = np.array(
                    row["controller_events"][0]["installation"]["coefficients"], dtype=np.float64
                )
            if (
                target.shape != (2, 6)
                or not np.isfinite(target).all()
                or abs(target).max() > 0.060001
            ):
                raise ValueError("actual bounded plan coefficients required")
            good = bool(row["outcome"]["task_pass"] and row["healthy_full_horizon"])
            records.append(
                dict(
                    kind=kind,
                    course_index=i,
                    entry_admitted=True,
                    feature_row=len(features),
                    source_record_hash=ref["hash"],
                    actual_teacher_plan_active=changed,
                    qualified_expert_target=good,
                    formal_outcome=row["outcome"],
                    whole_body_healthy=row["healthy_full_horizon"],
                )
            )
            features.append(feature)
            coefficients.append(target)
            active.append(changed)
            expert_valid.append(good)
            nominal.append(kind == "nominal")
    dataset = dict(
        features=np.stack(features),
        actual_plan_coefficients=np.stack(coefficients),
        actual_teacher_active=np.array(active, np.bool_),
        qualified_expert_target=np.array(expert_valid, np.bool_),
        nominal_training_fold=np.array(nominal, np.bool_),
    )
    local_qualified = sources[1][3]["qualified_on_local_perturbations"]
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    artifact = a.output / "entry-memory.npz"
    np.savez_compressed(artifact, **dataset)
    write_once(
        a.output / "complete.json",
        dict(
            status="CAUSAL_TEACHER_ENTRY_MEMORY_RECONSTRUCTED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            artifact=dict(path=artifact.name, hash=digest(artifact)),
            records=records,
            feature_rows=len(features),
            feature_count=549,
            feature_contract="original_measured547_plus_current_reference_phase_plus_actual_remaining_time",
            active_teacher_rows=sum(active),
            successful_expert_rows=sum(expert_valid),
            failed_teacher_rows_preserved_and_not_good_motor_targets=True,
            perturbation_rows_must_not_enter_initial_supervised_training=True,
            local_physical_teacher_qualified=local_qualified,
            student_training_prerequisites_satisfied=local_qualified,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            private_fresh_accessed=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
