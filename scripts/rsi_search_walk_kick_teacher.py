"""Bounded public16 teacher search; each shard preserves every legal failure."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--shard", type=int, choices=range(4), required=True)
    p.add_argument("--experimental-approach-config", type=Path)
    p.add_argument("--experimental-approach-config-hash")
    a = p.parse_args()
    if (a.experimental_approach_config is None) != (a.experimental_approach_config_hash is None):
        p.error("explicit experimental config and hash must be provided together")
    from rosclaw_soccer.rsi.walk_kick_teacher_plan import teacher_courses
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_probe_walk_to_kick import authenticated_case_pins

    clean(a.expected_head)
    courses = teacher_courses()
    bank_hash = preflight_receiving_courses(courses)  # ENTIRE16 before any NN/model allocation
    path = a.source / "complete.json"
    if a.output.exists() or path.is_symlink() or digest(path) != a.source_hash:
        raise ValueError("new output and exact completed source required")
    source = json.loads(path.read_text())
    if (
        source["status"] != "NATIVE8_B_AND_B_PLUS_C_CONTACT_PILOT_MEASURED_NOT_LEARNED"
        or source["promotion_authorized"] is not False
        or len(source["results"]) != 5
    ):
        raise ValueError("closed five-case original asset lineage required")
    pins = authenticated_case_pins(a.source, source)
    pins[str(path)] = a.source_hash
    fast_assets = {}
    fast = a.experimental_approach_config is not None
    if fast:
        from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence

        fast_assets, fast_pins = admit_heading_evidence(
            a.experimental_approach_config,
            a.experimental_approach_config_hash,
        )
        for key, value in fast_pins.items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting measured locomotion ancestry")
            pins[key] = value
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique qualified asset required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode
    from rosclaw_soccer.rsi.walk_kick_teacher_plan import (
        candidate_parameters,
        elite_candidates,
        outcome_rank,
        search_bounds,
        uniform_candidates,
    )

    scene = one("/cached-standard/declared-scene.xml")
    velocity_model = one("/velocity/v0/exported/policy.onnx")
    velocity_config = one("/velocity/v0/params/deploy.yaml")
    assets = dict(
        velocity_model=velocity_model,
        velocity_config=velocity_config,
        velocity_model_hash=pins[str(velocity_model)],
        velocity_config_hash=pins[str(velocity_config)],
    )
    assets.update(fast_assets)
    types = _load_robonaldo(one("/policy/robonaldo/FreeKick.py").parents[2])
    model = mujoco.MjModel.from_xml_string(shared_native_scene(scene.read_text(), prefixes=("a/",)))
    baseline_parameters = dict(
        speed_mps=1.0,
        entry_gap_m=0.9,
        reference_frame=None,
        root_lateral_offset_m=0.0,
    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.public16_walk_kick_teacher_fast_cem.v2"
            if fast
            else "soccer.public16_walk_kick_teacher_cem.v1",
            experimental_fast_approach=fast,
            source_head=a.expected_head,
            input_file_hashes=pins,
            course_bank_hash=bank_hash,
            courses=[
                dict(index=i, agent=c.agent_id, seed=c.seed, course=c.course)
                for i, c in enumerate(courses)
            ],
            shard=a.shard,
            assigned_indices=list(range(a.shard, 16, 4)),
            candidate_bounds=search_bounds(experimental_fast=fast).tolist(),
            generations=2,
            candidates_per_generation=8,
            elites=2,
            conditional_normal_min_std_fraction=0.15,
            candidate_evaluations_per_course=16,
            independent_parameterizations_not_nn_updates=True,
            baseline=baseline_parameters,
            horizon_controls=360,
            formal_horizon_controls=300,
            diagnostic_tail_participates_in_formal_scoring=False,
            entry_controls=[50, 239],
            entry_lateral_abs_max_m=0.25,
            approach_command_slew=0.05,
            ball_launch_spin="vyaw0, omega_y=vx/.11",
            ranking="lexicographic healthy360_and_safe, pass, unchanged_reward, earlier_touch",
            baseline_and_selected_fresh_nn_reexecution=True,
            maximum_worlds=76,
            maximum_native_steps=273600,
            original_exam_body_identity_claimed=False,
            original52_regraded=False,
            privileged_search=True,
            public_development_only=True,
            sealed_fresh_accessed=False,
            optimizer_updates=0,
            physical_teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    results, native_steps = [], 0
    try:
        for index in range(a.shard, 16, 4):
            c = courses[index]
            root = a.output / f"course-{index:02d}"
            root.mkdir()
            rows, files = {}, {}

            def evaluate(name, params, *, c=c, root=root, index=index, rows=rows, files=files):
                nonlocal native_steps
                row, arrays = evaluate_episode(
                    model=model,
                    assets=assets,
                    policy_types=types,
                    course=c.course,
                    parameters=params,
                    hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
                )
                native_steps += row["native_steps"]
                artifact = root / (name + ".npz")
                np.savez_compressed(artifact, **arrays)
                row.update(
                    name=name,
                    course_index=index,
                    artifact=dict(path=artifact.name, hash=digest(artifact)),
                )
                rows[name], files[name] = row, artifact
                write_once(root / (name + ".json"), row)
                return row

            first = uniform_candidates(c.seed, experimental_fast=fast)
            write_once(
                root / "generation-one-declaration.json",
                dict(
                    course=c.course,
                    seed=c.seed,
                    parameter_vectors=first.tolist(),
                    baseline=baseline_parameters,
                    search_outcomes_observed=False,
                ),
            )
            base = evaluate("baseline", baseline_parameters)
            generation_one = [
                evaluate(f"candidate-{i:02d}", candidate_parameters(x, experimental_fast=fast))
                for i, x in enumerate(first)
            ]
            second = elite_candidates(
                first,
                tuple(outcome_rank(row) for row in generation_one),
                seed=c.seed + 1000,
                experimental_fast=fast,
            )
            write_once(
                root / "generation-two-declaration.json",
                dict(
                    seed=c.seed + 1000,
                    parameter_vectors=second.tolist(),
                    first_generation_record_hashes={
                        f"candidate-{i:02d}": digest(root / f"candidate-{i:02d}.json")
                        for i in range(8)
                    },
                    second_generation_outcomes_observed=False,
                ),
            )
            for i, x in enumerate(second, 8):
                evaluate(f"candidate-{i:02d}", candidate_parameters(x, experimental_fast=fast))
            # Include the measured baseline as a retention fallback, never
            # manufacture net growth by dropping an already successful parent.
            eligible = ["baseline", *[f"candidate-{i:02d}" for i in range(16)]]
            selected = max(eligible, key=lambda name: outcome_rank(rows[name]))
            fresh_maps = {}
            for label, original in (
                ("baseline-reexecution", "baseline"),
                ("selected-reexecution", selected),
            ):
                evaluate(label, rows[original]["parameters"])
                with np.load(files[original]) as old, np.load(files[label]) as new:
                    exact = {key: bool(np.array_equal(old[key], new[key])) for key in old.files}
                if not all(exact.values()):
                    raise ValueError("fresh neural reexecution changed physical arrays")
                fresh_maps[label] = exact
            summary = dict(
                course_index=index,
                course=c.course,
                selected=selected,
                baseline_outcome=base["outcome"],
                selected_outcome=rows[selected]["outcome"],
                selected_healthy_full_horizon=rows[selected]["healthy_full_horizon"],
                fresh_reexecution_exact=fresh_maps,
                parameter_search_only_not_deployment_policy=True,
                raw_record_hashes={name: digest(root / (name + ".json")) for name in rows},
                worlds=len(rows),
                native_steps=sum(row["native_steps"] for row in rows.values()),
                physical_teacher_qualified=False,
                promotion_authorized=False,
            )
            write_once(root / "complete.json", summary)
            results.append(
                dict(
                    course_index=index, source_hash=digest(root / "complete.json"), summary=summary
                )
            )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PUBLIC16_TEACHER_CEM_SHARD_MEASURED_NOT_TRAINED",
                experimental_fast_approach=fast,
                source_head=a.expected_head,
                input_file_hashes=pins,
                course_bank_hash=bank_hash,
                results=results,
                worlds=sum(r["summary"]["worlds"] for r in results),
                native_steps=native_steps,
                optimizer_updates=0,
                physical_teacher_qualified=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                native_steps=native_steps,
                completed_courses=len(results),
                no_automatic_retry=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
