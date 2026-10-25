"""Native paired backend study, no new policy learning or proposal optimization."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-audit", "experimental-approach-config", "fidelity", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "source-audit-hash",
        "experimental-approach-config-hash",
        "fidelity-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--shard", type=int, choices=range(4), required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence
    from rosclaw_soccer.rsi.walk_kick_teacher_plan import search_course_bank
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    courses = search_course_bank(expanded_public=True)
    bank_hash = preflight_receiving_courses(courses)
    if (
        a.output.exists()
        or a.source_audit.is_symlink()
        or (
            digest(a.source_audit) != a.source_audit_hash
            or a.fidelity.is_symlink()
            or digest(a.fidelity) != a.fidelity_hash
        )
    ):
        raise ValueError("unused output and exact closed source/fidelity evidence required")
    source, fidelity = (json.loads(p.read_text()) for p in (a.source_audit, a.fidelity))
    if (
        source["status"]
        != "PUBLIC40_CEM_ALL760_RAW_LABELS_AND_REEXECUTION_AUTHENTICATED_NOT_PROMOTED"
        or source["course_bank_hash"] != bank_hash
        or [r["course_index"] for r in source["results"]] != list(range(40))
        or source["promotion_authorized"] is not False
        or fidelity["status"] != "DIFFERENTIABLE_KICK_PRIOR_NUMERIC_FIDELITY_CHECKED_NOT_TRAINED"
        or fidelity["numeric_fidelity_pass"] is not True
        or fidelity["actor_optimizer_updates"] != 0
    ):
        raise ValueError("entire public40 and untrained numeric-fidelity prerequisite required")
    fast_assets, point_pins = admit_heading_evidence(
        a.experimental_approach_config, a.experimental_approach_config_hash
    )
    pins = dict(source["input_file_hashes"])
    for name, h in {**fidelity["input_file_hashes"], **point_pins}.items():
        if name in pins and pins[name] != h:
            raise ValueError("conflicting physical/source identity")
        pins[name] = h
    pins.update({str(a.source_audit): a.source_audit_hash, str(a.fidelity): a.fidelity_hash})
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique bound physical asset required: " + suffix)
        return paths[0]

    selected = {}
    for index, c in enumerate(courses):
        summary = source["results"][index]
        suffix = f"/course-{index:02d}/{summary['selected']}.json"
        rows = [(Path(k), json.loads(Path(k).read_text())) for k in pins if k.endswith(suffix)]
        matches = [(p, r) for p, r in rows if r["course"] == list(c.course)]
        if len(matches) != 1:
            raise ValueError("unique course-matched source program required")
        rp, row = matches[0]
        artifact = rp.parent / row["artifact"]["path"]
        if (
            artifact.parent.resolve() != rp.parent.resolve()
            or artifact.is_symlink()
            or (digest(artifact) != row["artifact"]["hash"])
        ):
            raise ValueError("exact original native trajectory required")
        selected[index] = row, artifact
    import mujoco
    import numpy as np
    import torch
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.kick_prior_actor import torch_kick_policy_type
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode
    from rosclaw_soccer.sim.contracts import hash_json

    torch.set_num_threads(1)
    vm, vc = one("/velocity/v0/exported/policy.onnx"), one("/velocity/v0/params/deploy.yaml")
    assets = dict(
        velocity_model=vm,
        velocity_config=vc,
        velocity_model_hash=pins[str(vm)],
        velocity_config_hash=pins[str(vc)],
        **fast_assets,
    )
    original_types = _load_robonaldo(one("/policy/robonaldo/FreeKick.py").parents[2])
    state, output, policy, mapping = original_types
    torch_types = (
        state,
        output,
        torch_kick_policy_type(policy, one("/policy/robonaldo/model/policy-obs-aic.onnx")),
        mapping,
    )
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.public40_frozen_kick_backend_pair.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            assigned_indices=list(range(a.shard, 40, 4)),
            worlds=20,
            native_step_ceiling=72000,
            shard=a.shard,
            source_parameters_unchanged=True,
            onnx_must_exactly_reproduce_saved_source=True,
            paired_arms=["onnx", "torch"],
            horizon_controls=360,
            formal_horizon_controls=300,
            no_success_loss_required=True,
            no_safety_regression_required=True,
            actor_optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    results, steps = [], 0
    for index in range(a.shard, 40, 4):
        source_row, source_arrays = selected[index]
        root = a.output / f"course-{index:02d}"
        root.mkdir()
        rows = {}
        for arm, types in (("onnx", original_types), ("torch", torch_types)):
            row, arrays = evaluate_episode(
                model=model,
                assets=dict(assets, experimental_fast=source_row["experimental_fast_approach"]),
                policy_types=types,
                course=courses[index].course,
                parameters=source_row["parameters"],
                hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
            )
            steps += row["native_steps"]
            if hash_json(
                relabel_arrays(
                    arrays,
                    controls=row["controls"],
                    ball_address=row["ball_qpos_address"],
                    root_address=row["robot_qpos_address"],
                )
            ) != hash_json(row["outcome"]):
                raise ValueError("native outcome label failed readback")
            if arm == "onnx":
                with np.load(source_arrays) as old:
                    if any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                        raise ValueError("unchanged ONNX backend failed original exact replay")
            artifact = root / (arm + ".npz")
            np.savez_compressed(artifact, **arrays)
            row.update(
                arm=arm,
                course_index=index,
                artifact=dict(path=artifact.name, hash=digest(artifact)),
            )
            write_once(root / (arm + ".json"), row)
            rows[arm] = row
        results.append(dict(course_index=index, rows=rows))
        write_once(root / "complete.json", results[-1])
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="PUBLIC40_FROZEN_KICK_BACKEND_SHARD_EXECUTED_NOT_TRAINED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=results,
            native_steps=steps,
            worlds=20,
            actor_optimizer_updates=0,
            physical_football_gain_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
