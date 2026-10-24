"""Authenticate every seeded physics/replay pair and retain causal feedback.

No physics, policy execution, optimization or promotion. Ordinary falls are
boundary examples, never discarded or disguised as collector exceptions.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "core-source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("manifest-hash", "core-expected-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or a.manifest.is_symlink() or digest(a.manifest) != a.manifest_hash:
        raise ValueError("unused output and exact family manifest required")
    manifest = json.loads(a.manifest.read_text())
    plan_path = Path(manifest["plan"])
    if digest(plan_path) != manifest["plan_hash"]:
        raise ValueError("exact preregistered seed family required")
    plan = json.loads(plan_path.read_text())
    pairs = manifest["pairs"]
    if (
        manifest["schema"] != "soccer.correlated_torque_paired_family.v1"
        or [row["seed"] for row in pairs]
        != [plan["zero_exploration"]["seed"], *plan["exploration_seeds"]]
        or len(pairs) != 9
        or manifest["core_commit"] != a.core_expected_head
    ):
        raise ValueError("entire exact zero plus eight-seed family required")
    pins = {str(a.manifest): a.manifest_hash, str(plan_path): manifest["plan_hash"]}

    def admit(root, expected):
        path = root / "complete.json"
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact completed evidence required")
        report = json.loads(path.read_text())
        for key, value in report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting ancestry")
            pins[key] = value
        pins[str(path)] = expected
        return report

    def artifact(root, entry):
        path = root / entry["path"]
        if (
            path.is_symlink()
            or path.parent.resolve() != root.resolve()
            or digest(path) != entry["hash"]
        ):
            raise ValueError("confined immutable raw artifact required")
        pins[str(path)] = entry["hash"]
        return path

    baseline_root = Path(manifest["baseline"])
    baseline = admit(baseline_root, manifest["baseline_hash"])
    for row in baseline["records"]:
        artifact(baseline_root, row["artifact"])
    reports = []
    for pair in pairs:
        root, own_root = Path(pair["source"]), Path(pair["own"])
        source, own = admit(root, pair["source_hash"]), admit(own_root, pair["own_hash"])
        declaration_path = root / "declaration.json"
        declaration = json.loads(declaration_path.read_text())
        pins[str(declaration_path)] = digest(declaration_path)
        if (
            source["source_head"] != manifest["source_commit"]
            or not own["all_exact"]
            or own["source_hash"] != pair["source_hash"]
            or len(source["records"]) != 4
            or len(own["records"]) != 4
            or declaration["protocol"] != "robonaldo.native_torque_exploration.v1"
            or declaration["exploration_plan_hash"] != manifest["plan_hash"]
            or declaration["exploration_config"]["seed"] != pair["seed"]
            or declaration["exploration_config"]["scale"] != pair["scale"]
            or declaration["torque_intervention_fraction"] != 1.0
        ):
            raise ValueError("exact paired declared physical exploration required")
        for index, row in enumerate(source["records"]):
            proof = own["records"][index]
            if (
                row["index"] != index
                or row["arm"] != "single"
                or not proof["all_exact"]
                or not proof["exploration_rng_and_behavior_exact"]
                or not all(proof["guarded_proposal_exact"].values())
                or not all(proof["neural_episode_exact"].values())
            ):
                raise ValueError(
                    "entire independently replayed body/proposal/RNG identity required"
                )
            ep = row["torque_execution_artifacts"]["a/"]
            if ep["policy_receipt"]["artifact_hash"] != plan["model_hash"]:
                raise ValueError("exploration must retain the preregistered BC weights")
            artifact(own_root, proof["artifact"])
            for entry in (row["artifact"], ep, ep["guarded_proposal_trace"], ep["exploration_rng"]):
                artifact(root, entry)
        reports.append((pair, source, declaration))
    pin_active_core(a.core_source_root, a.core_expected_head, pins)
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.native_kick_reward import finite_returns, native_kick_rewards
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.torque_constraint_audit import audit_applied_torques
    from rosclaw_soccer.sim.contracts import hash_json

    scenes = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one canonical body/ball scene required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    body_hash = hash_json(layout.profile())
    hard = np.asarray(G1_HARD_TORQUE_LIMITS)
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="correlated_exploration_feedback_bank.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            manifest_hash=a.manifest_hash,
            training_indices=[0, 3],
            validation_indices=[1, 2],
            native_reward_contract="native_kick_reward.v1",
            gamma=0.9995,
            actor_updates=0,
            hardware_authorized=False,
        ),
    )
    records, source_steps = [], 0
    try:
        for pair, report, declaration in reports:
            root = Path(pair["source"])
            for row in report["records"]:
                index = row["index"]
                entry = row["torque_execution_artifacts"]["a/"]
                with (
                    np.load(root / row["artifact"]["path"]) as raw,
                    np.load(root / entry["path"]) as ep,
                    np.load(root / entry["guarded_proposal_trace"]["path"]) as trace,
                ):
                    n = row["native_steps"]
                    if (
                        not 0 < n <= 6520
                        or row["controls"] * 10 != n
                        or raw["qpos"].shape[0] != n
                        or ep["observations"].shape != (n, 102)
                        or entry["body_hash"] != body_hash
                        or not np.array_equal(trace["applied_actions"], raw["torque"][:, layout.ai])
                        or not np.array_equal(
                            ep["actions"], trace["applied_actions"].astype(np.float32)
                        )
                        or not np.all(trace["intervention_fractions"] == 1)
                    ):
                        raise ValueError("exact actual native force/body/time identity required")
                    before_v = np.vstack(
                        (ep["observations"][0, 29:58], raw["qvel"][:-1, layout.vi])
                    )
                    constraints = audit_applied_torques(
                        trace["applied_actions"],
                        trace["parent_actions"],
                        before_v,
                        hard,
                        guard_scale=0.85,
                        parent_deviation_ratio=0.08,
                        delta_ratio=0.08,
                        maximum_power_w=2500.0,
                    )
                    if any(
                        value for key, value in constraints.items() if key.endswith("violations")
                    ):
                        raise ValueError(
                            "actual applied constraints violated; preserve as fatal contract evidence"
                        )
                    initial_ball = [*declaration["starts"][index], 0.11]
                    feedback = native_kick_rewards(
                        ball_positions=np.vstack(
                            (initial_ball, raw["qpos"][:, layout.bq : layout.bq + 3])
                        ),
                        pelvis_after=raw["qpos"][:, layout.rq : layout.rq + 7],
                        joint_velocity_after=raw["qvel"][:, layout.vi],
                        applied_torque=trace["applied_actions"],
                        hard_limits=hard,
                        foot_contact=np.max(raw["ball_forces"][:, 0, :2], axis=1) > 1,
                        illegal_contact=raw["ball_forces"][:, 0, 2] > 1,
                        used_parent=trace["used_parent"],
                    )
                    plane = row["actual_goal_plane"]
                    measured = feedback["crossing_error_m"]
                    if (plane is None) != (measured is None) or (
                        plane is not None
                        and abs(plane["error_to_fixed_target_m"] - measured) > 1e-10
                    ):
                        raise ValueError("actual causal crossing disagrees with source")
                    zero_identity = None
                    if pair["scale"] == 0:
                        ref = baseline["records"][index]["artifact"]
                        with np.load(baseline_root / ref["path"]) as old:
                            zero_identity = {
                                key: bool(np.array_equal(raw[key], old[key])) for key in old.files
                            }
                        if not all(zero_identity.values()):
                            raise ValueError("zero exploration changed previous BC physics")
                    values = {
                        name: feedback[name]
                        for name in ("rewards", "fall_costs", "constraint_costs")
                    }
                    values.update(
                        {
                            name + "_returns": finite_returns(value, gamma=0.9995)
                            for name, value in list(values.items())
                        }
                    )
                    values["actor_proposal_valid"] = trace["actor_proposal_valid"].copy()
                    values["guard_used_parent"] = trace["used_parent"].copy()
                    values["illegal_contact_after"] = raw["ball_forces"][:, 0, 2] > 1
                    path = a.output / f"feedback-{pair['seed']}-{index}.npz"
                    np.savez_compressed(path, **values)
                    hit = (
                        measured is not None
                        and measured <= 0.1
                        and row["healthy_full_horizon"]
                        and not row["illegal_body_ball_contact"]
                    )
                    record = dict(
                        seed=pair["seed"],
                        scale=pair["scale"],
                        index=index,
                        split="train" if index in (0, 3) else "validation",
                        actual_proposals=int(trace["actor_proposal_valid"].sum()),
                        native_steps=n,
                        healthy_full_horizon=row["healthy_full_horizon"],
                        ordinary_body_failure_retained=any(row["body_failed"].values()),
                        goal_error_m=measured,
                        strict01_hit=hit,
                        baseline_goal_error_m=baseline["records"][index]["actual_goal_plane"][
                            "error_to_fixed_target_m"
                        ],
                        zero_scale_identity=zero_identity,
                        causal_crossing_native_index=feedback["crossing_index"],
                        actual_constraints=constraints,
                        source=str(root),
                        raw_artifact=row["artifact"],
                        episode_artifact=entry,
                        feedback_artifact=dict(path=path.name, hash=digest(path)),
                    )
                    records.append(record)
                    source_steps += n
                    write_once(a.output / f"record-{pair['seed']}-{index}.json", record)
        check_pins(pins)
        stochastic = [row for row in records if row["scale"] > 0]
        write_once(
            a.output / "complete.json",
            dict(
                status="CORRELATED_NATIVE_EXPLORATION_FEEDBACK_BANK_REVIEWED_NOT_PROMOTED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                manifest_hash=a.manifest_hash,
                records=records,
                stochastic_trials=len(stochastic),
                distinct_initial_geometries=4,
                stochastic_healthy=sum(row["healthy_full_horizon"] for row in stochastic),
                stochastic_body_failures=sum(
                    row["ordinary_body_failure_retained"] for row in stochastic
                ),
                stochastic_strict01_hits=sum(row["strict01_hit"] for row in stochastic),
                all_zero_scale_identity=True,
                all_paired_replays_exact=True,
                source_native_steps=source_steps,
                own_native_steps=source_steps,
                optimizer_updates=0,
                new_physics_steps=0,
                successor_ready_for_new_role_unknown=True,
                single_training_success_not_qualified_teacher=True,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), completed=len(records), new_physics_steps=0),
        )
        raise


if __name__ == "__main__":
    main()
