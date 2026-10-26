"""Five fixed ball starts, paired full H300 native episodes; no policy training."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

CASES = (
    ("nominal", (1.0, 0.0)),
    ("x-minus", (0.98, 0.0)),
    ("x-plus", (1.02, 0.0)),
    ("y-minus", (1.0, -0.02)),
    ("y-plus", (1.0, 0.02)),
)


def execute(player):
    """One actual native episode, original NN capture, no reset or private search."""
    original = player.policy.ort_session
    observations, actions, rows, targets, decoded = [], [], [], [], []
    initial_qpos = player.world.data.qpos.copy()
    initial_qvel = player.world.data.qvel.copy()

    class Capture:
        def run(self, outputs, inputs):
            if len(observations) >= 300 or inputs["obs"].shape != (1, 1244):
                raise ValueError("finite declared NN budget required")
            result = original.run(outputs, inputs)
            observations.append(inputs["obs"].copy())
            actions.append(result[0].copy())
            return result

    player.policy.ort_session = Capture()
    failure = None
    try:
        for _ in range(300):
            target, measured = player.propose()
            row = player.world.advance({player.prefix: target})
            rows.append(row)
            targets.append(target)
            decoded.append(measured["nn_decoded_target"])
            player.commit()
            if player.world.failed:
                break
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    arrays = {
        key: np.concatenate([r[key] for r in rows]) if rows else np.empty(shape)
        for key, shape in [
            ("qpos", (0, 43)),
            ("qvel", (0, 41)),
            ("torque", (0, 29)),
            ("ball_forces", (0, 1, 3)),
        ]
    }
    arrays.update(
        joint_targets=np.array(targets).reshape(-1, 29),
        nn_decoded_targets=np.array(decoded).reshape(-1, 29),
        neural_obs=np.array(observations).reshape(-1, 1, 1244),
        neural_actions=np.array(actions).reshape(-1, 1, 29),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
    )
    label = dict(event_status="INCOMPLETE_NATIVE_EPISODE", task_pass=False, safe=False)
    if len(rows) == 300 and failure is None:
        from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch

        grouped = np.zeros((300, 1, 6))
        grouped[:, 0, :3] = arrays["ball_forces"][:, 0].reshape(300, 10, 3).max(axis=1)
        q = arrays["qpos"].reshape(300, 10, 43)
        lo = player.layout
        label = label_first_touch(
            dict(
                force_n=grouped,
                pelvis_z_per_substep_m=q[:, :, lo.rq + 2][:, None],
                ball_position_after_step_m=q[:, -1, lo.bq : lo.bq + 3][:, None],
            )
        ).to_dict()
    return arrays, dict(
        controls=len(rows),
        native_steps=player.world.native_steps,
        neural_calls=len(observations),
        runtime_failure=failure,
        body_failed=player.world.failed,
        outcome=label,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readback", type=Path, required=True)
    p.add_argument("--readback-hash", required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--replay-hash", required=True)
    p.add_argument("--nominee", type=Path, required=True)
    p.add_argument("--parent-baseline", type=Path, required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
    from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial, validate_cold_ball_xy
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_cftrack_native_context import build_cold_cftrack
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if (
        a.output.exists()
        or a.readback.is_symlink()
        or a.replay.is_symlink()
        or digest(a.readback) != a.readback_hash
        or digest(a.replay) != a.replay_hash
    ):
        raise ValueError("unused paired bank and exact completed NN/native qualifications required")
    audit = json.loads(a.readback.read_text())
    replay = json.loads(a.replay.read_text())
    if (
        audit["status"] != "CFTRACK_ACTUAL_NN1244_HISTORY_AND_TARGETS_VERIFIED"
        or audit["actual_neural_readbacks"] != 600
        or audit["h300_scoring_of_new_cold_entry_protocol"]["task_pass"] is not True
        or replay["status"] != "CFTRACK_FULL_NATIVE_REPLAY_EXACT"
        or replay["compared_controls"] != 600
        or replay["readback_hash"] != a.readback_hash
    ):
        raise ValueError("nominal complete physical task/NN/native evidence required")
    pins = dict(replay["input_file_hashes"])
    pins[str(a.replay)] = a.replay_hash
    complete = a.nominee / "complete.json"
    parent_complete = a.parent_baseline / "complete.json"
    if (
        complete.is_symlink()
        or parent_complete.is_symlink()
        or digest(complete) != audit["qualification_hash"]
        or pins.get(str(complete)) != digest(complete)
        or pins.get(str(parent_complete)) != digest(parent_complete)
    ):
        raise ValueError("same source-pinned candidate and original parent required")
    nominee = json.loads(complete.read_text())
    parent = json.loads(parent_complete.read_text())
    raw = nominee["motor_plan"]
    frozen = CFtrackMotorResidualPlan(
        raw["origin_counter"],
        raw["start_control"],
        raw["peak_control"],
        tuple(raw["delta_hg"]),
        raw["donor_policy_hash"],
    )
    frozen.offsets()
    if len(CASES) != 5 or len({name for name, _ in CASES}) != 5:
        raise ValueError("five distinct preregistered initial ball cases required")
    for _, xy in CASES:
        validate_cold_ball_xy(xy)
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("source-qualified external policy required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique qualified native body required")
    a.output.mkdir()
    parameters = asdict(frozen)
    parameters.pop("donor_policy_hash")
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            cases=CASES,
            arms=["parent", "frozen-motor"],
            official_horizon=300,
            maximum_actual_native_steps=30000,
            maximum_neural_calls=3000,
            private_search_steps=0,
            neural_optimizer_updates=0,
            frozen_motor_parameters=parameters,
            frozen_motor_parameter_hash=hash_json(parameters),
            donor_hash_rebound_per_fresh_episode_not_optimization=True,
            scope="centimeter_static_ball_development_perturbations_not_dynamic_or_generalized_teacher",
            one_worker_author_default_ort_threads=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    template, _, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    before = cftrack_state_hash(template.policy)
    records = []
    nominal_arrays = {}
    for name, xy in CASES:
        for arm in ["parent", "frozen-motor"]:
            player = cold_cftrack_trial(template, ball_xy=xy)
            reference = condition_direct_kick_approach(player.policy)
            binding = cftrack_state_hash(player.policy)
            plan = None if arm == "parent" else replace(frozen, donor_policy_hash=binding)
            if plan is not None:
                player.install_motor_plan(plan)
            arrays, metadata = execute(player)
            path = a.output / (name + "-" + arm + ".npz")
            np.savez_compressed(path, **arrays)
            pins[str(path)] = digest(path)
            row = dict(
                case=name,
                ball_xy=xy,
                arm=arm,
                reference_conditioning=reference,
                motor_plan=asdict(plan) if plan else None,
                cold_policy_hash=binding,
                artifact=dict(path=path.name, hash=pins[str(path)]),
                **metadata,
            )
            write_once(a.output / (name + "-" + arm + ".json"), row)
            records.append(row)
            if name == "nominal":
                nominal_arrays[arm] = arrays
            if cftrack_state_hash(template.policy) != before or template.world.native_steps != 0:
                raise ValueError("trial altered never-executed template")
    # Only after all ten actual episodes, compare nominal source trajectories.
    for arm, root, producer in [
        ("parent", a.parent_baseline, parent),
        ("frozen-motor", a.nominee, nominee),
    ]:
        path = root / producer["artifact"]["path"]
        if path.is_symlink() or pins.get(str(path)) != digest(path):
            raise ValueError("bound source raw required after execution")
        with np.load(path, allow_pickle=False) as old:
            if any(
                not np.array_equal(nominal_arrays[arm][k], old[k][: len(nominal_arrays[arm][k])])
                for k in [
                    "qpos",
                    "qvel",
                    "torque",
                    "ball_forces",
                    "joint_targets",
                    "neural_obs",
                    "neural_actions",
                ]
            ):
                raise ValueError(
                    "new cold episode initialization does not reproduce nominal source prefix"
                )
    pairs = [records[i : i + 2] for i in range(0, 10, 2)]
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_STATIC_PERTURBATION_BANK_RECORDED_UNVERIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            records=records,
            parent_task_passes=sum(r[0]["outcome"]["task_pass"] for r in pairs),
            candidate_task_passes=sum(r[1]["outcome"]["task_pass"] for r in pairs),
            rescues=[
                r[0]["case"]
                for r in pairs
                if not r[0]["outcome"]["task_pass"] and r[1]["outcome"]["task_pass"]
            ],
            regressions=[
                r[0]["case"]
                for r in pairs
                if r[0]["outcome"]["task_pass"] and not r[1]["outcome"]["task_pass"]
            ],
            actual_native_steps=sum(r["native_steps"] for r in records),
            neural_calls=sum(r["neural_calls"] for r in records),
            nominal_both_arms_source_prefix_exact=True,
            template_unexecuted_unchanged=True,
            independent_perturbation_nn_readback_not_yet_done=True,
            independent_perturbation_native_replay_not_yet_done=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            commercial_publicity_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
