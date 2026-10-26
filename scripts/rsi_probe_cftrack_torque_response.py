"""Five current-state torque models, one positive probe and one proposal each."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or a.review.is_symlink() or digest(a.review) != a.review_hash:
        raise ValueError("unused response trial and independent torque review required")
    proof = json.loads(a.review.read_text())
    if (
        proof["status"] != "TORQUE_INTENT_INDEPENDENT_NN_AND_MANUAL_NATIVE_EXACT"
        or proof["new_neural_calls"] != 320
        or proof["new_native_steps"] != 3200
    ):
        raise ValueError("complete zero/nonzero independent NN/native qualification required")
    pins = dict(proof["input_file_hashes"])
    pins[str(a.review)] = a.review_hash
    bank_path = a.bank / "complete.json"
    if bank_path.is_symlink() or pins.get(str(bank_path)) != digest(bank_path):
        raise ValueError("bound independently verified five-case development bank required")
    bank = json.loads(bank_path.read_text())
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("qualified external original NN source required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(s) for s in pins if s.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique qualified canonical body required")
    from rosclaw_soccer.rsi.cold_cftrack_trial import validate_cold_ball_xy

    cases = (
        ("nominal", (1.0, 0.0)),
        ("x-minus", (0.98, 0.0)),
        ("x-plus", (1.02, 0.0)),
        ("y-minus", (1.0, -0.02)),
        ("y-plus", (1.0, 0.02)),
    )
    for _, xy in cases:
        validate_cold_ball_xy(xy)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.cftrack.torque_response_development.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            cases=cases,
            maximum_original_nn_calls=2400,
            maximum_native_steps=24000,
            maximum_arms_per_case=3,
            positive_probe_axis_hg=7,
            positive_probe_effort_fraction=0.125,
            candidate_effort_fraction_bounds=[0, 0.25],
            envelope_extra_rate_max_fraction_s=2.0,
            native_total_torque_caps_unchanged=True,
            old_position_residual_protocol_unchanged=True,
            new_torque_units_not_inheriting_old_point25rad_authority=True,
            planning_minimum_forward_m=1.2,
            planning_maximum_direction_ratio=0.2,
            cost_lateral_weight=4.0,
            cost_l2=0.1,
            cost_positive_l1=0.25,
            phase_from_own_current_parent_first_contact=True,
            one_current_probe_then_one_model_proposal=True,
            no_amplitude_or_seed_grid=True,
            development_not_fresh=True,
            full_h300_not_evaluated=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
    from rosclaw_soccer.rsi.cftrack_motor_authority import summarize_motor_future
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cftrack_torque_response import propose_torque_response
    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial, run_cold_torque_trial
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_cftrack_native_context import build_cold_cftrack

    template, _, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    template_hash = cftrack_state_hash(template.policy)
    records = []
    parents = {}
    nn_calls = native_steps = reserved = 0

    def new_player(xy):
        player = cold_cftrack_trial(template, ball_xy=xy)
        condition_direct_kick_approach(player.policy)
        return player

    def execute(name, arm, player, plan):
        nonlocal nn_calls, native_steps, reserved
        if reserved + 1600 > 24000:
            raise ValueError("fixed experiment budget exhausted; no refunds")
        reserved += 1600
        write_once(
            a.output / (name + "-" + arm + "-before.json"),
            dict(plan=asdict(plan), reserved_native_steps=reserved),
        )
        raw, run = run_cold_torque_trial(player, plan)
        nn_calls += run["neural_calls"]
        native_steps += run["native_steps"]
        path = a.output / (name + "-" + arm + ".npz")
        np.savez_compressed(path, **raw)
        pins[str(path)] = digest(path)
        future = dict(
            raw,
            controls=run["controls"],
            runtime_failure=run["runtime_failure"],
            body_failed=run["body_failed"],
        )
        metrics = summarize_motor_future(
            future, ball_address=player.layout.bq, root_address=player.layout.rq
        )
        result = dict(
            arm=arm,
            plan=asdict(plan),
            run=run,
            metrics=metrics,
            artifact=dict(path=path.name, hash=pins[str(path)]),
        )
        write_once(a.output / (name + "-" + arm + ".json"), result)
        return raw, result

    for name, xy in cases:
        player = new_player(xy)
        body_hash = hash_json(player.layout.profile())
        parent_hash = cftrack_state_hash(player.policy)
        plan = CFtrackTorqueIntentPlan(46, 24, (0.0,) * 29, body_hash, parent_hash)
        parent_raw, parent = execute(name, "parent", player, plan)
        parents[name] = parent_raw
        first = parent["metrics"]["first_foot_control"]
        support = candidate = model = failure = None
        if (
            complete_torque_trial(parent["run"])
            and parent["metrics"]["private_safe"]
            and parent["metrics"]["post60_observed"]
            and type(first) is int
            and 25 <= first <= 99
        ):
            player = new_player(xy)
            fractions = [0.0] * 29
            fractions[7] = 0.125
            support_plan = CFtrackTorqueIntentPlan(
                first - 25, 24, tuple(fractions), body_hash, cftrack_state_hash(player.policy)
            )
            _, support = execute(name, "support", player, support_plan)
            try:
                if not complete_torque_trial(support["run"]):
                    raise ValueError("complete current support and actual feedback required")
                player = new_player(xy)
                proposal, model = propose_torque_response(
                    parent["metrics"],
                    support["metrics"],
                    body_hash=body_hash,
                    parent_policy_hash=cftrack_state_hash(player.policy),
                )
                write_once(
                    a.output / (name + "-model-before.json"),
                    dict(model=model, proposal=asdict(proposal), hardware_authorized=False),
                )
                _, candidate = execute(name, "candidate", player, proposal)
            except ValueError as error:
                failure = dict(type=type(error).__name__, message=str(error))
        else:
            failure = dict(
                type="UnsafeOrCensoredParent", message="no safe complete current anchor; abstain"
            )
        m = candidate["metrics"] if candidate is not None else None
        accepted = bool(
            m
            and complete_torque_trial(candidate["run"])
            and m["private_safe"]
            and m["post60_observed"]
            and m["forward_60_m"] >= 1.2
            and m["lateral_forward_ratio"] <= 0.2
        )
        row = dict(
            case=name,
            ball_xy=xy,
            parent=parent,
            support=support,
            candidate=candidate,
            model=model,
            proposal_failure=failure,
            planning_margin_accepted=accepted,
            official_h300_exam=False,
        )
        records.append(row)
        write_once(a.output / (name + ".json"), row)
    prefixes = []
    for name, _ in cases:
        original = next(r for r in bank["records"] if r["case"] == name and r["arm"] == "parent")
        path = a.bank / original["artifact"]["path"]
        if path.parent != a.bank or path.is_symlink() or pins.get(str(path)) != digest(path):
            raise ValueError("bound parent evidence required after all proposals only")
        raw = parents[name]
        with np.load(path, allow_pickle=False) as saved:
            prefixes.append(
                len(raw["joint_targets"]) == 160
                and all(
                    np.array_equal(raw[k], saved[k][: len(raw[k])])
                    for k in (
                        "qpos",
                        "qvel",
                        "torque",
                        "ball_forces",
                        "joint_targets",
                        "nn_decoded_targets",
                    )
                )
            )
    unchanged = (
        cftrack_state_hash(template.policy) == template_hash and template.world.native_steps == 0
    )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_TORQUE_RESPONSE_DEVELOPMENT_RECORDED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            records=records,
            accepted_planning_margin_cases=[
                r["case"] for r in records if r["planning_margin_accepted"]
            ],
            parent_prefixes_exact=all(prefixes),
            cold_template_unchanged=unchanged,
            new_neural_calls=nn_calls,
            new_native_steps=native_steps,
            reserved_native_steps=reserved,
            neural_optimizer_updates=0,
            independent_candidate_readback_pending=True,
            full_h300_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
