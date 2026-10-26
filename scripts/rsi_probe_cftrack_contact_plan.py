"""Finite OWN parent/contact-plan preview; no actual policy replacement."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--future-qualification", type=Path, required=True)
    p.add_argument("--future-hash", required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_cftrack_native_context import build_cold_cftrack
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.future_qualification / "complete.json"
    if a.output.exists() or complete.is_symlink() or digest(complete) != a.future_hash:
        raise ValueError("unused output and exact closed own-future qualification required")
    qualified = json.loads(complete.read_text())
    if qualified["status"] != "CFTRACK_OWN_NN_NATIVE_FUTURE_EXACT":
        raise ValueError("qualified current-body NN/native interface required")
    pins = dict(qualified["input_file_hashes"])
    pins[str(complete)] = a.future_hash
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("source-qualified external policy required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one pinned native body required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            actual_native_steps=0,
            private_native_step_ceiling=3200,
            future_nn_ceiling=320,
            constructor_warmups=0,
            neural_optimizer_updates=0,
            reference_authority="absolute_FK_ankle_pitch_task_pose_not_joint_target",
            reference_window_samples=49,
            reference_half_window_controls=24,
            additional_sampled_reference_speed_limit_m_s=3.0,
            additional_sampled_reference_angular_speed_limit_rad_s=8.0,
            reference_torso_workspace_limit_m=1.25,
            old_joint_reference_certificates_not_inherited=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_contact_generator import generate_cftrack_contact_plan
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.rsi.shared_cftrack_forecast import SharedCFtrackForecaster

    player, body_hash, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    forecaster = SharedCFtrackForecaster(
        player,
        expected_policy_type=type(player.policy),
        expected_model_hash=body_hash,
        native_step_budget=3200,
    )
    before = cftrack_state_hash(player.policy)
    plan, generation, parent = generate_cftrack_contact_plan(player, forecaster)

    def save(name, value):
        path = a.output / (name + ".npz")
        np.savez_compressed(path, **{k: v for k, v in value.items() if isinstance(v, np.ndarray)})
        pins[str(path)] = digest(path)
        return {k: v for k, v in value.items() if not isinstance(v, np.ndarray)}

    parent_meta = save("parent-future", parent)
    # Load recorded source arrays ONLY now for comparison, never for control.
    original = a.future_qualification / "owned-future.npz"
    if original.is_symlink() or pins.get(str(original)) != digest(original):
        raise ValueError("bound original own prediction required for provenance comparison")
    with np.load(original, allow_pickle=False) as raw:
        if any(
            not np.array_equal(parent[k], raw[k])
            for k in ("qpos", "qvel", "torque", "ball_forces", "joint_targets")
        ):
            raise ValueError("original own parent changed before contact conditioning")
    candidate_meta = None
    if plan is not None:
        candidate_meta = save(
            "conditioned-future", forecaster.forecast(controls=160, contact_plan=plan)
        )
    if cftrack_state_hash(player.policy) != before or player.world.native_steps != 0:
        raise ValueError("private proposal modified actual donor")
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_CONTACT_PRIVATE_PREVIEW_RECORDED"
            if plan is not None
            else "CFTRACK_CONTACT_PLAN_ABSTAINED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            generation=generation,
            plan=asdict(plan) if plan is not None else None,
            parent=parent_meta,
            candidate=candidate_meta,
            future_counters=forecaster.counters(),
            actual_native_steps=0,
            constructor_warmups=0,
            neural_optimizer_updates=0,
            new_reference_actual_execution_not_qualified=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
