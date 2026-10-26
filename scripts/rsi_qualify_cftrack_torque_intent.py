"""New SIM torque protocol: zero exactness plus one nonzero native-feedback probe."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path


class CaptureSession:
    def __init__(self, original):
        self.original = original
        self.observations = []
        self.actions = []

    def run(self, outputs, inputs):
        if len(self.observations) >= 160 or inputs["obs"].shape != (1, 1244):
            raise ValueError("bounded original NN1244 required")
        result = self.original.run(outputs, inputs)
        self.observations.append(inputs["obs"].copy())
        self.actions.append(result[0].copy())
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or a.audit.is_symlink() or digest(a.audit) != a.audit_hash:
        raise ValueError("unused qualification and bound completed PD intent audit required")
    audit = json.loads(a.audit.read_text())
    bank_path = a.bank / "complete.json"
    if (
        audit["status"] != "CFTRACK_VERIFIED_BANK_PD_INTENT_AUDITED"
        or bank_path.is_symlink()
        or digest(bank_path) != audit["source_bank_hash"]
    ):
        raise ValueError("exact independently replayed development bank required")
    bank = json.loads(bank_path.read_text())
    pins = dict(audit["input_file_hashes"])
    pins[str(a.audit)] = a.audit_hash
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("qualified external policy required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(s) for s in pins if s.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique qualified body required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.cftrack.torque_intent_qualification.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            arms=["zero", "hg8-positive-small"],
            controls_per_arm=160,
            maximum_native_steps=3200,
            maximum_neural_calls=320,
            nonzero_fraction_effort=0.01,
            nonzero_hg_index=8,
            start_control=46,
            peak_control=24,
            torque_added_after_parent_pd_saturation=True,
            measured_stop_outward_residual_suppressed=True,
            original_nn_and_pd_setpoints_unchanged=True,
            not_algebraically_equivalent_to_position_residual=True,
            unchanged_native_effort_fraction=0.85,
            no_amplitude_grid=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
    from rosclaw_soccer.rsi.cftrack_motor_authority import summarize_motor_future
    from rosclaw_soccer.rsi.cftrack_torque_intent import (
        CFtrackTorqueIntentPlan,
        CFtrackTorqueIntentPolicy,
    )
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.rsi.pd_intent_audit import audit_pd_intent
    from rosclaw_soccer.rsi.shared_native_torque_layer import SharedNativeTorqueLayer
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_cftrack_native_context import build_cold_cftrack

    template, _, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    before = cftrack_state_hash(template.policy)
    records = []
    raws = []
    for arm in ("zero", "hg8-positive-small"):
        player = cold_cftrack_trial(template, ball_xy=(1.0, 0.0))
        condition_direct_kick_approach(player.policy)
        w, lo = player.world, player.layout
        fractions = [0.0] * 29
        if arm != "zero":
            fractions[8] = 0.01
        plan = CFtrackTorqueIntentPlan(
            46, 24, tuple(fractions), hash_json(lo.profile()), cftrack_state_hash(player.policy)
        )
        policy = CFtrackTorqueIntentPolicy(plan, w.parameters[player.prefix].effort)
        layer = SharedNativeTorqueLayer(
            w,
            player.prefix,
            policy,
            target_y_m=0.0,
            target_z_m=0.11,
            horizon_controls=160,
            expected_parent_policy_hash=plan.parent_policy_hash,
        )
        original = player.policy.ort_session
        capture = CaptureSession(original)
        obs, actions = capture.observations, capture.actions
        player.policy.ort_session = capture
        rows, targets, decoded = [], [], []
        failure = None
        initial_qpos, initial_qvel = w.data.qpos.copy(), w.data.qvel.copy()
        try:
            for _ in range(160):
                target, measured = player.propose()
                row = w.advance({player.prefix: target}, torque_layers={player.prefix: layer})
                rows.append(row)
                targets.append(target)
                decoded.append(measured["nn_decoded_target"])
                player.commit()
                if w.failed:
                    break
        except Exception as error:
            failure = dict(type=type(error).__name__, message=str(error))
        finally:
            player.policy.ort_session = original
        raw = {
            key: np.concatenate([r[key] for r in rows]) if rows else np.empty((0, width))
            for key, width in [
                ("qpos", w.model.nq),
                ("qvel", w.model.nv),
                ("torque", w.model.nu),
                ("parent_torque", w.model.nu),
            ]
        }
        raw.update(
            ball_forces=np.concatenate([r["ball_forces"] for r in rows])
            if rows
            else np.empty((0, 1, 3)),
            joint_targets=np.array(targets).reshape(-1, 29),
            nn_decoded_targets=np.array(decoded).reshape(-1, 29),
            nn_observations=np.array(obs).reshape(-1, 1244),
            nn_actions=np.array(actions).reshape(-1, 29),
            initial_qpos=initial_qpos,
            initial_qvel=initial_qvel,
        )
        path = a.output / (arm + ".npz")
        np.savez_compressed(path, **raw)
        pins[str(path)] = digest(path)
        metrics = summarize_motor_future(
            dict(raw, controls=len(rows), runtime_failure=failure, body_failed=w.failed),
            ball_address=lo.bq,
            root_address=lo.rq,
        )
        numeric = (
            audit_pd_intent(
                decoded=raw["nn_decoded_targets"],
                targets=raw["joint_targets"],
                positions=raw["qpos"][:, lo.qi],
                torques=raw["torque"][:, lo.ai],
                limits=w.model.jnt_range[lo.joints].copy(),
                effort=w.parameters[player.prefix].effort,
            )
            if rows
            else None
        )
        row = dict(
            arm=arm,
            plan=asdict(plan),
            controls=len(rows),
            native_steps=w.native_steps,
            neural_calls=len(obs),
            runtime_failure=failure,
            metrics=metrics,
            numeric=numeric,
            layer_commands=layer.commands,
            layer_feedback=layer.applied,
            saturation_steps=policy.saturation_steps,
            measured_stop_guard_steps=policy.stop_guard_steps,
            actual_changed_torque_entries=int((raw["torque"] != raw["parent_torque"]).sum()),
            artifact=dict(path=path.name, hash=pins[str(path)]),
        )
        write_once(a.output / (arm + ".json"), row)
        records.append(row)
        raws.append(raw)
    # Provenance comparison only AFTER new controller executions, never playback.
    source = next(r for r in bank["records"] if r["case"] == "nominal" and r["arm"] == "parent")
    path = a.bank / source["artifact"]["path"]
    if path.parent != a.bank or path.is_symlink() or pins.get(str(path)) != digest(path):
        raise ValueError("exact source parent trajectory required")
    with np.load(path, allow_pickle=False) as saved:
        zero_exact = all(
            np.array_equal(raws[0][k], saved[k][: len(raws[0][k])])
            for k in (
                "qpos",
                "qvel",
                "torque",
                "ball_forces",
                "joint_targets",
                "nn_decoded_targets",
            )
        )
    template_unchanged = (
        cftrack_state_hash(template.policy) == before and template.world.native_steps == 0
    )
    qualified = (
        zero_exact
        and template_unchanged
        and records[0]["actual_changed_torque_entries"] == 0
        and records[1]["actual_changed_torque_entries"] > 0
        and all(
            r["controls"] == 160
            and r["runtime_failure"] is None
            and r["layer_commands"] == r["layer_feedback"] == 1600
            and r["metrics"]["private_safe"]
            and not r["numeric"]["effort_cap_exceeded"]
            for r in records
        )
    )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_TORQUE_INTENT_ZERO_AND_FEEDBACK_QUALIFIED"
            if qualified
            else "CFTRACK_TORQUE_INTENT_NOT_QUALIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            records=records,
            zero_native_prefix_exact=zero_exact,
            cold_template_unchanged=template_unchanged,
            new_neural_calls=sum(r["neural_calls"] for r in records),
            new_native_steps=sum(r["native_steps"] for r in records),
            neural_optimizer_updates=0,
            independent_nonzero_readback_pending=True,
            independent_nonzero_native_replay_pending=True,
            qualification_not_skill_improvement=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
