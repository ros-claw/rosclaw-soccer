"""Qualify copied-policy 64-control predictions on actual G1; no task gain claim."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("effect-review", "reference-record", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--effect-review-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--pre-proposal-96", action="store_true")
    p.add_argument("--pre-proposal-128", action="store_true")
    p.add_argument("--synergy-model", type=Path)
    p.add_argument("--synergy-model-hash")
    a = p.parse_args()
    if a.pre_proposal_96 and a.pre_proposal_128:
        raise ValueError("one explicit forecast qualification protocol required")
    pre_proposal = a.pre_proposal_96 or a.pre_proposal_128
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or digest(a.effect_review) != a.effect_review_hash:
        raise ValueError("unused output and exact closed local-pilot effect readback required")
    review = json.loads(a.effect_review.read_text())
    if review["status"] != "TASK_SPACE_NATIVE_EFFECT_READBACK_COMPLETE":
        raise ValueError("closed physical effect analysis required")
    pins = dict(review["input_file_hashes"])
    pins[str(a.effect_review)] = a.effect_review_hash
    synergy_model = None
    if a.synergy_model is not None:
        if not pre_proposal or digest(a.synergy_model) != a.synergy_model_hash:
            raise ValueError("exact synergy model and explicit pre-proposal protocol required")
        synergy_model = json.loads(a.synergy_model.read_text())
        if synergy_model["status"] != "PUBLIC_EXECUTED_MOTION_SYNERGIES_FITTED":
            raise ValueError("executed public motion subspace required")
        pins.update(synergy_model["input_file_hashes"])
        pins[str(a.synergy_model)] = a.synergy_model_hash
    if pins.get(str(a.reference_record)) != digest(a.reference_record):
        raise ValueError("explicit pinned original trajectory required")
    reference = json.loads(a.reference_record.read_text())
    if reference["arm"] != "onnx":
        raise ValueError("original unmodified ONNX controller required")
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique qualified research asset required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence
    from rosclaw_soccer.rsi.motion_synergy_reference import (
        MotionSynergyPlan,
        condition_synergy_reference,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_nn_forecast import SharedNNForecaster
    from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode

    cfgs = [
        Path(k)
        for k in pins
        if k.endswith(".json")
        and Path(k).stat().st_size < 2048
        and json.loads(Path(k).read_text()).get("schema")
        == "soccer.native_single_finisher_experimental_approach.v1"
    ]
    if len(cfgs) != 1:
        raise ValueError("unique qualified experimental approach required")
    fast, _ = admit_heading_evidence(cfgs[0], pins[str(cfgs[0])])
    vm, vc = one("/velocity/v0/exported/policy.onnx"), one("/velocity/v0/params/deploy.yaml")
    assets = dict(
        velocity_model=vm,
        velocity_config=vc,
        velocity_model_hash=pins[str(vm)],
        velocity_config_hash=pins[str(vc)],
        **fast,
    )
    assets["experimental_fast"] = reference["experimental_fast_approach"]
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    model_hash = compiled_model_hash(model)
    types = _load_robonaldo(one("/policy/robonaldo/FreeKick.py").parents[2])
    instances = []
    frames = (80, 100, 150, 220)
    horizon = 128 if a.pre_proposal_128 else 96 if a.pre_proposal_96 else 64
    forecast_budget = len(frames) * horizon * 10
    basis_isaac = None
    if synergy_model is not None:
        with np.load(
            a.synergy_model.parent / synergy_model["artifact"]["path"], allow_pickle=False
        ) as raw:
            basis_hg = raw["basis_hg"]
        if basis_hg.shape != (6, 29):
            raise ValueError("rank6 motion subspace required")
        basis_isaac = np.zeros_like(basis_hg)
        basis_isaac[:, types[3]] = basis_hg

    class Probe(SharedRoboNaldoPlayer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.predictor = SharedNNForecaster(
                self,
                expected_policy_type=types[2],
                expected_model_hash=model_hash,
                native_step_budget=forecast_budget,
            )
            self.predictions = []
            instances.append(self)

        def install_measured_entry(self, samples, *, reference_frame):
            receipt = super().install_measured_entry(samples, reference_frame=reference_frame)
            if basis_isaac is not None:
                plan = MotionSynergyPlan(
                    reference_frame,
                    tuple(tuple(float(x) for x in r) for r in basis_isaac),
                    ((0.01, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                )
                condition_synergy_reference(self.policy, plan)
            return receipt

        def propose(self):
            if pre_proposal and self.world.control_steps in frames:
                self.predictions.append(
                    self.predictor.forecast(controls=horizon, before_proposal=True)
                )
            target, receipt = super().propose()
            if not pre_proposal and self.world.control_steps in frames:
                self.predictions.append(self.predictor.forecast(controls=horizon))
            return target, receipt

    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            model_hash=model_hash,
            probes=frames,
            controls_per_probe=horizon,
            before_proposal=pre_proposal,
            reference_conditioning="synergy_nonzero"
            if synergy_model is not None
            else "OFF_original_arrays",
            actual_worlds=1,
            actual_native_step_ceiling=3600,
            forecast_native_step_ceiling=forecast_budget,
            every_future_target_and_native_array_must_match=True,
            no_changed_targets=synergy_model is None,
            copied_history_no_enter_or_reset=True,
            optimizer_updates=0,
            teacher_effectiveness_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    row, arrays = evaluate_episode(
        model=model,
        assets=assets,
        policy_types=types,
        course=tuple(reference["course"]),
        parameters=reference["parameters"],
        hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
        player_type=Probe,
    )
    if len(instances) != 1 or row["entry_control"] is None:
        raise ValueError("actual admitted neural skill required")
    probe = instances[0]
    if tuple(r["donor_control_frame"] for r in probe.predictions) != frames:
        raise ValueError("all four preregistered NN forecast boundaries required")
    for predicted in probe.predictions:
        start = predicted["donor_control_frame"]
        if predicted["controls"] != horizon or predicted["body_failed"]:
            raise ValueError("original healthy full forecast future not reproduced")
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            if not np.array_equal(predicted[key], arrays[key][start * 10 : (start + horizon) * 10]):
                raise ValueError("closed-loop NN forecast differs from actual future: " + key)
        if not np.array_equal(
            predicted["joint_targets"], arrays["joint_targets"][start : start + horizon]
        ):
            raise ValueError("future NN calls do not reproduce actual future PD targets")
        artifact = a.output / f"forecast-{start:03d}.npz"
        np.savez_compressed(
            artifact,
            **{
                k: predicted[k]
                for k in ("qpos", "qvel", "torque", "ball_forces", "joint_targets", "policy_ticks")
            },
        )
        pins[str(artifact)] = digest(artifact)
    with np.load(a.reference_record.parent / reference["artifact"]["path"]) as old:
        same = set(old.files) == set(arrays) and all(
            np.array_equal(old[k], v) for k, v in arrays.items()
        )
        if synergy_model is None and not same:
            raise ValueError("owned forecasts changed original actual whole trajectory")
        if synergy_model is not None and same:
            raise ValueError(
                "nonzero reference qualification did not change actual whole trajectory"
            )
    artifact = a.output / "actual.npz"
    np.savez_compressed(artifact, **arrays)
    pins[str(artifact)] = digest(artifact)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                f"G1_COPIED_NN_CLOSED_LOOP_{horizon}_CONTROL_FORECASTS_EXACT"
                if synergy_model is None
                else "G1_SYNERGY_REFERENCE_NN_FUTURES_EXACT"
            ),
            before_proposal=pre_proposal,
            reference_conditioning="synergy_nonzero"
            if synergy_model is not None
            else "OFF_original_arrays",
            source_head=a.expected_head,
            input_file_hashes=pins,
            actual_record=row,
            actual_worlds=1,
            actual_native_steps=row["native_steps"],
            forecasts=probe.predictor.counters(),
            all_future_nn_targets_and_native_arrays_exact=True,
            actual_whole_trajectory_unchanged=synergy_model is None,
            teacher_effectiveness_proven=False,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
