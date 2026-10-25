"""Qualify held-PD forecasts against a full original G1 trajectory, no teacher gain claim."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "reference-record", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or a.audit.is_symlink() or digest(a.audit) != a.audit_hash:
        raise ValueError("new output and exact closed cohort audit required")
    audit = json.loads(a.audit.read_text())
    if audit["status"] != "ALL_THREE_CAUSAL_COHORTS_NATIVE_NEURAL_RNG_AND_ROLLBACK_REPLAY_VERIFIED":
        raise ValueError("completed independent native cohort review required")
    pins = dict(audit["input_file_hashes"])
    pins[str(a.audit)] = a.audit_hash
    if a.reference_record.is_symlink() or pins.get(str(a.reference_record)) != digest(
        a.reference_record
    ):
        raise ValueError("exact previously verified original reference trajectory required")
    reference = json.loads(a.reference_record.read_text())
    if reference["arm"] != "onnx":
        raise ValueError("original ONNX controller reference required")
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique original physical asset required")
        return paths[0]

    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import (
        SharedTargetForecaster,
        compiled_model_hash,
    )
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
        raise ValueError("unique original qualified approach required")
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

    class Probe(SharedRoboNaldoPlayer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.forecaster = SharedTargetForecaster(
                self.world, expected_model_hash=model_hash, native_step_budget=3600
            )
            self.predictions = []
            instances.append(self)

        def propose(self):
            target, receipt = super().propose()
            self.predictions.append(self.forecaster.forecast({self.prefix: target}))
            return target, receipt

        def commit(self):
            super().commit()
            predicted = self.predictions[-1]
            if any(
                not np.array_equal(predicted[k][-1], actual)
                for k, actual in (
                    ("qpos", self.world.data.qpos),
                    ("qvel", self.world.data.qvel),
                    ("torque", self.world.data.ctrl),
                )
            ):
                raise ValueError("held-PD forecast did not reproduce actual next native boundary")

    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            model_hash=model_hash,
            original_course=reference["course"],
            original_parameters=reference["parameters"],
            actual_worlds=1,
            actual_native_step_ceiling=3600,
            forecast_native_step_ceiling=3600,
            no_target_changes=True,
            every_forecast_must_match_all_actual_native_arrays=True,
            optimizer_updates=0,
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
        raise ValueError("actual measured neural entry required for forecast qualification")
    probe = instances[0]
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        predicted = np.concatenate([x[key] for x in probe.predictions])
        if not np.array_equal(predicted, arrays[key][row["entry_control"] * 10 :]):
            raise ValueError("forecast failed exact every-substep physical array replay: " + key)
    with np.load(a.reference_record.parent / reference["artifact"]["path"]) as old:
        if set(old.files) != set(arrays) or any(
            not np.array_equal(old[k], v) for k, v in arrays.items()
        ):
            raise ValueError("forecast observation perturbed the actual original rollout")
    artifact = a.output / "actual.npz"
    np.savez_compressed(artifact, **arrays)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="G1_SHARED_PD_FORECAST_ALL_NEURAL_SUBSTEPS_EXACT_AND_ORIGINAL_ROLLOUT_UNCHANGED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            actual_worlds=1,
            actual_native_steps=row["native_steps"],
            forecasts=probe.forecaster.counters(),
            actual_record=row,
            artifact=dict(path=artifact.name, hash=digest(artifact)),
            teacher_effectiveness_proven=False,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
