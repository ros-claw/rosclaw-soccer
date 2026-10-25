"""Execute known rescued bodies with the complement enabled; retain exact old skills."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference-audit",
        "reference-root",
        "backbone-model",
        "plastic-model",
        "qualification",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "reference-hash",
        "backbone-hash",
        "plastic-hash",
        "qualification-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if (
        a.output.exists()
        or digest(a.reference_audit) != a.reference_hash
        or digest(a.qualification) != a.qualification_hash
    ):
        raise ValueError("unused output and exact independently verified sources required")
    reference = json.loads(a.reference_audit.read_text())
    qualification = json.loads(a.qualification.read_text())
    if (
        not reference["qualified_on_expanded_public"]
        or reference.get("perturbed_public_validation", False)
        or qualification["status"] != "G1_FEEDBACK_COMPLEMENT_REFERENCE_NN_FUTURES_EXACT"
        or qualification["reference_conditioning"] != "feedback_complement_nonzero"
    ):
        raise ValueError(
            "old expanded teacher qualification and exact new nonzero reference required"
        )
    pins = dict(reference["input_file_hashes"])
    for key, value in qualification["input_file_hashes"].items():
        if key in pins and pins[key] != value:
            raise ValueError("conflicting physical/model provenance")
        pins[key] = value
    pins[str(a.reference_audit)], pins[str(a.qualification)] = (
        a.reference_hash,
        a.qualification_hash,
    )
    for path, expected in ((a.backbone_model, a.backbone_hash), (a.plastic_model, a.plastic_hash)):
        if pins.get(str(path)) != expected or digest(path) != expected:
            raise ValueError("exact qualified backbone and complement model required")
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique qualified source required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.synergy_forecast_kick_player import SynergyForecastKickPlayer
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode

    def model_basis(path):
        report = json.loads(path.read_text())
        artifact = path.parent / report["artifact"]["path"]
        if (
            pins.get(str(artifact)) != report["artifact"]["hash"]
            or digest(artifact) != report["artifact"]["hash"]
        ):
            raise ValueError("bound exact model artifact required")
        with np.load(artifact, allow_pickle=False) as raw:
            return raw["basis_hg"].copy()

    old_basis, new_basis = model_basis(a.backbone_model), model_basis(a.plastic_model)
    configs = [
        Path(k)
        for k in pins
        if k.endswith(".json")
        and Path(k).stat().st_size < 2048
        and json.loads(Path(k).read_text()).get("schema")
        == "soccer.native_single_finisher_experimental_approach.v1"
    ]
    if len(configs) != 1:
        raise ValueError("one qualified experimental approach codec required")
    fast, _ = admit_heading_evidence(configs[0], pins[str(configs[0])])
    vm, vc = one("/velocity/v0/exported/policy.onnx"), one("/velocity/v0/params/deploy.yaml")
    assets = dict(
        velocity_model=vm,
        velocity_config=vc,
        velocity_model_hash=pins[str(vm)],
        velocity_config_hash=pins[str(vc)],
        **fast,
    )
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    types = _load_robonaldo(one("/policy/robonaldo/FreeKick.py").parents[2])
    instances = []

    class Controlled(SynergyForecastKickPlayer):
        basis_hg, plastic_basis_hg = old_basis, new_basis
        forecast_controls = 128

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    cp = a.reference_root / "complete.json"
    if pins.get(str(cp)) != digest(cp):
        raise ValueError("exact expanded physical reference required")
    producer = json.loads(cp.read_text())
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            courses=[0, 30],
            actual_world_ceiling=2,
            actual_native_step_ceiling=7200,
            maximum_requested_forecast_native_steps=94720,
            plastic_enabled_but_must_not_replace_passing_old_choice=True,
            full_actual_trajectories_must_equal_previous_champion=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    results, total_native, requested, integrated = [], 0, 0, 0
    for i in (0, 30):
        ref = producer["results"][i]["arms"]["C"]
        rp = a.reference_root / ref["record"]
        if pins.get(str(rp)) != ref["hash"] or digest(rp) != ref["hash"]:
            raise ValueError("bound exact old successful control record required")
        old = json.loads(rp.read_text())
        instances.clear()
        row, arrays = evaluate_episode(
            model=model,
            assets=dict(assets, experimental_fast=old["experimental_fast_approach"]),
            policy_types=types,
            course=tuple(old["course"]),
            parameters=old["parameters"],
            hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
            player_type=Controlled,
        )
        if len(instances) != 1:
            raise ValueError("one actual measured private teacher required")
        ctrl = instances[0]
        if (
            ctrl.events[0]["selected_basis"] != "backbone"
            or ctrl.events[0]["plastic_generations"]
            or len(ctrl.candidate_predictions) != 37
        ):
            raise ValueError("passing old skill was not retained before plastic search")
        with np.load(rp.parent / old["artifact"]["path"], allow_pickle=False) as raw:
            if set(raw.files) != set(arrays) or any(
                not np.array_equal(raw[k], v) for k, v in arrays.items()
            ):
                raise ValueError("enabled complement altered old complete actual physical skill")
        if not row["outcome"]["task_pass"] or not row["healthy_full_horizon"]:
            raise ValueError("old actual rescued skill lost")
        total_native += row["native_steps"]
        requested += ctrl.forecaster.attempted_steps
        integrated += ctrl.forecaster.integrated_steps
        artifact = a.output / f"actual-{i:02d}.npz"
        np.savez_compressed(artifact, **arrays)
        results.append(
            dict(
                course_index=i,
                outcome=row["outcome"],
                whole_body_healthy=True,
                old_whole_actual_trajectory_exact=True,
                plastic_search_calls=0,
                counters=ctrl.forecaster.counters(),
                artifact=dict(path=artifact.name, hash=digest(artifact)),
            )
        )
    if total_native != 7200 or requested != 94720:
        raise ValueError("frozen two-body retention budget differs")
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="STABLE_PLASTIC_BACKBONE_RESCUES_FULL_ACTUAL_TRAJECTORIES_EXACT",
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=results,
            actual_worlds=2,
            actual_native_steps=total_native,
            forecast_requested_native_steps=requested,
            forecast_integrated_native_steps=integrated,
            complement_effectiveness_proven=False,
            neural_optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
