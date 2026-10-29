"""Source-bound SIM-only one-factor velocity/codec first-touch ablation."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import clean, source_pins


def arms():
    """Fixed, predeclared values; never select an arm using outcome."""
    return (
        ("strict-1.0", False, 1.0),
        ("experimental-1.0", True, 1.0),
        ("experimental-1.5", True, 1.5),
    )


def exact_arrays(left, right):
    return set(left) == set(right) and all(
        np.array_equal(left[k], right[k], equal_nan=True) for k in left
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "approach-config", "protocol", "output", "core-source"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "source-hash",
        "declaration-hash",
        "approach-config-hash",
        "protocol-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--shard", type=int, choices=range(4), required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    inputs = (
        (a.source / "complete.json", a.source_hash),
        (a.source / "declaration.json", a.declaration_hash),
        (a.protocol, a.protocol_hash),
    )
    if any(path.is_symlink() or digest(path) != expected for path, expected in inputs):
        raise ValueError("immutable source and protocol required")
    source = json.loads(inputs[0][0].read_text())
    declaration = json.loads(inputs[1][0].read_text())
    indices = list(range(a.shard, 40, 4))
    parents = {}
    for i in indices:
        rows = [r for r in source["rows"] if r["course_index"] == i and r["arm"] == "parent"]
        if len(rows) != 1:
            raise ValueError("one source parent per assigned course required")
        path = a.source / f"course-{i:02d}-parent.npz"
        if path.is_symlink() or digest(path) != rows[0]["raw_hash"]:
            raise ValueError("exact parent native trajectory required")
        parents[i] = (rows[0], path)
    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence

    fast_assets, pins = admit_heading_evidence(a.approach_config, a.approach_config_hash)
    pins.update({str(path): expected for path, expected in inputs})
    pins.update({str(path): row["raw_hash"] for row, path in parents.values()})
    source_assets = declaration["input_file_hashes"]

    def asset(suffix):
        found = [(Path(path), h) for path, h in source_assets.items() if path.endswith(suffix)]
        if len(found) != 1 or found[0][0].is_symlink() or digest(found[0][0]) != found[0][1]:
            raise ValueError("unique source-bound physical asset required: " + suffix)
        pins[str(found[0][0])] = found[0][1]
        return found[0]

    scene, _ = asset("/cached-standard/declared-scene.xml")
    velocity, velocity_hash = asset("/velocity/v0/exported/policy.onnx")
    config, config_hash = asset("/velocity/v0/params/deploy.yaml")
    kick, _ = asset("/policy/robonaldo/FreeKick.py")
    for suffix in (
        "/FSM/FSMState.py",
        "/common/ctrlcomp.py",
        "/common/utils.py",
        "/common/path_config.py",
        "/policy/robonaldo/config/freekick.yaml",
        "/policy/robonaldo/model/freekick_motion.npz",
        "/policy/robonaldo/model/policy-obs-aic.onnx",
    ):
        asset(suffix)
    source_pins(pins)
    core = a.core_source.resolve()
    if not (core / "rosclaw" / "__init__.py").is_file():
        raise ValueError("explicit frozen ROSClaw core source required")
    pins.update({str(path): digest(path) for path in (core / "rosclaw").rglob("*.py")})
    check_pins(pins)
    import mujoco
    import rosclaw
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode
    from rosclaw_soccer.sim.contracts import hash_json

    if Path(rosclaw.__file__).resolve() != core / "rosclaw" / "__init__.py":
        raise ValueError("imported ROSClaw core differs from pinned source")

    model = mujoco.MjModel.from_xml_string(shared_native_scene(scene.read_text(), prefixes=("a/",)))
    types = _load_robonaldo(kick.parents[2])
    assets = dict(
        velocity_model=velocity,
        velocity_config=config,
        velocity_model_hash=velocity_hash,
        velocity_config_hash=config_hash,
        **fast_assets,
    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_hash=a.source_hash,
            protocol_hash=a.protocol_hash,
            shard=a.shard,
            indices=indices,
            arms=arms(),
            input_file_hashes=pins,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    rows = []
    try:
        for i in indices:
            parent, parent_path = parents[i]
            expected_parameters = dict(
                speed_mps=1.0, entry_gap_m=0.9, reference_frame=None, root_lateral_offset_m=0.0
            )
            if parent["result"]["parameters"] != expected_parameters:
                raise ValueError("original parent parameter mismatch")
            for name, fast, speed in arms():
                params = dict(expected_parameters, speed_mps=speed)
                local_assets = dict(assets, experimental_fast=fast)
                original = None
                for repeat in range(2):
                    result, arrays = evaluate_episode(
                        model=model,
                        assets=local_assets,
                        policy_types=types,
                        course=parent["result"]["course"],
                        parameters=params,
                        hard_limits=np.asarray(G1_HARD_TORQUE_LIMITS),
                    )
                    label = relabel_arrays(
                        arrays,
                        controls=result["controls"],
                        ball_address=result["ball_qpos_address"],
                        root_address=result["robot_qpos_address"],
                    )
                    if hash_json(label) != hash_json(result["outcome"]):
                        raise ValueError("independent physical outcome mismatch")
                    if name == "strict-1.0":
                        with np.load(parent_path, allow_pickle=False) as saved:
                            if not exact_arrays(arrays, saved):
                                raise ValueError("strict parent native replay mismatch")
                    if original is not None and not exact_arrays(arrays, original):
                        raise ValueError("fresh native replay mismatch")
                    if original is None:
                        original = {k: v.copy() for k, v in arrays.items()}
                    path = a.output / f"course-{i:02d}-{name}-{repeat}.npz"
                    np.savez_compressed(path, **arrays)
                    rows.append(
                        dict(
                            course=i,
                            arm=name,
                            repeat=repeat,
                            result=result,
                            artifact_hash=digest(path),
                            artifact=path.name,
                        )
                    )
        clean(a.expected_head)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="SPEED_ISOLATION_DEVELOPMENT_SIM_ONLY",
                source_head=a.expected_head,
                protocol_hash=a.protocol_hash,
                shard=a.shard,
                rows=rows,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error_type=type(error).__name__, error=str(error), completed_rows=rows),
        )
        raise


if __name__ == "__main__":
    main()
