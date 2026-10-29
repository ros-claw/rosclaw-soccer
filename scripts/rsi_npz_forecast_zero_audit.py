"""Reproduce a source-bound NPZmotion baseline in an isolated private SIM forecast."""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import clean, source_pins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--baseline-hash", required=True)
    p.add_argument("--external", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.baseline.is_symlink() or digest(a.baseline) != a.baseline_hash:
        raise ValueError("unused output and exact baseline receipt required")
    baseline = json.loads(a.baseline.read_text())
    external = a.external.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(external), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(external), "status", "--porcelain"])
    ground = bool(baseline.get("ground_reference_ball"))
    if (
        dirty
        or head != baseline["external_head"]
        or baseline["reference_source"] != "NPZmotion"
        or baseline["reference_all_initial_velocities"] is not True
        or ground != bool(baseline.get("zero_reference_ball_spin"))
        or baseline["runtime_failure"] is not None
        or baseline["controls"] < 300
        or baseline["promotion_authorized"] is not False
    ):
        raise ValueError("exact complete unpromoted original or ground/nospin baseline required")
    pins = dict(baseline["input_file_hashes"])
    pins[str(a.baseline.resolve())] = a.baseline_hash
    artifact = a.baseline.parent / baseline["artifact"]["path"]
    pins[str(artifact.resolve())] = baseline["artifact"]["hash"]
    reference = Path(baseline["motion_source"])
    scene = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scene) != 1 or str(reference) not in pins:
        raise ValueError("unique pinned standard scene and reference required")
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            baseline_hash=a.baseline_hash,
            expected_head=a.expected_head,
            external_head=head,
            controls=300,
            native_step_budget=3000,
            ground_zero_spin=ground,
            input_file_hashes=pins,
            hardware_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.shared_cftrack_forecast import SharedCFtrackForecaster
    from scripts.rsi_cftrack_native_context import build_reference_cftrack

    failure = None
    result = None
    comparisons = {}
    try:
        player, model_hash = build_reference_cftrack(
            scene[0].read_text(), external, reference, ground_zero_spin=ground
        )
        forecast = SharedCFtrackForecaster(
            player,
            expected_policy_type=type(player.policy),
            expected_model_hash=model_hash,
            native_step_budget=3000,
            reference_source="NPZmotion",
        )
        result = forecast.forecast(controls=300, capture_inference=True)
        np.savez_compressed(
            a.output / "private-forecast.npz",
            **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
        )
        with np.load(artifact, allow_pickle=False) as original:
            for key in (
                "qpos",
                "qvel",
                "torque",
                "ball_forces",
                "joint_targets",
                "nn_decoded_targets",
                "neural_obs",
                "neural_actions",
            ):
                expected = original[key][
                    : 3000 if key in ("qpos", "qvel", "torque", "ball_forces") else 300
                ]
                observed = result[key]
                comparisons[key] = dict(
                    shape_match=observed.shape == expected.shape,
                    exact=bool(np.array_equal(observed, expected)),
                    max_abs=float(np.max(np.abs(observed - expected)))
                    if observed.shape == expected.shape and observed.size
                    else None,
                )
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    check_pins(pins)
    clean(a.expected_head)
    passed = bool(
        result is not None
        and result["controls"] == 300
        and not result["runtime_failure"]
        and len(comparisons) == 8
        and all(x["exact"] for x in comparisons.values())
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="EXACT_ZERO_BASELINE" if passed else "ZERO_BASELINE_NOT_REPRODUCED",
            source_head=a.expected_head,
            baseline_hash=a.baseline_hash,
            ground_zero_spin=ground,
            runtime_failure=failure
            if failure
            else result["runtime_failure"]
            if result is not None
            else None,
            controls=result["controls"] if result is not None else 0,
            comparisons=comparisons,
            hardware_authorized=False,
            promotion_authorized=False,
            artifact_hash=digest(a.output / "private-forecast.npz") if result is not None else None,
        ),
    )


if __name__ == "__main__":
    main()
