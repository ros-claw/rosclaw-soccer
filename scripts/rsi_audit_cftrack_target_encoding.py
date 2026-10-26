"""Assess executed velocity targets in CFtrack coordinates; no action installation."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.source / "complete.json"
    if (
        a.output.exists()
        or complete.is_symlink()
        or a.review.is_symlink()
        or digest(complete) != a.source_hash
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("unused output and exact source and review required")
    source, review = json.loads(complete.read_text()), json.loads(a.review.read_text())
    if (
        source["status"] != "WARM_LEFT_SETTLING_MEASURED_NOT_KICK"
        or review["status"] != "WARM_LEFT_SETTLING_NN_AND_NATIVE_EXACT_NOT_KICK"
        or review["source_hash"] != a.source_hash
        or source["controls"] != 157
        or source["native_steps"] != 1570
    ):
        raise ValueError("exact independently replayed source required")
    trace = a.source / source["artifact"]["path"]
    if (
        trace.is_symlink()
        or trace.parent != a.source
        or digest(trace) != source["artifact"]["hash"]
    ):
        raise ValueError("exact confined executed-target artifact required")
    pins = dict(source["input_file_hashes"])
    pins.update(
        {
            str(complete): a.source_hash,
            str(a.review): a.review_hash,
            str(trace): source["artifact"]["hash"],
        }
    )
    source_pins(pins)
    check_pins(pins)
    configs = [Path(x) for x in pins if x.endswith("/policy/omnicontact/config/OmniContact.yaml")]
    if len(configs) != 1:
        raise ValueError("unique pinned CFtrack configuration required")
    import numpy as np
    import yaml

    from rosclaw_soccer.rsi.cftrack_action_history import describe_previous_target_encoding

    config = yaml.safe_load(configs[0].read_text())
    default = np.asarray(config["default_angles_lab"], np.float32)
    scale = np.asarray(config["action_scale_lab"], np.float32)
    mj2lab = np.asarray(config["mj2lab"], np.int32)
    lab2mj = np.asarray(config["lab2mj"], np.int32)
    with np.load(trace, allow_pickle=False) as f:
        targets = f["joint_targets"].copy()
    if targets.shape != (157, 29) or not np.isfinite(targets).all():
        raise ValueError("complete finite actually executed target history required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.source_hash,
            review_hash=a.review_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            target_frames=157,
            config_hash=pins[str(configs[0])],
            no_clipping=True,
            no_history_installed=True,
            no_physics_or_training=True,
            hardware_authorized=False,
        ),
    )
    rows = []
    for i, target in enumerate(targets):
        r = describe_previous_target_encoding(target, default, scale, mj2lab, lab2mj)
        rows.append(
            dict(
                control=i + 1,
                max_target_roundtrip_error_rad=r["maximum_target_roundtrip_error_rad"],
                max_implied_action_abs=r["maximum_absolute_implied_action"],
            )
        )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_TARGET_ENCODING_MEASURED_NOT_HANDOFF",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            review_hash=a.review_hash,
            input_file_hashes=pins,
            target_frames=len(rows),
            rows=rows,
            max_roundtrip_error_rad=max(r["max_target_roundtrip_error_rad"] for r in rows),
            max_implied_action_abs=max(r["max_implied_action_abs"] for r in rows),
            no_history_installed=True,
            no_physics_or_training=True,
            prior_policy_action_proven=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
