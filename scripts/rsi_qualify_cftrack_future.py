"""Single owned current-CFtrack NN future, exact native/actual/source-prefix check."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readback", type=Path, required=True)
    p.add_argument("--readback-hash", required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_cftrack_native_context import build_cold_cftrack
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or a.readback.is_symlink() or digest(a.readback) != a.readback_hash:
        raise ValueError("unused future qualification and committed actual NN readback required")
    audit = json.loads(a.readback.read_text())
    complete = a.qualification / "complete.json"
    if (
        audit["status"] != "CFTRACK_ACTUAL_NN1244_HISTORY_AND_TARGETS_VERIFIED"
        or complete.is_symlink()
        or digest(complete) != audit["qualification_hash"]
    ):
        raise ValueError("exact independent actual-input proof required")
    old = json.loads(complete.read_text())
    if old["reference_conditioning"] is None or old["runtime_failure"] is not None:
        raise ValueError("qualified original direct-approach reference baseline required")
    pins = dict(audit["input_file_hashes"])
    pins[str(a.readback)] = a.readback_hash
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("exact source-pinned external policy required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique source-qualified native body required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            actual_worlds=1,
            actual_native_step_ceiling=1600,
            private_future_native_step_ceiling=1600,
            total_nn_ceiling=320,
            constructor_warmups=0,
            neural_optimizer_updates=0,
            saved_source_future_not_used_for_control=True,
            parent_neural_policy_or_physics_not_modified=True,
            qualification_not_skill_improvement=True,
            hardware_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.rsi.shared_cftrack_forecast import SharedCFtrackForecaster
    from rosclaw_soccer.sim.contracts import hash_json

    player, body_hash, receipt = build_cold_cftrack(
        scenes[0].read_text(), external, direct_approach=True
    )
    if hash_json(receipt) != hash_json(old["reference_conditioning"]):
        raise ValueError("source-bound original reference generation differs")
    forecaster = SharedCFtrackForecaster(
        player,
        expected_policy_type=type(player.policy),
        expected_model_hash=body_hash,
        native_step_budget=1600,
    )
    before = cftrack_state_hash(player.policy)
    future = forecaster.forecast(controls=160)
    if cftrack_state_hash(player.policy) != before or player.world.control_steps != 0:
        raise ValueError("forecast modified the actual donor")
    path = a.output / "owned-future.npz"
    np.savez_compressed(path, **{k: v for k, v in future.items() if isinstance(v, np.ndarray)})
    pins[str(path)] = digest(path)
    write_once(
        a.output / "before-actual.json",
        dict(
            future={k: v for k, v in future.items() if not isinstance(v, np.ndarray)},
            future_artifact_hash=pins[str(path)],
            donor_hash=before,
        ),
    )
    if future["controls"] != 160 or future["body_failed"] or future["runtime_failure"] is not None:
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="CFTRACK_FUTURE_NOT_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                actual_native_steps=0,
                future_counters=forecaster.counters(),
                runtime_failure=future["runtime_failure"],
                partial_private_evidence_preserved=True,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
        return
    rows, targets = [], []
    for _ in range(160):
        target, _ = player.propose()
        row = player.world.advance({player.prefix: target})
        rows.append(row)
        targets.append(target)
        player.commit()
        if row["body_failed"][player.prefix]:
            break
    actual = {
        k: np.concatenate([r[k] for r in rows]) for k in ("qpos", "qvel", "torque", "ball_forces")
    }
    actual["joint_targets"] = np.array(targets)
    path = a.output / "actual-unvalidated.npz"
    np.savez_compressed(path, **actual)
    pins[str(path)] = digest(path)
    # Only now read source saved arrays, solely for independent provenance.
    original = a.qualification / old["artifact"]["path"]
    if original.is_symlink() or pins.get(str(original)) != digest(original):
        raise ValueError("bound source raw required after actual execution")
    with np.load(original, allow_pickle=False) as saved:
        matched_source = all(np.array_equal(v, saved[k][: len(v)]) for k, v in actual.items())
    matched_future = len(rows) == future["controls"] and all(
        np.array_equal(v, future[k]) for k, v in actual.items()
    )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_OWN_NN_NATIVE_FUTURE_EXACT"
            if matched_source and matched_future and len(rows) == 160
            else "CFTRACK_FUTURE_NOT_QUALIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            matched_source_actual_prefix=matched_source,
            matched_owned_future_actual=matched_future,
            actual_controls=len(rows),
            actual_native_steps=player.world.native_steps,
            actual_nn_calls=len(rows),
            future_counters=forecaster.counters(),
            constructor_warmups=0,
            neural_optimizer_updates=0,
            actual_enter_called_only_for_cold_entry=True,
            owned_copy_enter_or_history_reset_called=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
