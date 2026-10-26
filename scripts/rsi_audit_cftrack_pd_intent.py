"""Read-only PD intent audit of an independently NN/native-verified H300 bank."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--replay-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.bank / "complete.json"
    if (
        a.output.exists()
        or a.replay.is_symlink()
        or complete.is_symlink()
        or digest(a.replay) != a.replay_hash
    ):
        raise ValueError("unused output and independently replayed bank required")
    replay = json.loads(a.replay.read_text())
    bank = json.loads(complete.read_text())
    if (
        replay["status"] != "CFTRACK_STATIC_BANK_NATIVE_REPLAY_EXACT"
        or replay["source_bank_hash"] != digest(complete)
        or len(replay["records"]) != 10
        or len(bank["records"]) != 10
        or not all(r["exact"] for r in replay["records"])
    ):
        raise ValueError("complete paired H300 bank and exact independent replay required")
    pins = dict(replay["input_file_hashes"])
    pins[str(a.replay)] = a.replay_hash
    pins[str(complete)] = digest(complete)
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(s) for s in pins if s.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique verified physical scene required")
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.pd_intent_audit import audit_pd_intent
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if compiled_model_hash(model) != (
        "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact independently verified compiled body required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            new_neural_calls=0,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            measured_soft_stop_penetration_reported_without_changing_contract=True,
        ),
    )
    records = []
    for source, verified in zip(bank["records"], replay["records"], strict=True):
        if (source["case"], source["arm"]) != (verified["case"], verified["arm"]):
            raise ValueError("source and replay row identity mismatch")
        path = a.bank / source["artifact"]["path"]
        if (
            path.parent != a.bank
            or path.is_symlink()
            or source["artifact"]["hash"] != digest(path)
            or pins.get(str(path)) != digest(path)
        ):
            raise ValueError("unchanged source native evidence required")
        with np.load(path, allow_pickle=False) as saved:
            report = audit_pd_intent(
                decoded=saved["nn_decoded_targets"],
                targets=saved["joint_targets"],
                positions=saved["qpos"][:, layout.qi],
                torques=saved["torque"][:, layout.ai],
                limits=model.jnt_range[layout.joints].copy(),
                effort=0.85 * np.array(G1_HARD_TORQUE_LIMITS),
            )
        if report["controls"] != 300 or report["effort_cap_exceeded"]:
            raise ValueError("complete H300 and unchanged effort cap required")
        records.append(dict(case=source["case"], arm=source["arm"], **report))
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_VERIFIED_BANK_PD_INTENT_AUDITED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_bank_hash=digest(complete),
            independent_replay_hash=a.replay_hash,
            records=records,
            new_neural_calls=0,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            position_residual_guard_unchanged=True,
            torque_interface_not_yet_qualified=True,
            goal_complete=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
