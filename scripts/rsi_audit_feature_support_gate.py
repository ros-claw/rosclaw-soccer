"""Verify compact support on ALL qualified own features, not motor behavior."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("geometry-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("geometry-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    rp = a.geometry_root / "complete.json"
    if (
        a.output.exists()
        or rp.is_symlink()
        or digest(rp) != a.geometry_hash
        or (a.geometry_root / "failed.json").exists()
    ):
        raise ValueError("exact closed data geometry and unused gate audit output required")
    r = json.loads(rp.read_text())
    if (
        r["status"] != "TRAIN_RESCUE_SUPPORT_GEOMETRY_MEASURED_NOT_POLICY_OR_PHYSICS"
        or r["ambiguous_corrective_rows"] != 0
        or r["corrective_feature_rows"] != 104
        or r["protected_feature_rows"] != 27421
        or r["prospective_radius_fraction"] != 0.45
        or r["onpolicy_intervention_coverage_proven"] is not False
        or any(
            r[k] is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("same fully separated finite training-support diagnostic required")
    gp = a.geometry_root / r["geometry_path"]
    if gp.is_symlink() or gp.resolve().parent != a.geometry_root.resolve():
        raise ValueError("confined exact geometry tensor archive required")
    pins = dict(r["input_file_hashes"])
    pins.update({str(rp): a.geometry_hash, str(gp): r["geometry_hash"]})
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.feature_support_gate import FeatureSupportGate
        from rosclaw_soccer.rsi.protected_feature_geometry import protected_feature_geometry

        with np.load(gp, allow_pickle=False) as z:
            raw = {k: torch.from_numpy(z[k].copy()) for k in z.files}
        actual = protected_feature_geometry(raw["protected_features"], raw["corrective_features"])
        if any(not torch.equal(v, raw[k]) for k, v in actual.items()):
            raise ValueError("independent complete geometry recomputation differs")
        gate = FeatureSupportGate(
            raw["corrective_features"], raw["scale"], raw["prospective_radius"]
        )
        protected_values = torch.cat([gate(x) for x in raw["protected_features"].split(256)])
        corrective_values = gate(raw["corrective_features"])
        if not torch.equal(protected_values, torch.zeros_like(protected_values)) or not torch.equal(
            corrective_values, torch.ones_like(corrective_values)
        ):
            raise ValueError("all protected zeros and corrective anchor ones must be exact")
        output = a.output / "activation.npz"
        np.savez_compressed(
            output, protected=protected_values.numpy(), corrective=corrective_values.numpy()
        )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="ALL_PROTECTED_FEATURES_ZERO_RESCUE_ANCHORS_ONE_GATE_AUDITED_NOT_MOTOR",
                source_head=a.expected_head,
                input_file_hashes=pins,
                geometry_hash=a.geometry_hash,
                parent_training_hash=r["parent_training_hash"],
                protected_rows=len(protected_values),
                corrective_rows=len(corrective_values),
                protected_bitwise_zero=True,
                corrective_bitwise_one=True,
                own_geometry_independently_recomputed=True,
                activation_path=output.name,
                activation_hash=digest(output),
                gate_buffers_only_not_motor_checkpoint=True,
                physical_steps_executed=0,
                optimizer_updates=0,
                motor_protected_identity_proven=False,
                onpolicy_intervention_coverage_proven=False,
                Fresh_verified=False,
                team_game_verified=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                physical_steps_executed=0,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
