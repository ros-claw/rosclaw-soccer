"""Measure own-feature support conflicts before another motor learning family.

Finite training-support separation is not on-policy reachability, generalization,
physical success or permission to deploy a gate. No actor or simulator is run.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("feature-root", "retention-feature-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("feature-hash", "retention-feature-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    fp, rp = [r / "complete.json" for r in (a.feature_root, a.retention_feature_root)]
    if (
        a.output.exists()
        or digest(fp) != a.feature_hash
        or any(p.is_symlink() or (p.parent / "failed.json").exists() for p in (fp, rp))
    ):
        raise ValueError("unused output and closed immutable own-feature predecessors required")
    f, r = [json.loads(p.read_text()) for p in (fp, rp)]
    if (
        f["status"] != "PASS_REFINEMENT_OWN_FEATURES_ZERO_IDENTITY_NOT_PHYSICS"
        or r["status"]
        != "CURRENT_PARENT_RETENTION_OWN_FEATURES_MATCH_ACTUAL_PHYSICS_NOT_NEW_LEARNING"
        or r["source_head"] != a.retention_feature_head
        or r["feature_hash"] != a.feature_hash
        or f["parent_training_hash"] != r["parent_training_hash"]
        or f["parent_checkpoint_hash"] != r["parent_checkpoint_hash"]
        or r["onpolicy_for_current_parent"] is not True
        or len(f["records"]) != 62
        or len(r["records"]) != 17
        or any(
            d[k] is not False
            for d in (f, r)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("same parent and complete old17/current41/rescue4/current-ret17 required")
    pins = dict(f["input_file_hashes"])
    for key, value in r["input_file_hashes"].items():
        if key in pins and pins[key] != value:
            raise ValueError("conflicting feature ancestry")
        pins[key] = value
    pins.update({str(fp): a.feature_hash, str(rp): digest(rp)})
    sources = []
    for root, rows in ((a.feature_root, f["records"]), (a.retention_feature_root, r["records"])):
        for row in rows:
            path = root / row["feature_path"]
            if path.is_symlink() or path.resolve().parent != root.resolve():
                raise ValueError("confined ordinary feature archive required")
            pins[str(path)] = row["feature_hash"]
            sources.append((root, row, path))
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.protected_feature_geometry import protected_feature_geometry

        protected, corrective, owners = [], [], []
        groups = {}
        for root, row, path in sources:
            with np.load(path, allow_pickle=False) as z:
                features = z["features"].copy()
                if (
                    features.shape != (row["rows"], 288)
                    or features.dtype != np.float32
                    or not np.isfinite(features).all()
                ):
                    raise ValueError("whole finite own288 feature rows required")
                role = row["role"] if root == a.feature_root else "retained_current_motion17"
                groups[role] = groups.get(role, 0) + 1
                if role == "qualified_rescue":
                    difference = np.max(np.abs(z["target_delta"] - z["parent_delta"]), axis=1)
                    active = np.flatnonzero(difference >= 0.001)
                    if not len(active):
                        raise ValueError(
                            "every rescue must contain a measurable actual corrective action"
                        )
                    corrective.append(features[active])
                    owners.extend([[row["index"], int(i)] for i in active])
                else:
                    protected.append(features)
        if groups != dict(
            retained_original17=17,
            retained_current41=41,
            qualified_rescue=4,
            retained_current_motion17=17,
        ):
            raise ValueError("all75 protected and4 real-rescue histories required")
        protection = torch.from_numpy(np.concatenate(protected))
        correction = torch.from_numpy(np.concatenate(corrective))
        geometry = protected_feature_geometry(protection, correction)
        out = a.output / "geometry.npz"
        np.savez_compressed(
            out,
            protected_features=protection.numpy(),
            corrective_features=correction.numpy(),
            corrective_row_owners=np.array(owners),
            **{k: v.numpy() for k, v in geometry.items()},
        )
        distance = geometry["nearest_protected_distance"]
        report = dict(
            status="TRAIN_RESCUE_SUPPORT_GEOMETRY_MEASURED_NOT_POLICY_OR_PHYSICS",
            source_head=a.expected_head,
            input_file_hashes=pins,
            feature_hash=a.feature_hash,
            retention_feature_hash=digest(rp),
            parent_training_hash=f["parent_training_hash"],
            groups=groups,
            protected_feature_rows=len(protection),
            corrective_feature_rows=len(correction),
            separable_corrective_rows=int(geometry["separable"].sum()),
            ambiguous_corrective_rows=int((~geometry["separable"]).sum()),
            nearest_protected_normalized_RMS=dict(
                minimum=float(distance.min()),
                median=float(distance.median()),
                maximum=float(distance.max()),
            ),
            prospective_radius_fraction=0.45,
            active_correction_threshold_rad=0.001,
            geometry_path=out.name,
            geometry_hash=digest(out),
            finite_training_support_only=True,
            onpolicy_intervention_coverage_proven=False,
            physics_success_proven=False,
            gate_implemented=False,
            physical_steps_executed=0,
            optimizer_updates=0,
            Fresh_rows_read=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        )
        check_pins(pins)
        clean(a.expected_head)
        write_once(a.output / "complete.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "input_file_hashes"}), flush=True)
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
