"""TRAIN-only separability and actual parameter-effect diagnosis, no new physics or fit."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--data-hash", required=True)
    p.add_argument("--trained-roots", type=Path, nargs=3, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or digest(a.data_root / "complete.json") != a.data_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused diagnosis, exact TRAIN data and frozen source required")
    data = json.loads((a.data_root / "complete.json").read_text())
    pins = dict(data["input_file_hashes"])
    f = Path(data["feature_path"])
    if digest(f) != data["feature_hash"]:
        raise ValueError("exact causal feature archive required")
    for path in (f, a.data_root / "complete.json", a.data_root / "declaration.json"):
        pins[str(path)] = digest(path)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.paid_full_entry_student import validate_full_entry_data

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with np.load(f, allow_pickle=False) as archive:
        z = {k: archive[k].copy() for k in archive.files}
    validate_full_entry_data(data, json.loads((a.data_root / "declaration.json").read_text()), z)
    a.output.mkdir()
    pos_ids, neg_ids = np.flatnonzero(z["teacher"]), np.flatnonzero(~z["teacher"])
    positives = torch.from_numpy(z["features"][pos_ids].astype(np.float64))
    negatives = torch.from_numpy(z["features"][neg_ids].astype(np.float64))
    nearest = np.empty(len(pos_ids), np.float64)
    distance_pairs = 0
    with torch.no_grad():
        for start in range(0, len(positives), 16):
            x = positives[start : start + 16]
            minimum = torch.full((len(x),), float("inf"), dtype=torch.float64)
            for j in range(0, len(negatives), 512):
                n = negatives[j : j + 512]
                d = (x[:, None, :] - n[None, :, :]).square().sum(dim=-1)
                minimum = torch.minimum(minimum, d.min(dim=1).values)
                distance_pairs += len(x) * len(n)
            nearest[start : start + len(x)] = minimum.numpy()
    cases = []
    for r in data["trajectories"]:
        if r["role"] not in ("novel_first_cartesian_teacher", "canonical_first_R2_teacher"):
            continue
        mask = (pos_ids >= r["start"]) & (pos_ids < r["stop"])
        values = nearest[mask]
        active = z["active"][pos_ids[mask]]
        cases.append(
            dict(
                role=r["role"],
                case=r["case"],
                rows=len(values),
                exact_protected_feature_collisions=int((values == 0).sum()),
                active_exact_collisions=int(((values == 0) & active).sum()),
                minimum_distance=float(np.sqrt(values.min())),
                distance_quantiles=np.quantile(np.sqrt(values), [0, 0.1, 0.5, 0.9, 1]).tolist(),
            )
        )
    effects = []
    for seed, root in enumerate(a.trained_roots):
        rp = root / "complete.json"
        r = json.loads(rp.read_text())
        cp = Path(r["checkpoint_path"])
        if (
            r["config"]["seed_index"] != seed
            or r["data_hash"] != a.data_hash
            or digest(cp) != r["checkpoint_hash"]
            or cp.is_symlink()
            or cp.stat().st_size > 16000000
        ):
            raise ValueError("exact predeclared trained student required")
        pins[str(rp)] = digest(rp)
        pins[str(cp)] = digest(cp)
        state = torch.load(cp, weights_only=True, map_location="cpu")
        final = [state["plastic"][k] for k in ("net.4.weight", "net.4.bias")]
        delta = max(
            float((v - state["initial_plastic"][k]).abs().max())
            for k, v in state["plastic"].items()
            if k.startswith("net.")
        )
        effects.append(
            dict(
                seed=seed,
                protocol=r["config"]["protocol"],
                final_layer_exact_zero=all(v.count_nonzero() == 0 for v in final),
                maximum_head_parameter_change=delta,
                active_gate_open_rows=r["retention_certificate"]["active_gate_open_rows"],
                teacher_gate_open_rows=r["retention_certificate"].get("teacher_gate_open_rows"),
                optimizer_updates=r["counts"]["optimizer_updates"],
            )
        )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="PAID_FULL_ENTRY_TRAIN_SUPPORT_GEOMETRY_AND_WEIGHT_EFFECT_DIAGNOSIS",
            source_head=a.expected_head,
            input_file_hashes=pins,
            data_hash=a.data_hash,
            cases=cases,
            weight_effects=effects,
            pairwise_feature_distances=distance_pairs,
            distance_dtype="float64 direct differences, no norm-square subtraction cancellation",
            teacher_rows=len(positives),
            protected_rows=len(negatives),
            Fresh_rows_read=0,
            own_native_steps=0,
            own_neural_rows=0,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
