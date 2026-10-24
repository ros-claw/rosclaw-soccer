"""Measure qualified expert coverage and OWN initial-feature identifiability.

This builds TRAIN labels/prototypes, not an executable exam lookup. No option
commitment controller, combined physical performance or Fresh is claimed.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("parent-own-root", "retention-feature-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("parent-own-hash", "retention-feature-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument(
        "--expert",
        nargs=3,
        action="append",
        required=True,
        metavar=("SEED", "OWN_HASH", "OWN_ROOT"),
    )
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    if a.output.exists() or [int(x[0]) for x in a.expert] != [115023, 115024, 115025]:
        raise ValueError("unused output and all3 exact learned neural experts required")
    pins = {}

    def closed(root, h):
        path = root / "complete.json"
        if path.is_symlink() or digest(path) != h or (root / "failed.json").exists():
            raise ValueError("exact complete independently qualified evidence required")
        d = json.loads(path.read_text())
        for key, value in d["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting neural expert ancestry")
            pins[key] = value
        pins[str(path)] = h
        if any(
            d[k] is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        ):
            raise ValueError("unpromoted SIM-only evidence required")
        return d

    parent = closed(a.parent_own_root, a.parent_own_hash)
    retention = closed(a.retention_feature_root, a.retention_feature_hash)
    if (
        parent["status"] != "TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH"
        or retention["status"]
        != "CURRENT_PARENT_RETENTION_OWN_FEATURES_MATCH_ACTUAL_PHYSICS_NOT_NEW_LEARNING"
        or retention["parent_training_hash"] != parent["training_hash"]
        or len(retention["records"]) != 17
    ):
        raise ValueError("whole current parent80 and actual own-motion17 feature evidence required")
    sp, source = receipt(pins, parent["producer_hash"])
    _, trained = receipt(pins, parent["training_hash"])
    _, bank = receipt(pins, trained["data_hash"])
    if (
        source["checkpoint_kind"] != "consolidated-adapter-v1"
        or len(source["records"]) != 80
        or source["training_hash"] != parent["training_hash"]
        or parent["records"]
        != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
    ):
        raise ValueError("original whole105025 parent source required")
    experts = {}
    sources = {}
    for seed, h, root in a.expert:
        own = closed(Path(root), h)
        ep, raw = receipt(pins, own["producer_hash"])
        _, t = receipt(pins, own["training_hash"])
        if (
            own["status"] != "TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH"
            or raw["checkpoint_kind"] != "consolidated-refinement-v1"
            or raw["training_hash"] != own["training_hash"]
            or t["config"]["seed"] != int(seed)
            or t["parent_training_hash"] != parent["training_hash"]
            or own["records"]
            != [{k: v for k, v in r.items() if k != "artifacts"} for r in raw["records"]]
        ):
            raise ValueError("same-parent independently replayed learned neural expert required")
        experts[int(seed)] = own["records"]
        sources[int(seed)] = (ep.parent, raw)
    for base, d in [(sp.parent, source), *sources.values()]:
        for row in d["records"]:
            art = row["artifacts"]["A"]
            path = base / art["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (base / f"trial{row['index']}").resolve()
            ):
                raise ValueError("all exact qualified own initial-state source traces required")
            pins[str(path)] = art["hash"]
    for row in retention["records"]:
        path = a.retention_feature_root / row["feature_path"]
        if path.is_symlink() or path.resolve().parent != a.retention_feature_root.resolve():
            raise ValueError("confined current motion features required")
        pins[str(path)] = row["feature_hash"]
    source_pins(pins)
    check_pins(pins)
    from rosclaw_soccer.rsi.neural_option_evidence import neural_option_evidence

    potential = neural_option_evidence(parent["records"], experts)
    a.output.mkdir()
    calls = 0
    try:
        import numpy as np
        import torch

        from scripts.rsi_stroke_search_common import open_motor

        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        _, actor = open_motor(
            pins,
            parent["training_hash"],
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="consolidated-adapter-v1",
        )
        features = []
        labels = []
        origins = []
        for i, row in enumerate(source["records"]):
            with np.load(sp.parent / row["artifacts"]["A"]["path"], allow_pickle=False) as z:
                obs = z["NN_obs"][0].copy()
                expected = z["NN_action"][0].copy()
                if z["NN_h_before"][0].any() or z["NN_c_before"][0].any():
                    raise ValueError("private zero episode start required")
            for base, d in sources.values():
                with np.load(
                    base / d["records"][i]["artifacts"]["A"]["path"], allow_pickle=False
                ) as z:
                    if not np.array_equal(z["NN_obs"][0], obs):
                        raise ValueError("different expert start is not a paired skill option")
            with torch.no_grad():
                mean, _, e = actor.forward_with_evidence(
                    torch.from_numpy(obs).reshape(1, 1, 160), actor.initial_state()
                )
            calls += 1
            if not np.array_equal(mean[0, 0].numpy(), expected):
                raise ValueError("own parent first action differs from actual source")
            features.append(e["critic_features"][0, 0].numpy().copy())
            labels.append(potential["plan"][i]["option_label"])
            origins.append(
                dict(
                    kind="public_pass",
                    state_id=row["course"]["state_id"],
                    task=row["course"]["task"],
                )
            )
        for row in retention["records"]:
            with np.load(a.retention_feature_root / row["feature_path"], allow_pickle=False) as z:
                obs = z["observations"][0].copy()
                expected = z["features"][0].copy()
            with torch.no_grad():
                _, _, e = actor.forward_with_evidence(
                    torch.from_numpy(obs).reshape(1, 1, 160), actor.initial_state()
                )
            calls += 1
            if not np.array_equal(e["critic_features"][0, 0].numpy(), expected):
                raise ValueError("current protected own initial feature differs")
            features.append(expected)
            labels.append(0)
            origins.append(dict(kind="protected_motion", context=row["context"]))
        x = torch.from_numpy(np.stack(features))
        y = torch.tensor(labels, dtype=torch.int64)
        scale = x.double().std(dim=0, unbiased=False).clamp_min(0.001)
        distance = torch.cdist(
            x.double() / scale, x.double() / scale, compute_mode="donot_use_mm_for_euclid_dist"
        ) / (288**0.5)
        nearest_other = (
            distance.masked_fill(y[:, None] == y[None, :], float("inf")).min(dim=1).values
        )
        ambiguous = nearest_other <= torch.finfo(torch.float64).eps ** 0.5
        out = a.output / "initiation.npz"
        np.savez_compressed(
            out,
            features=x.numpy(),
            labels=y.numpy(),
            scale=scale.numpy(),
            nearest_other_distance=nearest_other.numpy(),
            prospective_radius=(0.45 * nearest_other).numpy(),
        )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="NEURAL_OPTION_COVERAGE_AND_OWN_INITIAL_FEATURES_AUDITED_NOT_COMPOSED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                parent_training_hash=parent["training_hash"],
                empirical_potential=potential,
                origins=origins,
                initial_feature_rows=len(x),
                own_actor_decisions=calls,
                contradictory_initial_feature_labels=int(ambiguous.sum()),
                minimum_different_label_distance=float(nearest_other.min()),
                initiation_path=out.name,
                initiation_hash=digest(out),
                only_own_initial_features_used=True,
                state_id_task_id_are_provenance_not_selector_inputs=True,
                composition_executed=False,
                selector_implemented=False,
                physical_steps_executed=0,
                optimizer_updates=0,
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
                own_actor_decisions=calls,
                physical_steps_executed=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
