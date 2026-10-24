"""Whole private CPU option replay against97 qualified recorded histories.

This does not advance physics or execute saved actions. Model choices come
ONLY from own initial features; expected labels select comparison witnesses.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("feasibility-root", "retention-feature-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("feasibility-hash", "retention-feature-hash", "parent-own-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--bundle-root", type=Path)
    p.add_argument("--bundle-hash")
    p.add_argument(
        "--expert", nargs=2, action="append", required=True, metavar=("SEED", "OWN_HASH")
    )
    a = p.parse_args()
    if (a.bundle_root is None) != (a.bundle_hash is None):
        p.error("both exact bundle root and hash are required for declared-bound replay")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    fp = a.feasibility_root / "complete.json"
    rp = a.retention_feature_root / "complete.json"
    if (
        a.output.exists()
        or any(p.is_symlink() or (p.parent / "failed.json").exists() for p in (fp, rp))
        or digest(fp) != a.feasibility_hash
        or digest(rp) != a.retention_feature_hash
        or [int(e[0]) for e in a.expert] != [115023, 115024, 115025]
    ):
        raise ValueError("exact closed initiation evidence and all3 qualified experts required")
    f, r = [json.loads(p.read_text()) for p in (fp, rp)]
    if (
        f["status"] != "NEURAL_OPTION_COVERAGE_AND_OWN_INITIAL_FEATURES_AUDITED_NOT_COMPOSED"
        or f["contradictory_initial_feature_labels"] != 0
        or f["initial_feature_rows"] != 97
        or r["status"]
        != "CURRENT_PARENT_RETENTION_OWN_FEATURES_MATCH_ACTUAL_PHYSICS_NOT_NEW_LEARNING"
        or r["parent_training_hash"] != f["parent_training_hash"]
    ):
        raise ValueError("same-parent complete unambiguous own97 initiation dataset required")
    pins = dict(f["input_file_hashes"])
    pins.update({str(fp): a.feasibility_hash, str(rp): a.retention_feature_hash})
    ip = a.feasibility_root / f["initiation_path"]
    if ip.is_symlink() or ip.resolve().parent != a.feasibility_root.resolve():
        raise ValueError("confined initiation tensor archive required")
    pins[str(ip)] = f["initiation_hash"]
    _, parent_own = receipt(pins, a.parent_own_hash)
    pp, parent_source = receipt(pins, parent_own["producer_hash"])
    _, trained = receipt(pins, f["parent_training_hash"])
    _, bank = receipt(pins, trained["data_hash"])
    if parent_own["training_hash"] != f["parent_training_hash"]:
        raise ValueError("exact current parent's qualified source required")
    expert_specs = {}
    sources = {0: (pp.parent, parent_source)}
    for option, (seed, h) in enumerate(a.expert, start=1):
        _, own = receipt(pins, h)
        sp, source = receipt(pins, own["producer_hash"])
        tp, t = receipt(pins, own["training_hash"])
        if (
            own["status"] != "TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH"
            or source["checkpoint_kind"] != "consolidated-refinement-v1"
            or t["config"]["seed"] != int(seed)
            or t["parent_training_hash"] != f["parent_training_hash"]
        ):
            raise ValueError("qualified same-parent source neural experts required")
        expert_specs[option] = (tp.parent, own["training_hash"])
        sources[option] = (sp.parent, source)
    histories = []
    for choice in f["empirical_potential"]["plan"]:
        label = choice["option_label"]
        base, source = sources[label]
        row = source["records"][choice["index"]]
        path = base / row["artifacts"]["A"]["path"]
        if (
            path.is_symlink()
            or path.resolve().parent != (base / f"trial{choice['index']}").resolve()
        ):
            raise ValueError("confined selected source history required")
        pins[str(path)] = row["artifacts"]["A"]["hash"]
        histories.append(dict(scope="public_pass", label=label, paths=[path], provenance=choice))
    for row in r["records"]:
        paths = []
        for art in row["sources"]:
            path = Path(art["path"])
            if path.is_symlink() or pins.get(str(path)) != art["hash"]:
                raise ValueError("exact qualified native retention source required")
            paths.append(path)
        histories.append(
            dict(
                scope="protected_motion",
                label=0,
                paths=paths,
                provenance=dict(context=row["context"]),
            )
        )
    bundle = None
    if a.bundle_root is not None:
        bp, dp = a.bundle_root / "complete.json", a.bundle_root / "declaration.json"
        if (
            a.bundle_root.is_symlink()
            or any(p.is_symlink() for p in (bp, dp))
            or (a.bundle_root / "failed.json").exists()
            or digest(bp) != a.bundle_hash
        ):
            raise ValueError("exact closed separately assembled option bundle required")
        bundle = json.loads(bp.read_text())
        if (
            bundle["checkpoint_kind"] != "committed-neural-options-v1"
            or bundle["feasibility_hash"] != a.feasibility_hash
            or bundle["parent_training_hash"] != f["parent_training_hash"]
            or bundle["expert_training_hashes"] != [v[1] for v in expert_specs.values()]
            or bundle["source_head"] != a.expected_head
            or digest(dp) != bundle["declaration_hash"]
        ):
            raise ValueError("same declared source, parent, experts and initiation data required")
        for key, value in bundle["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting prototype/bundle input ancestry")
            pins[key] = value
        pins.update(
            {
                str(bp): a.bundle_hash,
                str(dp): bundle["declaration_hash"],
                bundle["checkpoint_path"]: bundle["checkpoint_hash"],
            }
        )
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    calls = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.committed_neural_motor import CommittedNeuralMotor
        from rosclaw_soccer.rsi.neural_option_initiation import NeuralOptionInitiation
        from rosclaw_soccer.rsi.private_motion_history import join_private_motion_history
        from rosclaw_soccer.rsi.refinement_motor_checkpoint import load_refinement_motor
        from rosclaw_soccer.sim.contracts import hash_json
        from scripts.rsi_stroke_search_common import open_motor

        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        if bundle is not None:
            _, model = open_motor(
                pins,
                a.bundle_hash,
                {i: contexts[i] for i in (1, 4, 20, 21)},
                checkpoint_kind="committed-neural-options-v1",
            )
            if (
                type(model) is not CommittedNeuralMotor
                or model.generation_hash != bundle["generation_hash"]
            ):
                raise ValueError("generic loader must return the exact audited option model")
            generation = model.generation_hash
        else:
            _, parent = open_motor(
                pins,
                f["parent_training_hash"],
                {i: contexts[i] for i in (1, 4, 20, 21)},
                checkpoint_kind="consolidated-adapter-v1",
            )
            experts = {}
            for option, (root, h) in expert_specs.items():
                experts[option], _ = load_refinement_motor(
                    parent.memory.parent,
                    parent_lineage_hash=trained["parent_lineage_hash"],
                    root=root,
                    expected_receipt_hash=h,
                )
            with np.load(ip, allow_pickle=False) as z:
                initial = {k: z[k].copy() for k in z.files}
            initiation = NeuralOptionInitiation(
                torch.from_numpy(initial["features"]), torch.from_numpy(initial["labels"])
            )
            if not torch.equal(
                initiation.scale, torch.from_numpy(initial["scale"])
            ) or not torch.equal(initiation.radii, torch.from_numpy(initial["prospective_radius"])):
                raise ValueError("own initiation geometry changed")
            generation = str(
                hash_json(
                    dict(
                        schema="prototype.committed_neural_motor.v1",
                        source_head=a.expected_head,
                        feasibility_hash=a.feasibility_hash,
                        parent_training_hash=f["parent_training_hash"],
                        experts={k: v[1] for k, v in expert_specs.items()},
                    )
                )
            )
            model = CommittedNeuralMotor(parent, experts, initiation, generation_hash=generation)
        records = []
        for index, history in enumerate(histories):
            phases = []
            for path in history["paths"]:
                with np.load(path, allow_pickle=False) as z:
                    phases.append({k: z[k].copy() for k in z.files if k.startswith("NN_")})
            raw = phases[0] if len(phases) == 1 else join_private_motion_history(*phases)
            state = model.initial_state()
            for i, obs in enumerate(raw["NN_obs"]):
                if any(
                    not np.array_equal(t.numpy(), raw[k][i])
                    for t, k in zip(state.recurrent, ("NN_h_before", "NN_c_before"), strict=True)
                ):
                    raise ValueError(
                        "own committed recurrent prehistory differs from qualified source"
                    )
                mean, state, e = model.forward_with_evidence(
                    torch.from_numpy(obs).reshape(1, 1, 160), state
                )
                calls += 1
                if (
                    state.option != history["label"]
                    or state.controls != i + 1
                    or not np.array_equal(mean[0, 0].numpy(), raw["NN_action"][i])
                    or any(
                        not np.array_equal(t.numpy(), raw[k][i])
                        for t, k in zip(state.recurrent, ("NN_h_after", "NN_c_after"), strict=True)
                    )
                    or (e["final_mean_delta_radians"].abs() > 0.2).any()
                ):
                    raise ValueError(
                        "own option/action/state/total original motor authority differs"
                    )
            record = dict(
                index=index,
                scope=history["scope"],
                provenance=history["provenance"],
                rows=len(raw["NN_obs"]),
                option=state.option,
                all_source_actions_bitwise_equal=True,
                all_own_hc_bitwise_equal=True,
                option_committed_through_recovery=True,
            )
            records.append(record)
            print(json.dumps(record), flush=True)
        if len(records) != 97:
            raise ValueError(
                "all80 selected physical histories+17current motion histories required"
            )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS"
                    if bundle is not None
                    else "COMMITTED_NEURAL_OPTIONS_OWN97_ACTIONS_MATCH_SOURCES_NOT_PHYSICS"
                ),
                source_head=a.expected_head,
                input_file_hashes=pins,
                feasibility_hash=a.feasibility_hash,
                generation_hash=generation,
                records=records,
                own_actor_decisions=calls,
                physical_steps_executed=0,
                optimizer_updates=0,
                selected_neural_skills_have_exact_actions_on_recorded_streams=True,
                combined_physical_performance_proven=False,
                native_checkpoint_admitted=bundle is not None,
                old_retention_proven=False,
                **(
                    dict(
                        checkpoint_kind="committed-neural-options-v1",
                        source_hash=a.bundle_hash,
                        checkpoint_hash=bundle["checkpoint_hash"],
                        data_hash=bundle["data_hash"],
                        declared_reference_bundle_loaded=True,
                    )
                    if bundle is not None
                    else {}
                ),
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
