"""Replay eight old Dual worlds with passive actual-contact witnesses, SIM only."""

import argparse
import json
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt, source_pins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("frontier-root", "source-root", "replay-root", "protocol", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    for name in (
        "frontier-head",
        "frontier-hash",
        "source-hash",
        "replay-hash",
        "protocol-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    paths = [a.source_root / "complete.json", a.replay_root / "complete.json", a.protocol]
    for path, expected in zip(paths, (a.source_hash, a.replay_hash, a.protocol_hash), strict=True):
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact historical receipts and predeclared protocol required")
    source, replay = (json.loads(path.read_text()) for path in paths[:2])
    if (
        source["status"] != "ROUTER_SHARED_RELAY_SOURCE_PENDING_REPLAY"
        or replay["status"] != "ROUTER_SHARED_RELAY_OWN_REPLAY_EXACT"
        or replay["source_hash"] != a.source_hash
        or len(source["records"]) != 16
        or len(replay["records"]) != 16
        or source["hardware_authorized"] is not False
        or source["promotion_authorized"] is not False
    ):
        raise ValueError("exact completed source and own replay required")
    declaration = a.source_root / "declaration.json"
    if declaration.is_symlink() or digest(declaration) != source["declaration_hash"]:
        raise ValueError("source declaration mismatch")
    pins, base_hash, candidates, contexts = admit_frontier(
        a.frontier_root, a.frontier_head, a.frontier_hash
    )
    if [x["seed"] for x in candidates] != [65023]:
        raise ValueError("original acquired parent required")
    old_pins = json.loads(declaration.read_text())["input_file_hashes"]
    for name, expected in pins.items():
        if name in old_pins and old_pins[name] != expected:
            raise ValueError("source lineage mismatch")
    # Bind historical dependencies but not historical checkout paths: the new
    # checkout deliberately instruments source while retaining all asset hashes.
    for name, expected in old_pins.items():
        if "/src/rosclaw_soccer/" not in name and "/scripts/" not in name:
            pins[name] = expected
    for path in [*paths, declaration]:
        pins[str(path.resolve())] = digest(path)
    source_pins(pins)
    check_pins(pins)
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_shared_relay_trial import run_shared_cftrack_trial
    from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
    from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.native_pass_courses import NativePassCourse
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_stroke_search_common import open_motor

    indices = (2, 3, 5, 7)
    bench, base = open_motor(pins, base_hash, {i: contexts[i] for i in indices})
    candidate = candidates[0]
    _, training = receipt(pins, candidate["training_hash"])
    actor, _ = load_dual_candidate_for_sim_audit(
        base.parent,
        parent_lineage_hash=training["parent_lineage_hash"],
        root=Path(candidate["training_root"]),
        expected_receipt_hash=candidate["training_hash"],
        checkpoint_kind=candidate["kind"],
    )
    setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor, context_indices=indices)
    courses = [(i, side) for i in indices for side in (-0.35, 0.35)]
    typed = []
    for i, side in courses:
        _, w, _, _, _ = setup.prepare(i, side)
        typed.append(
            NativePassCourse(
                i, 4.0, side, hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist()))
            )
        )
    preflight = preflight_receiving_courses(tuple(typed))
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            courses=courses,
            preflight=preflight,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    records = []
    for index, (context, side) in enumerate(courses):
        try:
            old = source["records"][index * 2]
            old_replay = replay["records"][index * 2]
            if (old["index"], old["context"], old["lateral_m"], old["arm"]) != (
                index * 2,
                context,
                side,
                "dual",
            ):
                raise ValueError("exact declared historical world required")
            result = run_shared_cftrack_trial(
                setup,
                actor,
                context=context,
                lateral_m=side,
                use_cftrack=False,
                policy_factory=lambda: None,
                expected_policy_type=object,
                record_witness=True,
            )
            witness = result.pop("witness")
            folder = a.output / f"trial{index}"
            folder.mkdir()
            artifacts = {}
            for key in (
                "initial",
                "physics",
                "passer_nn",
                "receiver_return_nn",
                "receiver_shadow",
                "receiver_velocity",
                "cftrack",
            ):
                arrays = result.pop(key)
                for prior, root in ((old, a.source_root), (old_replay, a.replay_root)):
                    saved = prior["artifacts"].get(key)
                    if bool(arrays) != bool(saved):
                        raise ValueError("artifact presence mismatch: " + key)
                    if saved:
                        path = root / saved["path"]
                        if (
                            path.is_symlink()
                            or not path.resolve().is_relative_to(root.resolve())
                            or digest(path) != saved["hash"]
                        ):
                            raise ValueError("historical artifact hash mismatch")
                        with np.load(path, allow_pickle=False) as archive:
                            if set(archive.files) != set(arrays) or any(
                                not np.array_equal(archive[k], arrays[k]) for k in arrays
                            ):
                                raise ValueError("physical/neural replay mismatch: " + key)
                if arrays:
                    path = folder / f"{key}.npz"
                    np.savez_compressed(path, **arrays)
                    artifacts[key] = dict(path=str(path.relative_to(a.output)), hash=digest(path))
            for prior in (old, old_replay):
                if any(
                    result[k] != prior[k]
                    for k in ("termination", "controls", "native_steps", "handoff")
                ):
                    raise ValueError("historical trial metadata mismatch")
            witness_path = folder / "witness.json"
            write_once(witness_path, witness)
            record = dict(
                index=index,
                context=context,
                lateral_m=side,
                **result,
                artifacts=artifacts,
                witness=dict(
                    path=str(witness_path.relative_to(a.output)), hash=digest(witness_path)
                ),
            )
            write_once(folder / "complete.json", record)
            records.append(record)
        except Exception as error:
            write_once(
                a.output / "failed.json",
                dict(
                    index=index,
                    context=context,
                    lateral_m=side,
                    completed_worlds=len(records),
                    error_type=type(error).__name__,
                    error=str(error),
                    hardware_authorized=False,
                    promotion_authorized=False,
                ),
            )
            raise
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="PASSIVE_WITNESS_REPLAY_EXACT_DEVELOPMENT_ONLY",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            replay_hash=a.replay_hash,
            declaration_hash=digest(a.output / "declaration.json"),
            records=records,
            worlds=len(records),
            hardware_authorized=False,
            promotion_authorized=False,
            team_game_verified=False,
        ),
    )


if __name__ == "__main__":
    main()
