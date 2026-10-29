"""Bounded two-player paired router experiment or own-history native replay."""

import argparse
import json
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt, source_pins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("frontier-root", "router-source", "router-history", "protocol", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "frontier-head",
        "frontier-hash",
        "router-source-hash",
        "router-history-hash",
        "protocol-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--replay-source", type=Path)
    p.add_argument("--replay-source-hash")
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or (a.replay_source is None) != (a.replay_source_hash is None)
    ):
        raise ValueError("unused external output and explicit replay binding required")
    for path, h in (
        (a.router_source, a.router_source_hash),
        (a.router_history, a.router_history_hash),
        (a.protocol, a.protocol_hash),
    ):
        if path.is_symlink() or digest(path) != h:
            raise ValueError("exact router source, history and protocol required")
    old = json.loads(a.router_source.read_text())
    history = json.loads(a.router_history.read_text())
    if (
        old["arm"] != "progressive"
        or old["promotion_authorized"] is not False
        or history["status"] != "CONSUMED_ROUTER_HISTORY_EXACT"
        or history["source_hash"] != a.router_source_hash
        or history["controls"] != 260
        or history["matched"] != 260
        or history["runtime_failure"] is not None
    ):
        raise ValueError("own-history-qualified exact unpromoted router required")
    old_declaration = a.router_source.parent / "declaration.json"
    if digest(old_declaration) != old["declaration_hash"]:
        raise ValueError("exact historical source declaration required")
    old_config = json.loads(old_declaration.read_text())
    fresh_root = Path(old_config["fresh_root"])
    fresh_receipt, fresh_declaration = fresh_root / "complete.json", fresh_root / "declaration.json"
    if digest(fresh_receipt) != old_config["fresh_hash"]:
        raise ValueError("consumed original declaration source required")
    fresh = json.loads(fresh_receipt.read_text())
    if digest(fresh_declaration) != fresh["declaration_hash"]:
        raise ValueError("exact frozen router declaration required")
    router_config = json.loads(fresh_declaration.read_text())
    pins, base_hash, candidates, contexts = admit_frontier(
        a.frontier_root, a.frontier_head, a.frontier_hash
    )
    if [c["seed"] for c in candidates] != [65023]:
        raise ValueError("exact acquired parent required")
    for path, value in old["input_file_hashes"].items():
        if path in pins and pins[path] != value:
            raise ValueError("source lineages disagree")
        pins[path] = value
    for path, h in (
        (a.router_source, a.router_source_hash),
        (a.router_history, a.router_history_hash),
        (old_declaration, old["declaration_hash"]),
        (fresh_receipt, old_config["fresh_hash"]),
        (fresh_declaration, fresh["declaration_hash"]),
        (a.protocol, a.protocol_hash),
    ):
        pins[str(path.resolve())] = h
    source_pins(pins)
    check_pins(pins)
    import rosclaw

    if str(Path(rosclaw.__file__).resolve()) not in pins:
        raise ValueError("loaded ROSClaw core must be in the verified source lineage")
    replay = None
    if a.replay_source is not None:
        rp = a.replay_source / "complete.json"
        if rp.is_symlink() or digest(rp) != a.replay_source_hash:
            raise ValueError("exact source receipt required")
        replay = json.loads(rp.read_text())
        prior_declaration = a.replay_source / "declaration.json"
        if (
            replay["status"] != "ROUTER_SHARED_RELAY_SOURCE_PENDING_REPLAY"
            or replay["source_head"] != a.expected_head
            or len(replay["records"]) != 16
            or digest(prior_declaration) != replay["declaration_hash"]
            or json.loads(prior_declaration.read_text())["input_file_hashes"] != pins
        ):
            raise ValueError("exact source code, inputs and 16 original worlds required")
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_shared_relay_trial import run_shared_cftrack_trial
    from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
    from rosclaw_soccer.rsi.paid_ancestor_router_student import FrozenAncestralRouterActor
    from rosclaw_soccer.rsi.paid_fresh_evidence import load_fresh_actor
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
    policies = [
        Path(x) for x in pins if x.endswith("/paid-reference-source-20261021/ckp/policy_30000.onnx")
    ]
    if len(policies) != 1:
        raise ValueError("unique exact trained router reference policy required")
    router, variant = load_fresh_actor(policies[0], router_config, "progressive")
    if type(router) is not FrozenAncestralRouterActor or variant != 2:
        raise ValueError("exact frozen router required")
    setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor, context_indices=indices)
    courses = [
        (i, side, arm) for i in indices for side in (-0.35, 0.35) for arm in ("dual", "router")
    ]
    typed = []
    for i in indices:
        for side in (-0.35, 0.35):
            _, w, _, _, _ = setup.prepare(i, side)
            typed.append(
                NativePassCourse(
                    i,
                    4.0,
                    side,
                    hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist())),
                )
            )
    qualification = preflight_receiving_courses(tuple(typed))
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            protocol_hash=a.protocol_hash,
            input_file_hashes=pins,
            courses=courses,
            preflight=qualification,
            ball_dimensions=setup.ball_dimensions,
            own_replay_source_hash=a.replay_source_hash,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    records = []
    prior = None
    for index, (context, side, arm) in enumerate(courses):
        try:
            result = run_shared_cftrack_trial(
                setup,
                actor,
                context=context,
                lateral_m=side,
                use_cftrack=False,
                policy_factory=lambda: None,
                expected_policy_type=object,
                receiver_actor=router if arm == "router" else None,
            )
            folder = a.output / f"trial{index}"
            folder.mkdir()
            artifacts = {}
            arrays_all = {}
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
                arrays_all[key] = arrays
                if not arrays:
                    continue
                path = folder / f"{key}.npz"
                np.savez_compressed(path, **arrays)
                artifacts[key] = dict(path=str(path.relative_to(a.output)), hash=digest(path))
                if replay is not None:
                    saved = replay["records"][index]["artifacts"].get(key)
                    if saved is None or digest(a.replay_source / saved["path"]) != saved["hash"]:
                        raise ValueError("source replay artifact commitment mismatch")
                    with np.load(a.replay_source / saved["path"], allow_pickle=False) as archive:
                        if set(archive.files) != set(arrays) or any(
                            not np.array_equal(archive[k], arrays[k]) for k in arrays
                        ):
                            raise ValueError("own-history or physical replay differs: " + key)
            record = dict(
                index=index, context=context, lateral_m=side, arm=arm, **result, artifacts=artifacts
            )
            write_once(folder / "complete.json", record)
            if replay is not None:
                saved = replay["records"][index]
                if {k: v for k, v in record.items() if k != "artifacts"} != {
                    k: v for k, v in saved.items() if k != "artifacts"
                } or set(artifacts) != set(saved["artifacts"]):
                    raise ValueError("source metadata or artifacts differ")
            if arm == "dual":
                prior = (result, arrays_all)
            else:
                prior_result, prior_arrays = prior
                if any(
                    not np.array_equal(prior_arrays["initial"][k], arrays_all["initial"][k])
                    for k in ("qpos", "qvel")
                ):
                    raise ValueError("paired initial states differ")
                handoff_a, handoff_b = prior_result["handoff"], result["handoff"]
                if (handoff_a is None) != (handoff_b is None) or (
                    handoff_a and handoff_a["control"] != handoff_b["control"]
                ):
                    raise ValueError("paired readiness/handoff differs")
                common = (
                    handoff_a["control"]
                    if handoff_a
                    else min(prior_result["controls"], result["controls"])
                )
                for key in ("physics", "passer_nn", "receiver_velocity"):
                    x, y = prior_arrays[key], arrays_all[key]
                    if set(x) != set(y) or any(
                        not np.array_equal(x[k][:common], y[k][:common]) for k in x
                    ):
                        raise ValueError(
                            "paired pre-handoff physical/neural prefix differs: " + key
                        )
            records.append(record)
        except Exception as error:
            write_once(
                a.output / "failed.json",
                dict(
                    index=index,
                    context=context,
                    lateral_m=side,
                    arm=arm,
                    error_type=type(error).__name__,
                    error=str(error),
                    completed_worlds=len(records),
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
            status="ROUTER_SHARED_RELAY_OWN_REPLAY_EXACT"
            if replay
            else "ROUTER_SHARED_RELAY_SOURCE_PENDING_REPLAY",
            source_head=a.expected_head,
            declaration_hash=digest(a.output / "declaration.json"),
            source_hash=a.replay_source_hash,
            records=records,
            worlds=len(records),
            hardware_authorized=False,
            promotion_authorized=False,
            team_game_verified=False,
        ),
    )


if __name__ == "__main__":
    main()
