"""Bounded two-player paired router experiment or own-history native replay."""

import argparse
import json
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt, source_pins


def freeze_router(actor):
    import torch

    before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    actor.requires_grad_(False).eval()
    if (
        actor.training
        or any(p.requires_grad for p in actor.parameters())
        or any(not torch.equal(v, before[k]) for k, v in actor.state_dict().items())
    ):
        raise ValueError("router must freeze without changing any weights or buffers")
    return actor


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
    p.add_argument("--recovery-root", type=Path, required=True)
    p.add_argument("--recovery-failure-hash", required=True)
    p.add_argument("--recovery-declaration-hash", required=True)
    p.add_argument("--recovery-world-hash", required=True)
    p.add_argument("--recovery-protocol", type=Path, required=True)
    p.add_argument("--recovery-protocol-hash", required=True)
    p.add_argument("--passer-recovery-pair", action="store_true")
    p.add_argument("--study-protocol", type=Path)
    p.add_argument("--study-protocol-hash")
    p.add_argument("--baseline-root", type=Path)
    p.add_argument("--baseline-hash")
    a = p.parse_args()
    clean(a.expected_head)
    study_fields = (a.study_protocol, a.study_protocol_hash, a.baseline_root, a.baseline_hash)
    if a.passer_recovery_pair != all(study_fields) or (
        not a.passer_recovery_pair and any(study_fields)
    ):
        raise ValueError("passer recovery requires exact study protocol and baseline")
    if a.passer_recovery_pair:
        if (
            a.study_protocol_hash
            != "sha256:0a0ec2f59e9f94461b2a1f728cb4fb5d60f63fe33fcf34c54b42d484cdbd4b88"
            or a.baseline_hash
            != "sha256:93bf688c6e35c302c4a61be0e86f690227fe5a1084d04a08e127c8d27571f8bd"
        ):
            raise ValueError("exact preregistration and historical baseline required")
        baseline_file = a.baseline_root / "complete.json"
        if (
            a.study_protocol.is_symlink()
            or digest(a.study_protocol) != a.study_protocol_hash
            or baseline_file.is_symlink()
            or digest(baseline_file) != a.baseline_hash
        ):
            raise ValueError("study evidence mismatch")
        baseline_receipt = json.loads(baseline_file.read_text())
        if (
            baseline_receipt.get("status") != "ROUTER_SHARED_RELAY_SOURCE_PENDING_REPLAY"
            or len(baseline_receipt["records"]) != 16
        ):
            raise ValueError("complete historical baseline required")
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
        (a.recovery_protocol, a.recovery_protocol_hash),
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
    failed_path = a.recovery_root / "failed.json"
    prior_declaration = a.recovery_root / "declaration.json"
    prior_world_path = a.recovery_root / "trial0/complete.json"
    for path, h in (
        (failed_path, a.recovery_failure_hash),
        (prior_declaration, a.recovery_declaration_hash),
        (prior_world_path, a.recovery_world_hash),
    ):
        if path.is_symlink() or digest(path) != h:
            raise ValueError("exact failed attempt and world-zero evidence required")
    failed = json.loads(failed_path.read_text())
    prior_d = json.loads(prior_declaration.read_text())
    prior_world = json.loads(prior_world_path.read_text())
    if (
        failed.get("index") != 1
        or failed.get("arm") != "router"
        or failed.get("completed_worlds") != 1
        or failed.get("error") != "frozen CPU Dual motor and healthy actual world required"
        or prior_d.get("protocol_hash") != a.protocol_hash
        or prior_world.get("index") != 0
        or prior_world.get("arm") != "dual"
    ):
        raise ValueError("exact declared pre-router admission failure required")
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
        (a.recovery_protocol, a.recovery_protocol_hash),
        (failed_path, a.recovery_failure_hash),
        (prior_declaration, a.recovery_declaration_hash),
        (prior_world_path, a.recovery_world_hash),
    ):
        pins[str(path.resolve())] = h
    for artifact in prior_world["artifacts"].values():
        path = a.recovery_root / artifact["path"]
        if path.is_symlink() or path.resolve().parent != (a.recovery_root / "trial0").resolve():
            raise ValueError("confined old world evidence required")
        pins[str(path.resolve())] = artifact["hash"]
    if a.passer_recovery_pair:
        pins[str(a.study_protocol.resolve())] = a.study_protocol_hash
        pins[str(baseline_file.resolve())] = a.baseline_hash
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
    router = freeze_router(router)
    setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor, context_indices=indices)
    arms = ("router", "router_recovery") if a.passer_recovery_pair else ("dual", "router")
    courses = [(i, side, arm) for i in indices for side in (-0.35, 0.35) for arm in arms]
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
            passer_recovery_pair=a.passer_recovery_pair,
            study_protocol_hash=a.study_protocol_hash,
            baseline_hash=a.baseline_hash,
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
                receiver_actor=router if arm in ("router", "router_recovery") else None,
                passer_velocity_recovery=arm == "router_recovery",
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
                *(("passer_velocity",) if arm == "router_recovery" else ()),
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
            if index == 0 and not a.passer_recovery_pair:
                if {k: v for k, v in record.items() if k != "artifacts"} != {
                    k: v for k, v in prior_world.items() if k != "artifacts"
                } or set(artifacts) != set(prior_world["artifacts"]):
                    raise ValueError("recovered world-zero metadata differs")
                for key, saved in prior_world["artifacts"].items():
                    with np.load(a.recovery_root / saved["path"], allow_pickle=False) as archive:
                        if set(archive.files) != set(arrays_all[key]) or any(
                            not np.array_equal(archive[k], arrays_all[key][k])
                            for k in archive.files
                        ):
                            raise ValueError(
                                "recovered world-zero physical/history mismatch: " + key
                            )
            if replay is not None:
                saved = replay["records"][index]
                if {k: v for k, v in record.items() if k != "artifacts"} != {
                    k: v for k, v in saved.items() if k != "artifacts"
                } or set(artifacts) != set(saved["artifacts"]):
                    raise ValueError("source metadata or artifacts differ")
            if a.passer_recovery_pair and arm == "router":
                (previous,) = [
                    r
                    for r in baseline_receipt["records"]
                    if r["context"] == context and r["lateral_m"] == side and r["arm"] == "router"
                ]
                if {k: v for k, v in record.items() if k != "artifacts"} != {
                    k: v for k, v in previous.items() if k != "artifacts"
                } or set(artifacts) != set(previous["artifacts"]):
                    raise ValueError("historical router baseline metadata mismatch")
                for key, saved in previous["artifacts"].items():
                    path = a.baseline_root / saved["path"]
                    if path.is_symlink() or digest(path) != saved["hash"]:
                        raise ValueError("historical baseline artifact mismatch")
                    with np.load(path, allow_pickle=False) as archive:
                        if set(archive.files) != set(arrays_all[key]) or any(
                            not np.array_equal(archive[k], arrays_all[key][k])
                            for k in archive.files
                        ):
                            raise ValueError("historical router baseline trajectory differs")
            if arm == arms[0]:
                prior = (result, arrays_all)
            else:
                prior_result, prior_arrays = prior
                if any(
                    not np.array_equal(prior_arrays["initial"][k], arrays_all["initial"][k])
                    for k in ("qpos", "qvel")
                ):
                    raise ValueError("paired initial states differ")
                if a.passer_recovery_pair:
                    common = result["passer_recovery_boundary"]
                    observed = min(prior_result["controls"], result["controls"])
                    common = observed if common is None else min(common, observed)
                    compare_keys = ("physics", "passer_nn")
                else:
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
                    compare_keys = ("physics", "passer_nn", "receiver_velocity")
                for key in compare_keys:
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
