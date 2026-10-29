"""Bounded shared-world CFtrack experiment or independent own-history replay."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def validate_replay_binding(root, replay, pins, *, source_head, external_head, early_cftrack=False):
    """Reject missing or changed completed source commitments before replay."""
    from scripts.rsi_native_context_recovery_common import digest

    declaration, scene = root / "declaration.json", root / "scene.xml"
    if (
        declaration.is_symlink()
        or scene.is_symlink()
        or digest(declaration) != replay.get("declaration_hash")
        or digest(scene) != replay.get("scene_hash")
        or (root / "failed.json").exists()
    ):
        raise ValueError("source declaration and scene must be bound to terminal receipt")
    source_declaration = json.loads(declaration.read_text())
    if (
        source_declaration.get("input_file_hashes") != pins
        or source_declaration.get("source_head") != source_head
        or source_declaration.get("external_head") != external_head
        or source_declaration.get("early_cftrack", False) is not early_cftrack
    ):
        raise ValueError("exact original source and external asset pins required for replay")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("frontier-root", "external-source", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    for flag in ("frontier-head", "frontier-hash", "external-head", "expected-head"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--replay-source", type=Path)
    p.add_argument("--replay-source-hash")
    p.add_argument("--early-cftrack", action="store_true")
    a = p.parse_args()
    if (a.replay_source is None) != (a.replay_source_hash is None):
        p.error("replay requires the exact source receipt hash")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt, source_pins

    clean(a.expected_head)
    external = a.external_source.resolve()
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or subprocess.check_output(
            ["git", "-C", str(external), "rev-parse", "HEAD"], text=True
        ).strip()
        != a.external_head
        or subprocess.check_output(["git", "-C", str(external), "status", "--porcelain"])
        or any(name in sys.modules for name in ("common", "policy", "FSM"))
    ):
        raise ValueError("fresh isolated process and exact frozen sources required")
    pins, base_hash, candidates, contexts = admit_frontier(
        a.frontier_root, a.frontier_head, a.frontier_hash
    )
    if [c["seed"] for c in candidates] != [65023]:
        raise ValueError("exact retained baseline actor required")
    entries = subprocess.check_output(["git", "-C", str(external), "ls-files", "-z"]).split(b"\0")
    for entry in filter(None, entries):
        path = external / entry.decode("utf-8")
        if path.is_symlink() or not path.is_file():
            raise ValueError("ordinary complete external tracked assets required")
        pins[str(path)] = digest(path)
    for required in (
        "policy/omnicontact/model/kick_50k.onnx",
        "policy/omnicontact/config/OmniContact.yaml",
    ):
        if str(external / required) not in pins:
            raise ValueError("explicit ONNX and configuration pins required")
    source_pins(pins)
    check_pins(pins)
    replay = None
    if a.replay_source is not None:
        source_file = a.replay_source / "complete.json"
        if source_file.is_symlink() or digest(source_file) != a.replay_source_hash:
            raise ValueError("exact completed source required")
        replay = json.loads(source_file.read_text())
        if (
            replay["status"]
            != (
                "SHARED_CFTRACK_EARLY_16_PENDING_OWN_REPLAY"
                if a.early_cftrack
                else "SHARED_CFTRACK_16_PENDING_OWN_REPLAY"
            )
            or replay["source_head"] != a.expected_head
            or replay["external_head"] != a.external_head
            or len(replay["records"]) != 16
        ):
            raise ValueError("matching source run and exact source code required")
        validate_replay_binding(
            a.replay_source,
            replay,
            pins,
            source_head=a.expected_head,
            external_head=a.external_head,
            early_cftrack=a.early_cftrack,
        )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_shared_relay_trial import run_shared_cftrack_trial
    from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
    from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.native_pass_courses import NativePassCourse
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_stroke_search_common import open_motor

    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.OmniContact import OmniContact

    def policy_factory():
        policy = OmniContact(
            StateAndCmd(29),
            PolicyOutput(29),
            onnx_path=str(external / "policy/omnicontact/model/kick_50k.onnx"),
        )
        policy.task, policy.reference_source, policy.active_object_name = (
            "kickball",
            "CFgen",
            "ball",
        )
        policy.box_dims = policy.ball_dims = np.full(3, 0.11, np.float32)
        policy.bbox_scale = policy.box_dims * 2
        policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)
        return policy

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
    courses = [
        (context, side, arm)
        for context in indices
        for side in (-0.35, 0.35)
        for arm in (False, True)
    ]
    typed = []
    for context in indices:
        for side in (-0.35, 0.35):
            _, world, _, _, _ = setup.prepare(context, side)
            typed.append(
                NativePassCourse(
                    context,
                    4.0,
                    side,
                    hash_json(dict(qpos=world.data.qpos.tolist(), qvel=world.data.qvel.tolist())),
                )
            )
    qualification = preflight_receiving_courses(tuple(typed))
    probe = policy_factory()
    if probe.ort_session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("CPU-only actual reference neural session required")
    del probe
    a.output.mkdir(parents=True)
    (a.output / "scene.xml").write_text(setup.xml)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.shared_cftrack_relay.development.v1",
            early_cftrack=a.early_cftrack,
            source_head=a.expected_head,
            external_head=a.external_head,
            input_file_hashes=pins,
            courses=courses,
            preflight=qualification,
            ball_dimensions=setup.ball_dimensions,
            replay_source_hash=a.replay_source_hash,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    records = []
    for index, (context, side, arm) in enumerate(courses):
        try:
            result = run_shared_cftrack_trial(
                setup,
                actor,
                context=context,
                lateral_m=side,
                use_cftrack=arm,
                policy_factory=policy_factory,
                expected_policy_type=OmniContact,
                early_cftrack=a.early_cftrack,
            )
        except Exception as error:
            write_once(
                a.output / "failed.json",
                dict(
                    index=index,
                    context=context,
                    lateral_m=side,
                    cftrack=arm,
                    error_type=type(error).__name__,
                    error=str(error),
                    completed_worlds=len(records),
                    hardware_authorized=False,
                    promotion_authorized=False,
                    goal_complete=False,
                ),
            )
            raise
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
            if arrays:
                path = folder / f"{key}.npz"
                np.savez_compressed(path, **arrays)
                artifacts[key] = dict(path=str(path.relative_to(a.output)), hash=digest(path))
                if replay is not None:
                    prior = replay["records"][index]["artifacts"][key]
                    if prior is None or digest(a.replay_source / prior["path"]) != prior["hash"]:
                        raise ValueError("original replay input hash mismatch")
                    with np.load(a.replay_source / prior["path"], allow_pickle=False) as original:
                        if set(original.files) != set(arrays) or any(
                            not np.array_equal(original[k], v) for k, v in arrays.items()
                        ):
                            raise ValueError(f"own-history replay differs for case {index} {key}")
            else:
                artifacts[key] = None
                if replay is not None and replay["records"][index]["artifacts"][key] is not None:
                    raise ValueError("replay omitted actual source history")
        row = dict(
            index=index, context=context, lateral_m=side, cftrack=arm, artifacts=artifacts, **result
        )
        if replay is not None:
            prior = replay["records"][index]
            if any(
                row[k] != prior[k]
                for k in (
                    "context",
                    "lateral_m",
                    "cftrack",
                    "termination",
                    "controls",
                    "native_steps",
                    "handoff",
                )
            ):
                raise ValueError("own physical or history boundary outcome differs")
        records.append(row)
        write_once(a.output / f"trial{index}.json", row)
        print(
            json.dumps(
                dict(
                    index=index,
                    termination=row["termination"],
                    controls=row["controls"],
                    handoff=row["handoff"] is not None,
                )
            ),
            flush=True,
        )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "SHARED_CFTRACK_EARLY_ALL_OWN_HISTORY_MATCH"
                if replay and a.early_cftrack
                else "SHARED_CFTRACK_ALL_OWN_HISTORY_MATCH"
                if replay
                else "SHARED_CFTRACK_EARLY_16_PENDING_OWN_REPLAY"
                if a.early_cftrack
                else "SHARED_CFTRACK_16_PENDING_OWN_REPLAY"
            ),
            source_head=a.expected_head,
            external_head=a.external_head,
            producer_hash=a.replay_source_hash,
            declaration_hash=digest(a.output / "declaration.json"),
            scene_hash=digest(a.output / "scene.xml"),
            records=records,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
