"""Build paired, pre-intervention decision data from audited SIM relay worlds."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.relay_balance_context import relay_balance_context
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import measured, report, verify_same_handoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", nargs=3, action="append", required=True, metavar=("SOURCE", "AUDIT", "AUDIT_HASH")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        len(args.pair) != 6
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("all three original and expanded audited blocks required")
    seen, conditions, source_pins, teacher_identity, samples, rows, frozen = (
        set(),
        set(),
        [],
        None,
        [],
        [],
        [],
    )
    for root_name, audit_name, audit_hash in args.pair:
        root, audit_path = Path(root_name), Path(audit_name)
        declaration_path = root / "declaration.json"
        declaration = json.loads(declaration_path.read_text())
        protocol = declaration["protocol"]
        expanded = protocol == "soccer.neural_relay_balance_expanded.development.v6"
        if not expanded and protocol != "soccer.neural_relay_balance_teacher.development.v5":
            raise ValueError("declared balance candidate protocol required")
        status = (
            "NEURAL_RELAY_BALANCE_EXPANDED64_COLLECTED_PENDING_OWN_REPLAY"
            if expanded
            else "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
        )
        source, _, source_hash = report(
            root, audit_path, audit_hash, status, worlds=64 if expanded else 32
        )
        block = declaration["development_context_block"]
        if (
            type(block) is not int
            or block not in (0, 1, 2)
            or (block, expanded) in seen
            or digest(declaration_path) != source["declaration_hash"]
        ):
            raise ValueError("unique exact block and declaration required")
        seen.add((block, expanded))
        teachers = declaration["balance_teacher_pairs"]
        if teacher_identity is None:
            teacher_identity = teachers
        if teachers != teacher_identity:
            raise ValueError("all examples must share the same three teacher identities")
        xmls = [
            Path(p)
            for p in source["input_file_hashes"]
            if p.endswith("/cached-standard/declared-scene.xml")
        ]
        if (
            len(xmls) != 1
            or digest(xmls[0]) != source["input_file_hashes"][str(xmls[0])]
            or (root / "scene.xml").read_text()
            != shared_native_scene(xmls[0].read_text(), prefixes=("a/", "b/"))
        ):
            raise ValueError("exact pinned physical scene required")
        model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        entries = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["balance_teacher"],
            ): r
            for r in source["records"]
        }
        if len(entries) != (64 if expanded else 32):
            raise ValueError("all unique paired arms required")
        for context, side, arm in sorted(entries):
            if arm != -1:
                continue
            key = (context, side)
            if key in conditions:
                raise ValueError("no repeated physical initial condition")
            conditions.add(key)
            base = entries[(context, side, -1)]
            candidates = [entries[(context, side, i)] for i in (0, 1, 2)]
            for candidate in candidates:
                verify_same_handoff(root, base, candidate)
            metrics = [measured(root, r) for r in (base, *candidates)]
            if base["handoff"] is None:
                rows.append(
                    dict(
                        context=context,
                        side=side,
                        block=block,
                        expanded=expanded,
                        handoff=False,
                        metrics=metrics,
                    )
                )
                continue
            boundary = root / base["handoff"]["boundary"]["path"]
            if (
                boundary.is_symlink()
                or boundary.resolve().parent != (root / f"trial{base['index']}").resolve()
                or digest(boundary) != base["handoff"]["boundary"]["hash"]
            ):
                raise ValueError("exact own physical handoff boundary required")
            with np.load(boundary, allow_pickle=False) as f:
                if not np.array_equal(f["before_qpos"], f["after_qpos"]) or not np.array_equal(
                    f["before_qvel"], f["after_qvel"]
                ):
                    raise ValueError("physical state changed across PD handoff")
                data = SimpleNamespace(qpos=f["before_qpos"], qvel=f["before_qvel"])
                q, v = layout.canonical(data)
                goal = np.asarray(base["handoff"]["return_goal"], dtype=np.float64)
                features = relay_balance_context(q, v, goal)
            samples.append(dict(features=features, context=context, side=side, metrics=metrics))
            rows.append(
                dict(
                    context=context,
                    side=side,
                    block=block,
                    expanded=expanded,
                    handoff=True,
                    boundary_hash=base["handoff"]["boundary"]["hash"],
                    metrics=metrics,
                )
            )
        source_pins.append(
            dict(block=block, expanded=expanded, source_hash=source_hash, audit_hash=audit_hash)
        )
        frozen.append((root / "complete.json", source_hash, audit_path, audit_hash))
    if (
        seen != {(i, expanded) for i in (0, 1, 2) for expanded in (False, True)}
        or len(conditions) != 72
        or len({s["context"] for s in samples}) < 2
    ):
        raise ValueError("all 72 initial cases and at least two decision contexts required")
    data = dict(
        features=np.stack([s["features"] for s in samples]).astype(np.float32),
        context=np.array([s["context"] for s in samples], np.int64),
        side=np.array([s["side"] for s in samples], np.float64),
        success=np.array(
            [[v["return_0p1m_and_healthy"] for v in s["metrics"]] for s in samples], bool
        ),
        failure=np.array([[v["body_failure"] for v in s["metrics"]] for s in samples], bool),
        kick=np.array([[v["outgoing_foot_strike"] for v in s["metrics"]] for s in samples], bool),
    )
    if any(
        digest(source_path) != source_hash or digest(audit_path) != audit_hash
        for source_path, source_hash, audit_path, audit_hash in frozen
    ):
        raise ValueError("source changed during decision export")
    args.output.mkdir(parents=True)
    output = args.output / "dataset.npz"
    np.savez_compressed(output, **data)
    clean(args.expected_head)
    write_once(
        args.output / "complete.json",
        dict(
            status="PAIRED_RELAY_BALANCE_DECISIONS_EXPORTED_NOT_TRAINED",
            source_head=args.expected_head,
            source_pins=source_pins,
            teacher_pairs=teacher_identity,
            dataset_hash=digest(output),
            conditions=len(conditions),
            handoff_decisions=len(samples),
            no_handoff_conditions=len(conditions) - len(samples),
            groups=len({s["context"] for s in samples}),
            outcome_columns=["baseline", "teacher0", "teacher1", "teacher2"],
            feature_semantics=(
                "17 current body-relative values and actual return goal "
                "before teacher action; no future outcomes"
            ),
            initial_and_neural_prefix_verified=True,
            rows=rows,
            independently_reintegrated_during_export=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
