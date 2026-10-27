"""Diagnose every no-handoff baseline from previously audited relay physics."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiving_handoff_diagnosis import diagnose_no_handoff
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", nargs=2, action="append", required=True, metavar=("ROOT", "AUDIT")
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        len(args.pair) != 6
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or args.dataset.is_symlink()
        or digest(args.dataset) != args.dataset_hash
    ):
        raise ValueError("six audited physical blocks, pinned data and unused output required")
    dataset = json.loads(args.dataset.read_text())
    if dataset.get("status") != "PAIRED_RELAY_BALANCE_DECISIONS_EXPORTED_NOT_TRAINED":
        raise ValueError("paired source dataset receipt required")
    pins = {(p["block"], p["expanded"]): p for p in dataset["source_pins"]}
    expected_rows = {(r["context"], r["side"]): r for r in dataset["rows"]}
    if len(pins) != 6 or len(expected_rows) != 72:
        raise ValueError("complete six-block unique paired dataset required")
    seen, rows, frozen = set(), [], []
    for root_name, audit_name in args.pair:
        root, audit_path = Path(root_name), Path(audit_name)
        declaration_path = root / "declaration.json"
        declaration = json.loads(declaration_path.read_text())
        protocol = declaration["protocol"]
        expanded = protocol == "soccer.neural_relay_balance_expanded.development.v6"
        if not expanded and protocol != "soccer.neural_relay_balance_teacher.development.v5":
            raise ValueError("unexpected source protocol")
        key = (declaration["development_context_block"], expanded)
        if key in seen or key not in pins:
            raise ValueError("duplicate or undeclared source block")
        seen.add(key)
        pin = pins[key]
        status = (
            "NEURAL_RELAY_BALANCE_EXPANDED64_COLLECTED_PENDING_OWN_REPLAY"
            if expanded
            else "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
        )
        source, _, source_hash = report(
            root, audit_path, pin["audit_hash"], status, worlds=64 if expanded else 32
        )
        if (
            source_hash != pin["source_hash"]
            or digest(declaration_path) != source["declaration_hash"]
        ):
            raise ValueError("source or declaration differs from audited dataset")
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
            raise ValueError("exact original physics scene required")
        model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        for row in source["records"]:
            if row["course"]["arm"]["balance_teacher"] != -1 or row["handoff"] is not None:
                continue
            condition = (row["course"]["context"], row["course"]["lateral_m"])
            if (
                condition not in expected_rows
                or expected_rows[condition]["handoff"]
                or row["termination"] != "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300"
            ):
                raise ValueError("exact original non-handoff condition and horizon required")
            path = bind(root, row, "physics")
            if path is None:
                raise ValueError("physical trajectory required")
            with np.load(path, allow_pickle=False) as f:
                item = diagnose_no_handoff(
                    f["qpos"],
                    f["qvel"],
                    f["ball_forces"],
                    f["body_failed"],
                    receiver_q=layout.rq,
                    receiver_v=layout.rv,
                    ball_q=layout.bq,
                    ball_v=layout.bv,
                    excluded_contact=bool(row["result"]["excluded_contact_bodies"]),
                )
            if not item["no_handoff_reconstruction_consistent"]:
                raise ValueError("original no-handoff predicate disagrees with measured archive")
            rows.append(
                dict(
                    context=condition[0],
                    side=condition[1],
                    block=key[0],
                    expanded=expanded,
                    source_trial=row["index"],
                    physics_hash=digest(path),
                    **item,
                )
            )
        frozen.append((root / "complete.json", source_hash, audit_path, pin["audit_hash"]))
    if (
        seen != set(pins)
        or len(rows) != dataset["no_handoff_conditions"]
        or len({(r["context"], r["side"]) for r in rows}) != len(rows)
    ):
        raise ValueError("all and only no-handoff physical conditions required")
    if digest(args.dataset) != args.dataset_hash or any(
        digest(s) != sh or digest(a) != ah for s, sh, a, ah in frozen
    ):
        raise ValueError("source changed during diagnosis")
    summary = dict(
        conditions=len(rows),
        passer_foot_over_10n=sum(r["actual_passer_foot_over_10n"] for r in rows),
        receiver_contact_over_1n=sum(r["actual_receiver_contact_over_1n"] for r in rows),
        receiver_contact_before_pass=sum(r["receiver_contact_precedes_passer_kick"] for r in rows),
        excluded_contact=sum(r["excluded_contact_ever"] for r in rows),
        within_0_65m=sum(r["minimum_root_ball_xy_m"] <= 0.65 for r in rows),
        post_touch_blocked={
            k: sum(r["post_touch_controls_failing_each_predicate"][k] > 0 for r in rows)
            for k in rows[0]["post_touch_controls_failing_each_predicate"]
        },
    )
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="RELAY_NO_HANDOFF_PHYSICAL_DIAGNOSIS_NOT_INTERVENTION",
            source_head=args.expected_head,
            dataset_hash=args.dataset_hash,
            summary=summary,
            rows=rows,
            mutually_exclusive_causes=False,
            own_independent_contact_reintegration=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
