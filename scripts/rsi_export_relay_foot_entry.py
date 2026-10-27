"""Extract causal control-entry foot/ball states from audited relay worlds."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiving_foot_observation import (
    first_contact_control_entry,
    measure_foot_ball_kinematics,
)
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("ROOT", "AUDIT"))
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--dataset-hash", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.pair) != 6
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.dataset.is_symlink()
        or digest(a.dataset) != a.dataset_hash
    ):
        raise ValueError("six audited blocks, pinned data and unused external output required")
    dataset = json.loads(a.dataset.read_text())
    if dataset.get("status") != "PAIRED_RELAY_BALANCE_DECISIONS_EXPORTED_NOT_TRAINED":
        raise ValueError("paired decision data required")
    pins = {(x["block"], x["expanded"]): x for x in dataset["source_pins"]}
    conditions = {(r["context"], r["side"]): r for r in dataset["rows"]}
    if len(pins) != 6 or len(conditions) != 72:
        raise ValueError("exact full paired dataset required")
    seen, rows, features, frozen = set(), [], [], []
    for root_name, audit_name in a.pair:
        root, audit = Path(root_name), Path(audit_name)
        declaration_path = root / "declaration.json"
        declaration = json.loads(declaration_path.read_text())
        expanded = declaration["protocol"] == "soccer.neural_relay_balance_expanded.development.v6"
        if (
            not expanded
            and declaration["protocol"] != "soccer.neural_relay_balance_teacher.development.v5"
        ):
            raise ValueError("exact teacher study protocols required")
        block = declaration["development_context_block"]
        key = (block, expanded)
        if key in seen or key not in pins:
            raise ValueError("unique complete study blocks required")
        seen.add(key)
        status = (
            "NEURAL_RELAY_BALANCE_EXPANDED64_COLLECTED_PENDING_OWN_REPLAY"
            if expanded
            else "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
        )
        source, _, sh = report(
            root, audit, pins[key]["audit_hash"], status, worlds=64 if expanded else 32
        )
        if sh != pins[key]["source_hash"] or digest(declaration_path) != source["declaration_hash"]:
            raise ValueError("exact pinned producer and declaration required")
        xmls = [
            Path(x)
            for x in source["input_file_hashes"]
            if x.endswith("/cached-standard/declared-scene.xml")
        ]
        if (
            len(xmls) != 1
            or digest(xmls[0]) != source["input_file_hashes"][str(xmls[0])]
            or (root / "scene.xml").read_text()
            != shared_native_scene(xmls[0].read_text(), prefixes=("a/", "b/"))
        ):
            raise ValueError("exact pinned native scene required")
        model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        for row in source["records"]:
            if row["course"]["arm"]["balance_teacher"] != -1:
                continue
            context, side = row["course"]["context"], row["course"]["lateral_m"]
            if (context, side) not in conditions or conditions[(context, side)]["handoff"] != (
                row["handoff"] is not None
            ):
                raise ValueError("dataset and baseline condition mismatch")
            pp, ip = bind(root, row, "physics"), bind(root, row, "initial")
            if pp is None or ip is None:
                raise ValueError("exact initial and full measured physics required")
            with (
                np.load(pp, allow_pickle=False) as data,
                np.load(ip, allow_pickle=False) as initial,
            ):
                event = first_contact_control_entry(
                    data["qpos"],
                    data["qvel"],
                    data["ball_forces"],
                    initial["qpos"],
                    initial["qvel"],
                )
            if (
                digest(pp) != row["artifacts"]["physics"]["hash"]
                or digest(ip) != row["artifacts"]["initial"]["hash"]
            ):
                raise ValueError("physical inputs changed during extraction")
            item = dict(
                context=context,
                side=side,
                block=block,
                expanded=expanded,
                source_trial=row["index"],
                physical_hash=digest(pp),
                initial_hash=digest(ip),
                handoff=bool(row["result"]["handoff_executed"]),
                precise_healthy_return=bool(row["result"]["return_0p1m_and_healthy"]),
                body_failure=row["termination"] == "BODY_FAILURE",
                excluded_contact_bodies=row["result"]["excluded_contact_bodies"],
            )
            if event is None:
                rows.append(dict(**item, receiver_contact=False, feature_index=None))
                continue
            q, v, control, substep, categories = event
            obs = measure_foot_ball_kinematics(
                model,
                q,
                v,
                foot_body_names=("b/left_ankle_roll_link", "b/right_ankle_roll_link"),
                ball_body_name="soccer_ball",
                ball_dofadr=layout.bv,
            )
            features.append(obs)
            rows.append(
                dict(
                    **item,
                    receiver_contact=True,
                    feature_index=len(features) - 1,
                    first_contact_control=control,
                    first_contact_substep=substep,
                    first_contact_force_categories=list(categories),
                )
            )
        frozen.extend(((root / "complete.json", sh), (audit, pins[key]["audit_hash"])))
    if (
        seen != set(pins)
        or len(rows) != 72
        or len({(r["context"], r["side"]) for r in rows}) != 72
        or digest(a.dataset) != a.dataset_hash
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("all unique conditions and unchanged physical provenance required")
    a.output.mkdir(parents=True)
    path = a.output / "foot-entry.npz"
    np.savez_compressed(path, foot_relative=np.stack(features))
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="RELAY_FOOT_ENTRY_DATA_EXPORTED_NOT_TRAINED",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            feature_hash=digest(path),
            conditions=len(rows),
            contact_conditions=len(features),
            records=rows,
            feature_semantics=(
                "at control-entry before retrospectively selected first B contact; "
                "left/right foot reference-point position and velocity minus ball center"
            ),
            contact_point_or_ball_spin_proven=False,
            deployable_event_predictor=False,
            future_outcomes_used_as_input=False,
            independently_reintegrated_during_export=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
