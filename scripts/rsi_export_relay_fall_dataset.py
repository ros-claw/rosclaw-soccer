"""Export pre-action fall-prediction samples from fully replayed relay worlds."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.relay_fall_samples import RelayLayout, relay_fall_samples
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


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
        len(args.pair) != 3
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("exact three audited blocks and unused external output required")
    parts, records, provenance, blocks = [], [], [], set()
    for source_name, audit_name, audit_hash in args.pair:
        source, audit_path = Path(source_name), Path(audit_name)
        source_file, decl_file = source / "complete.json", source / "declaration.json"
        if (
            any(p.is_symlink() for p in (source_file, decl_file, audit_path))
            or digest(audit_path) != audit_hash
        ):
            raise ValueError("pinned complete replay required")
        source_hash = digest(source_file)
        report, audit, decl = (
            json.loads(p.read_text()) for p in (source_file, audit_path, decl_file)
        )
        block = decl.get("development_context_block")
        if (
            type(block) is not int
            or block not in (0, 1, 2)
            or block in blocks
            or decl.get("protocol") != "soccer.neural_relay_development_contexts.v4"
            or report.get("status") != "NEURAL_RELAY_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
            or report.get("declaration_hash") != digest(decl_file)
            or audit.get("status") != "ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH"
            or audit.get("producer_hash") != source_hash
            or audit.get("worlds") != 32
            or report.get("counts") != audit.get("counts")
            or len(report.get("records", [])) != 32
            or len(audit.get("records", [])) != 32
            or any(
                row.get(key) is not False
                for row in (report, audit)
                for key in ("goal_complete", "promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("all unpromoted same-source 32-world own replays required")
        blocks.add(block)
        xmls = [
            Path(p)
            for p in report["input_file_hashes"]
            if p.endswith("/cached-standard/declared-scene.xml")
        ]
        if len(xmls) != 1 or digest(xmls[0]) != report["input_file_hashes"][str(xmls[0])]:
            raise ValueError("pinned source body scene required")
        xml = shared_native_scene(xmls[0].read_text(), prefixes=("a/", "b/"))
        if (source / "scene.xml").read_text() != xml:
            raise ValueError("compiled scene must match replay source")
        model = mujoco.MjModel.from_xml_string(xml)
        native = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        layout = RelayLayout(
            native.rq,
            native.rv,
            native.bq,
            native.bv,
            tuple(map(int, native.qi)),
            tuple(map(int, native.vi)),
        )
        provenance.append(
            dict(
                block=block,
                source_hash=source_hash,
                audit_hash=audit_hash,
                scene_hash=digest(source / "scene.xml"),
            )
        )
        for index, (row, own) in enumerate(zip(report["records"], audit["records"], strict=True)):
            if row["index"] != index or any(
                row[k] != own[k] for k in ("index", "course", "termination", "result")
            ):
                raise ValueError("source and own neural/physics replay differ")
            paths = []
            for name in ("initial", "physics"):
                artifact = row["artifacts"][name]
                path = source / artifact["path"]
                if (
                    path.is_symlink()
                    or path.resolve().parent != (source / f"trial{index}").resolve()
                    or digest(path) != artifact["hash"]
                ):
                    raise ValueError("confined immutable audited native arrays required")
                paths.append(path)
            with (
                np.load(paths[0], allow_pickle=False) as initial,
                np.load(paths[1], allow_pickle=False) as trace,
            ):
                x, y, controls = relay_fall_samples(
                    initial_qpos=initial["qpos"],
                    initial_qvel=initial["qvel"],
                    qpos=trace["qpos"],
                    qvel=trace["qvel"],
                    target_b=trace["target_b"],
                    body_failed=trace["body_failed"],
                    layout=layout,
                )
            course = row["course"]
            if course["lateral_m"] not in (-0.55, 0.55):
                raise ValueError("declared independent initial-condition grouping required")
            group = int(course["context"]) * 2 + int(course["lateral_m"] > 0)
            parts.append(
                dict(
                    features=x,
                    label=y,
                    control=controls,
                    group=np.full(len(y), group, np.int64),
                    episode=np.full(len(y), block * 32 + index, np.int64),
                )
            )
            records.append(
                dict(
                    block=block,
                    index=index,
                    course=course,
                    termination=row["termination"],
                    controls=row["controls"],
                    uncensored_samples=len(y),
                    positives=int(y.sum()),
                    physics_hash=row["artifacts"]["physics"]["hash"],
                )
            )
        if digest(source_file) != source_hash or digest(audit_path) != audit_hash:
            raise ValueError("source changed during export")
    combined = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    if blocks != {0, 1, 2} or len(set(combined["group"].tolist())) != 24:
        raise ValueError("all 24 paired initial contexts required")
    args.output.mkdir(parents=True)
    path = args.output / "dataset.npz"
    np.savez_compressed(path, **combined)
    clean(args.expected_head)
    write_once(
        args.output / "complete.json",
        dict(
            status="RELAY_CAUSAL_FALL_SAMPLES_EXPORTED_NOT_TRAINED",
            source_head=args.expected_head,
            provenance=provenance,
            dataset_hash=digest(path),
            samples=len(combined["label"]),
            positive_samples=int(combined["label"].sum()),
            groups=24,
            episodes=records,
            horizon_controls=25,
            state="pre-action actual prior physical boundary, last actual joint target",
            censored="unknown tail excluded; known B failure remains positive",
            labels_are_not_inputs=True,
            independent_physics_reintegrated=False,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
