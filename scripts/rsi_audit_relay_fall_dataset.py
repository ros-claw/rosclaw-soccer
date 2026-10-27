"""Read back causal relay samples from audited source arrays; not physics replay."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.relay_fall_samples import RelayLayout, relay_fall_samples
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    for name in ("dataset-hash", "expected-head"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    receipt_file, data_file = args.dataset / "complete.json", args.dataset / "dataset.npz"
    if (
        len(args.source) != 3
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or receipt_file.is_symlink()
        or data_file.is_symlink()
        or digest(receipt_file) != args.dataset_hash
    ):
        raise ValueError("exact complete data and unused external audit required")
    receipt = json.loads(receipt_file.read_text())
    if (
        receipt["status"] != "RELAY_CAUSAL_FALL_SAMPLES_EXPORTED_NOT_TRAINED"
        or digest(data_file) != receipt["dataset_hash"]
    ):
        raise ValueError("exact relay dataset required")
    with np.load(data_file, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    if (
        set(data) != {"features", "label", "control", "group", "episode"}
        or data["features"].shape != (receipt["samples"], 104)
        or any(
            data[k].shape != (receipt["samples"],) for k in ("label", "control", "group", "episode")
        )
    ):
        raise ValueError("exact typed sample arrays required")
    offset = 0
    for root, provenance in zip(args.source, receipt["provenance"], strict=True):
        report_file = root / "complete.json"
        if (
            digest(report_file) != provenance["source_hash"]
            or digest(root / "scene.xml") != provenance["scene_hash"]
        ):
            raise ValueError("source report and model changed")
        report = json.loads(report_file.read_text())
        model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
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
        for row in report["records"]:
            paths = []
            for key in ("initial", "physics"):
                artifact = row["artifacts"][key]
                path = root / artifact["path"]
                if (
                    path.is_symlink()
                    or path.resolve().parent != (root / f"trial{row['index']}").resolve()
                    or digest(path) != artifact["hash"]
                ):
                    raise ValueError("unchanged actual trace required")
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
                failure = bool(trace["body_failed"][-1, 1])
                n = len(trace["body_failed"])
            # Independently check the fixed-horizon label and right-censor rule.
            expected_controls = [t for t in range(n) if (failure and n - 1 < t + 25) or t + 25 <= n]
            expected_y = [failure and n - 1 < t + 25 for t in expected_controls]
            if controls.tolist() != expected_controls or y.tolist() != expected_y:
                raise ValueError("causal horizon or censor labels differ")
            end = offset + len(y)
            group = row["course"]["context"] * 2 + int(row["course"]["lateral_m"] > 0)
            episode = provenance["block"] * 32 + row["index"]
            if (
                not np.array_equal(data["features"][offset:end], x)
                or not np.array_equal(data["label"][offset:end], y)
                or not np.array_equal(data["control"][offset:end], controls)
                or not np.all(data["group"][offset:end] == group)
                or not np.all(data["episode"][offset:end] == episode)
            ):
                raise ValueError("exact causal dataset readback differs")
            offset = end
    if (
        offset != receipt["samples"]
        or int(data["label"].sum()) != receipt["positive_samples"]
        or digest(data_file) != receipt["dataset_hash"]
    ):
        raise ValueError("full dataset readback and aggregate required")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="RELAY_FALL_SAMPLES_READBACK_NOT_REINTEGRATED",
            source_head=args.expected_head,
            dataset_hash=args.dataset_hash,
            samples=offset,
            positive_samples=int(data["label"].sum()),
            independent_label_censor_recomputed=True,
            feature_extraction_reuses_export_code=True,
            independent_physics_reintegrated=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
