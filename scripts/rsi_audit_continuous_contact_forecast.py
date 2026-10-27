"""Independently read back all causal foot forecasts and censored labels."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_audit_relay_foot_entry import finite_difference
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def independent_first_label(window: np.ndarray) -> int:
    if window.shape != (5, 10, 3) or not np.isfinite(window).all():
        raise ValueError("exact 100ms observed receiver forces required")
    for control in window:
        for sample in control:
            if (sample > 1).any():
                return 2 if sample[2] > 1 else 1
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("ROOT", "AUDIT"))
    for name in ("dataset", "features", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("dataset-hash", "features-hash", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    fp, dp = a.features / "complete.json", a.features / "continuous.npz"
    if (
        len(a.pair) != 6
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() for p in (a.dataset, fp, dp))
        or digest(a.dataset) != a.dataset_hash
        or digest(fp) != a.features_hash
    ):
        raise ValueError("exact six sources, exported features and unused output required")
    receipt, dataset = json.loads(fp.read_text()), json.loads(a.dataset.read_text())
    if (
        receipt.get("status") != "RELAY_CONTINUOUS_CONTACT_DATA_NOT_TRAINED"
        or receipt.get("source_head") != a.producer_head
        or receipt.get("dataset_hash") != a.dataset_hash
        or receipt.get("horizon_controls") != 5
        or receipt.get("censored_tail_controls") != 4
        or receipt.get("future_labels_are_features") is not False
        or digest(dp) != receipt["data_hash"]
    ):
        raise ValueError("causal censored continuous forecast data required")
    with np.load(dp, allow_pickle=False) as f:
        x, y, g, ids, controls = (
            f[k].copy() for k in ("features", "labels", "context", "condition", "control")
        )
    n = receipt["samples"]
    if (
        x.shape != (n, 12)
        or y.shape != (n,)
        or g.shape != (n,)
        or ids.shape != (n,)
        or controls.shape != (n,)
        or not np.isfinite(x).all()
        or not np.isin(y, (0, 1, 2)).all()
        or np.bincount(y, minlength=3).tolist() != receipt["class_counts"]
    ):
        raise ValueError("exact finite complete continuous arrays required")
    recorded = {(r["context"], r["side"]): (i, r) for i, r in enumerate(receipt["records"])}
    pins = {(r["block"], r["expanded"]): r for r in dataset["source_pins"]}
    if len(recorded) != 72 or len(pins) != 6:
        raise ValueError("all unique original conditions and blocks required")
    seen, blocks, frozen, used = set(), set(), [], set()
    max_error = 0.0
    for root_name, audit_name in a.pair:
        root, audit = Path(root_name), Path(audit_name)
        decpath = root / "declaration.json"
        dec = json.loads(decpath.read_text())
        expanded = dec["protocol"] == "soccer.neural_relay_balance_expanded.development.v6"
        if not expanded and dec["protocol"] != "soccer.neural_relay_balance_teacher.development.v5":
            raise ValueError("unexpected source protocol")
        key = (dec["development_context_block"], expanded)
        if key in blocks or key not in pins:
            raise ValueError("unique source block required")
        blocks.add(key)
        status = (
            "NEURAL_RELAY_BALANCE_EXPANDED64_COLLECTED_PENDING_OWN_REPLAY"
            if expanded
            else "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
        )
        source, _, sh = report(
            root, audit, pins[key]["audit_hash"], status, worlds=64 if expanded else 32
        )
        if sh != pins[key]["source_hash"] or digest(decpath) != source["declaration_hash"]:
            raise ValueError("exact independently replayed original source required")
        xmls = [
            Path(v)
            for v in source["input_file_hashes"]
            if v.endswith("/cached-standard/declared-scene.xml")
        ]
        if (
            len(xmls) != 1
            or digest(xmls[0]) != source["input_file_hashes"][str(xmls[0])]
            or (root / "scene.xml").read_text()
            != shared_native_scene(xmls[0].read_text(), prefixes=("a/", "b/"))
        ):
            raise ValueError("exact original shared scene required")
        model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
        for trial in source["records"]:
            if trial["course"]["arm"]["balance_teacher"] != -1:
                continue
            condition = (trial["course"]["context"], trial["course"]["lateral_m"])
            if condition in seen or condition not in recorded:
                raise ValueError("unique original condition required")
            seen.add(condition)
            ix, row = recorded[condition]
            if (
                row["block"] != key[0]
                or row["expanded"] != expanded
                or row["source_trial"] != trial["index"]
            ):
                raise ValueError("exact record origin required")
            pp, ip = bind(root, trial, "physics"), bind(root, trial, "initial")
            if (
                pp is None
                or ip is None
                or digest(pp) != row["physical_hash"]
                or digest(ip) != row["initial_hash"]
            ):
                raise ValueError("exact physical input files required")
            with (
                np.load(pp, allow_pickle=False) as data,
                np.load(ip, allow_pickle=False) as initial,
            ):
                stop = min(
                    300, len(data["qpos"]), trial["handoff"]["control"] if trial["handoff"] else 300
                )
                start, count = row["first_sample"], row["sample_count"]
                if (
                    stop != row["stop_control"]
                    or count != stop - 4
                    or start < 0
                    or start + count > n
                    or not np.array_equal(ids[start : start + count], np.full(count, ix))
                    or not np.array_equal(controls[start : start + count], np.arange(count))
                    or not np.array_equal(g[start : start + count], np.full(count, condition[0]))
                ):
                    raise ValueError("exact pre-handoff control bounds and split metadata required")
                for t in range(count):
                    index = start + t
                    if index in used:
                        raise ValueError("sample coverage overlap")
                    used.add(index)
                    label = independent_first_label(data["ball_forces"][t : t + 5, :, 1, :])
                    q = initial["qpos"] if t == 0 else data["qpos"][t - 1, -1]
                    v = initial["qvel"] if t == 0 else data["qvel"][t - 1, -1]
                    value = finite_difference(model, q, v).reshape(12)
                    error = float(np.max(np.abs(value - x[index])))
                    max_error = max(max_error, error)
                    if int(y[index]) != label or error > 2e-4:
                        raise ValueError("future label or causal feature differs")
        frozen.extend(((root / "complete.json", sh), (audit, pins[key]["audit_hash"])))
    if (
        blocks != set(pins)
        or seen != set(recorded)
        or used != set(range(n))
        or digest(fp) != a.features_hash
        or digest(dp) != receipt["data_hash"]
        or digest(a.dataset) != a.dataset_hash
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("complete unchanged causal data provenance required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RELAY_CONTINUOUS_CONTACT_FINITE_DIFFERENCE_AUDIT_NOT_CONTROL",
            source_head=a.expected_head,
            features_hash=a.features_hash,
            dataset_hash=a.dataset_hash,
            conditions=72,
            samples=n,
            class_counts=receipt["class_counts"],
            max_abs_error=max_error,
            all_labels_independently_recomputed=True,
            all_features_independently_finite_differenced=True,
            independent_physics_reintegration=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
