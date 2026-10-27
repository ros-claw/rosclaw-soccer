"""Independently audit first-contact entry features with finite differences."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def finite_difference(model, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    def points(position: np.ndarray) -> np.ndarray:
        d = mujoco.MjData(model)
        d.qpos[:], d.qvel[:] = position, v
        mujoco.mj_forward(model, d)
        ball = d.xpos[model.body("soccer_ball").id].copy()
        values = []
        for name, lateral in (("b/left_ankle_roll_link", -0.02), ("b/right_ankle_roll_link", 0.02)):
            i = model.body(name).id
            point = d.xpos[i] + d.xmat[i].reshape(3, 3) @ np.array([0.13, lateral, -0.025])
            values.append(point - ball)
        return np.stack(values)

    delta = 1e-6
    after = q.copy()
    mujoco.mj_integratePos(model, after, v, delta)
    before, following = points(q), points(after)
    return np.concatenate((before, (following - before) / delta), axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("ROOT", "AUDIT"))
    for name in ("dataset", "feature-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("dataset-hash", "feature-hash", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    fp, npz = a.feature_root / "complete.json", a.feature_root / "foot-entry.npz"
    if (
        len(a.pair) != 6
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(x.is_symlink() for x in (fp, npz, a.dataset))
        or digest(fp) != a.feature_hash
        or digest(a.dataset) != a.dataset_hash
    ):
        raise ValueError("exact exported features, data and unused output required")
    receipt, dataset = json.loads(fp.read_text()), json.loads(a.dataset.read_text())
    if (
        receipt.get("status") != "RELAY_FOOT_ENTRY_DATA_EXPORTED_NOT_TRAINED"
        or receipt.get("source_head") != a.producer_head
        or receipt.get("dataset_hash") != a.dataset_hash
        or digest(npz) != receipt["feature_hash"]
        or receipt.get("conditions") != 72
        or receipt.get("future_outcomes_used_as_input") is not False
    ):
        raise ValueError("complete unpromoted causal entry export required")
    with np.load(npz, allow_pickle=False) as f:
        features = f["foot_relative"].copy()
    if features.shape != (receipt["contact_conditions"], 2, 6) or not np.isfinite(features).all():
        raise ValueError("finite ordered foot-point feature tensor required")
    pins = {(x["block"], x["expanded"]): x for x in dataset["source_pins"]}
    recorded = {(r["context"], r["side"]): r for r in receipt["records"]}
    if len(recorded) != 72 or len(pins) != 6:
        raise ValueError("unique complete feature and source records required")
    used, blocks, max_error, frozen = set(), set(), 0.0, []
    for root_name, audit_name in a.pair:
        root, audit = Path(root_name), Path(audit_name)
        decpath = root / "declaration.json"
        dec = json.loads(decpath.read_text())
        expanded = dec["protocol"] == "soccer.neural_relay_balance_expanded.development.v6"
        if not expanded and dec["protocol"] != "soccer.neural_relay_balance_teacher.development.v5":
            raise ValueError("exact archived protocol required")
        key = (dec["development_context_block"], expanded)
        if key in blocks or key not in pins:
            raise ValueError("unique original data blocks required")
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
            raise ValueError("original independently replayed source required")
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
            raise ValueError("exact original shared scene required")
        model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
        for trial in source["records"]:
            if trial["course"]["arm"]["balance_teacher"] != -1:
                continue
            ck = (trial["course"]["context"], trial["course"]["lateral_m"])
            row = recorded[ck]
            if (
                row["source_trial"] != trial["index"]
                or row["block"] != key[0]
                or row["expanded"] != expanded
                or row["handoff"] != (trial["handoff"] is not None)
                or row["body_failure"] != (trial["termination"] == "BODY_FAILURE")
                or row["precise_healthy_return"] != trial["result"]["return_0p1m_and_healthy"]
            ):
                raise ValueError("source identity or future label mismatch")
            pp, ip = bind(root, trial, "physics"), bind(root, trial, "initial")
            if (
                pp is None
                or ip is None
                or digest(pp) != row["physical_hash"]
                or digest(ip) != row["initial_hash"]
            ):
                raise ValueError("exact source trajectory and initial state required")
            with (
                np.load(pp, allow_pickle=False) as data,
                np.load(ip, allow_pickle=False) as initial,
            ):
                force = data["ball_forces"][:, :, 1, :]
                event = np.argwhere((force > 1).any(axis=2))
                if not len(event):
                    if row["receiver_contact"] or row["feature_index"] is not None:
                        raise ValueError("no-contact condition mislabeled")
                    continue
                control, substep = map(int, event[0])
                categories = np.flatnonzero(force[control, substep] > 1).tolist()
                if (
                    not row["receiver_contact"]
                    or row["first_contact_control"] != control
                    or row["first_contact_substep"] != substep
                    or row["first_contact_force_categories"] != categories
                ):
                    raise ValueError("actual first native contact mismatch")
                q = initial["qpos"].copy() if control == 0 else data["qpos"][control - 1, -1].copy()
                v = initial["qvel"].copy() if control == 0 else data["qvel"][control - 1, -1].copy()
            index = row["feature_index"]
            if type(index) is not int or index in used or not 0 <= index < len(features):
                raise ValueError("unique in-range feature index required")
            used.add(index)
            reconstructed = finite_difference(model, q, v)
            error = float(np.abs(reconstructed - features[index]).max())
            max_error = max(max_error, error)
            if error > 1e-4:
                raise ValueError("finite-difference foot/ball kinematics mismatch")
        frozen.extend(((root / "complete.json", sh), (audit, pins[key]["audit_hash"])))
    if (
        blocks != set(pins)
        or used != set(range(len(features)))
        or digest(fp) != a.feature_hash
        or digest(npz) != receipt["feature_hash"]
        or digest(a.dataset) != a.dataset_hash
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("complete unchanged source and feature coverage required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RELAY_FOOT_ENTRY_FINITE_DIFFERENCE_AUDITED_NOT_CONTROL",
            source_head=a.expected_head,
            feature_hash=a.feature_hash,
            dataset_hash=a.dataset_hash,
            conditions=72,
            contact_conditions=len(features),
            max_abs_error=max_error,
            independent_finite_difference=True,
            independent_physics_reintegration=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
