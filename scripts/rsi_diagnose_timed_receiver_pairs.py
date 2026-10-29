"""Diagnose first ball contacts and later body failures in replayed SIM pairs."""

import argparse
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_diagnose_receiver_contact_kinematics import first_contact
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def summary(rows: list[dict]) -> dict:
    contacts = [r["contact"] for r in rows if r["contact"]["observed_receiver_contact"]]
    bodies = Counter(c["receiver_body"] for r in contacts for c in r["reconstructed_contacts"])
    categories = Counter(c["category"] for r in contacts for c in r["reconstructed_contacts"])
    speeds = [c["relative_speed_mps"] for r in contacts for c in r["reconstructed_contacts"]]
    failure_rows = [r for r in rows if r["failure_control"] is not None]
    before_contact = [v for r in rows for v in r.get("precontact_max_joint_delta_rad", [])]
    return dict(
        trajectories=len(rows),
        contact_observed=len(contacts),
        contact_unmatched=sum(bool(r["unmatched_recorded_categories"]) for r in contacts),
        reconstructed_bodies=dict(bodies),
        reconstructed_categories=dict(categories),
        median_reconstructed_relative_speed_mps=float(np.median(speeds)) if speeds else None,
        precontact_delta_controls=len(before_contact),
        precontact_median_max_joint_delta_rad=float(np.median(before_contact))
        if before_contact
        else None,
        precontact_max_joint_delta_rad=max(before_contact) if before_contact else None,
        failure_trajectories=len(failure_rows),
        A_failure=sum("A" in r["failed_bodies"] for r in failure_rows),
        B_failure=sum("B" in r["failed_bodies"] for r in failure_rows),
        first_failure_after_handoff=sum(
            r["handoff_control"] is not None and r["failure_control"] >= r["handoff_control"]
            for r in failure_rows
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-hash", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        args.review.is_symlink()
        or digest(args.review) != args.review_hash
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("exact paired review and unused external output required")
    review = json.loads(args.review.read_text())
    if (
        review.get("status") != "TIMED_RECEIVER_PAIR_REPLAYED_DEVELOPMENT_NOT_PROMOTED"
        or len(review.get("provenance", [])) != 9
        or len(review.get("records", [])) != 72
        or any(
            review.get(k) is not False
            for k in ("goal_complete", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("complete replay-verified SIM research pairs required")
    expected = {(r["seed"], r["block"], r["context"], r["side"]): r for r in review["records"]}
    if len(expected) != 72:
        raise ValueError("unique reviewed pairs required")
    rows, pins, groups = [], [(args.review, args.review_hash)], set()
    for proof in review["provenance"]:
        seed, block = proof["seed"], proof["block"]
        if (seed, block) in groups or seed not in (92401, 92402, 92403) or block not in (0, 1, 2):
            raise ValueError("exact nine reviewed groups required")
        groups.add((seed, block))
        root, audit = Path(proof["source_root"]), Path(proof["audit_path"])
        source, _, source_hash = report(
            root,
            audit,
            proof["audit_hash"],
            "NEURAL_RELAY_RECEIVER_FUNCTIONAL_TIMED_PAIR16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        scene, declaration = root / "scene.xml", root / "declaration.json"
        if (
            source_hash != proof["source_hash"]
            or scene.is_symlink()
            or digest(scene) != proof["scene_hash"]
            or declaration.is_symlink()
            or digest(declaration) != proof["declaration_hash"]
        ):
            raise ValueError("exact reviewed physical source and scene required")
        model = mujoco.MjModel.from_xml_path(str(scene))
        pins.extend(
            (
                (root / "complete.json", source_hash),
                (audit, proof["audit_hash"]),
                (scene, proof["scene_hash"]),
                (declaration, proof["declaration_hash"]),
            )
        )
        for trial in source["records"]:
            course = trial["course"]
            key = (seed, block, course["context"], course["lateral_m"])
            if key not in expected or type(course["arm"]["receiver_learned"]) is not bool:
                raise ValueError("matched reviewed physical pair required")
            physics, initial, motor = (
                bind(root, trial, name) for name in ("physics", "initial", "receiver_velocity")
            )
            if physics is None or initial is None or motor is None:
                raise ValueError("native history and initial state required")
            pins.extend(
                (
                    (physics, trial["artifacts"]["physics"]["hash"]),
                    (initial, trial["artifacts"]["initial"]["hash"]),
                    (motor, trial["artifacts"]["receiver_velocity"]["hash"]),
                )
            )
            with (
                np.load(physics, allow_pickle=False) as raw,
                np.load(initial, allow_pickle=False) as start,
            ):
                contact = first_contact(model, dict(start), dict(raw))
                failed = raw["body_failed"]
                if failed.dtype != np.bool_ or failed.shape != (trial["controls"], 2):
                    raise ValueError("complete two-body failure trace required")
                first = np.flatnonzero(failed.any(axis=1))
                failed_bodies = [
                    name
                    for name, value in zip(("A", "B"), failed.any(axis=0), strict=True)
                    if value
                ]
            with np.load(motor, allow_pickle=False) as trace:
                controls, deltas = trace["control"], trace["realized_delta_radians"]
                if (
                    controls.ndim != 1
                    or deltas.shape != (len(controls), 29)
                    or not np.isfinite(deltas).all()
                    or np.any(np.diff(controls) <= 0)
                ):
                    raise ValueError("aligned finite actual receiver motor corrections required")
                contact_control = contact.get("first_contact_control")
                precontact = (
                    np.max(
                        np.abs(
                            deltas[(controls >= contact_control - 5) & (controls < contact_control)]
                        ),
                        axis=1,
                    ).tolist()
                    if contact_control is not None
                    else []
                )
            rows.append(
                dict(
                    seed=seed,
                    block=block,
                    context=key[2],
                    side=key[3],
                    candidate=course["arm"]["receiver_learned"],
                    trial=trial["index"],
                    contact=contact,
                    precontact_max_joint_delta_rad=precontact,
                    failure_control=int(first[0]) if first.size else None,
                    failed_bodies=failed_bodies,
                    handoff_control=trial["handoff"]["control"]
                    if trial["handoff"] is not None
                    else None,
                    termination=trial["termination"],
                )
            )
    if (
        len(rows) != 144
        or len(groups) != 9
        or len({(r["seed"], r["block"], r["context"], r["side"], r["candidate"]) for r in rows})
        != 144
        or any(digest(path) != h for path, h in pins)
    ):
        raise ValueError("all unique stable source histories required")
    summaries = {
        str(seed): {
            arm: summary(
                [r for r in rows if r["seed"] == seed and r["candidate"] == (arm == "candidate")]
            )
            for arm in ("baseline", "candidate")
        }
        for seed in (92401, 92402, 92403)
    }
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="TIMED_RECEIVER_CONTACT_GEOMETRY_DIAGNOSTIC_NOT_PROMOTED",
            source_head=args.expected_head,
            review_hash=args.review_hash,
            summaries=summaries,
            records=rows,
            reconstructed_geometry_not_force_attribution=True,
            later_failure_not_causally_attributed=True,
            reused_development_contexts=True,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
