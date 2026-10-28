"""Measure initial receiver contact geometry in the replayed exploration pilot."""

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("review-hash", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.review.is_symlink()
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("exact reviewed exploration and unused external output required")
    review = json.loads(a.review.read_text())
    if (
        review.get("status") != "LEARNED_RECEIVER_EXPLORATION_PILOT_REVIEWED"
        or review.get("sampled_episodes") != 24
        or any(
            review.get(k) is not False
            for k in ("training_performed", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("own replay and sample-density-reviewed exploration required")
    root, audit = a.pilot / "source", a.pilot / "audit/complete.json"
    source, _, source_hash = report(
        root,
        audit,
        review["sampled_audit_hash"],
        "NEURAL_RELAY_RECEIVER_LEARNED_EXPLORATION32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    if source_hash != review["sampled_source_hash"] or source["source_head"] != a.producer_head:
        raise ValueError("exact own-replayed exploration source required")
    scene = root / "scene.xml"
    if scene.is_symlink():
        raise ValueError("owned source scene required")
    scene_hash = digest(scene)
    model, rows = mujoco.MjModel.from_xml_path(str(scene)), []
    expected = {(r["context"], r["side"], r["seed"]): r for r in review["records"]}
    if len(expected) != 24:
        raise ValueError("24 unique reviewed sampled episodes required")
    for trial in source["records"]:
        arm = trial["course"]["arm"]
        if arm["exploration_seed"] is None:
            continue
        key = (trial["course"]["context"], trial["course"]["lateral_m"], arm["exploration_seed"])
        if key not in expected or expected[key]["source_trial"] != trial["index"]:
            raise ValueError("exact reviewed sample identity required")
        physical, initial = bind(root, trial, "physics"), bind(root, trial, "initial")
        if physical is None or initial is None:
            raise ValueError("actual complete native states required")
        with (
            np.load(physical, allow_pickle=False) as raw,
            np.load(initial, allow_pickle=False) as start,
        ):
            contact = first_contact(model, dict(start), dict(raw))
        rows.append(
            dict(
                context=key[0],
                side=key[1],
                seed=key[2],
                contact=contact,
                original=expected[key]["candidate"],
                strict_outcome=expected[key]["physical_clean_outcome"],
            )
        )
    if (
        len(rows) != 24
        or digest(root / "complete.json") != source_hash
        or digest(audit) != review["sampled_audit_hash"]
        or digest(scene) != scene_hash
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("all unchanged physical histories required")
    body_counts = Counter(
        c["receiver_body"] for r in rows for c in r["contact"]["reconstructed_contacts"]
    )
    category_counts = Counter(
        c["category"] for r in rows for c in r["contact"]["reconstructed_contacts"]
    )
    first_contact_groups = {}
    for group in ("foot_only", "other_only", "mixed"):
        subset = []
        for row in rows:
            contacts = row["contact"]["reconstructed_contacts"]
            has_foot = any(c["category"] in ("left_foot", "right_foot") for c in contacts)
            has_other = any(c["category"] == "other_body" for c in contacts)
            if (
                (has_foot and has_other and group == "mixed")
                or (has_foot and not has_other and group == "foot_only")
                or (has_other and not has_foot and group == "other_only")
            ):
                subset.append(row)
        first_contact_groups[group] = dict(
            episodes=len(subset),
            subsequent_or_same_step_nonfoot=sum(
                r["strict_outcome"]["receiver_nonfoot_substep"] is not None for r in subset
            ),
            either_body_failure=sum(not r["strict_outcome"]["both_bodies_healthy"] for r in subset),
            median_first_reconstructed_contact_point_z_m=float(
                np.median(
                    [
                        r["contact"]["reconstructed_contacts"][0]["contact_point_world_m"][2]
                        for r in subset
                    ]
                )
            )
            if subset
            else None,
            median_first_reconstructed_contact_relative_speed_mps=float(
                np.median(
                    [
                        r["contact"]["reconstructed_contacts"][0]["relative_speed_mps"]
                        for r in subset
                    ]
                )
            )
            if subset
            else None,
        )
    summary = dict(
        episodes=len(rows),
        observed_first_contact=sum(r["contact"]["observed_receiver_contact"] for r in rows),
        fully_reconstructed=sum(
            r["contact"]["observed_receiver_contact"]
            and not r["contact"]["unmatched_recorded_categories"]
            for r in rows
        ),
        body_contacts=dict(body_counts),
        category_contacts=dict(category_counts),
        first_contact_groups=first_contact_groups,
    )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="LEARNED_RECEIVER_PILOT_FIRST_CONTACT_DIAGNOSED",
            source_head=a.expected_head,
            source_hash=source_hash,
            review_hash=a.review_hash,
            scene_hash=scene_hash,
            summary=summary,
            records=rows,
            forces_not_assigned_to_individual_geom=True,
            fresh_holdout=False,
            causal_effect_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
