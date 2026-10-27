"""Reconstruct pre-step first receiver contact geometry from audited SIM histories."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.contact_point_kinematics import contact_point_relative_velocity
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report


def first_contact(model, initial: dict, physical: dict) -> dict:
    """Contact forces were recorded after mj_step, geometry used its input state."""
    forces, qpos, qvel = physical["ball_forces"], physical["qpos"], physical["qvel"]
    n = len(qpos)
    if (
        qpos.shape != (n, 10, model.nq)
        or qvel.shape != (n, 10, model.nv)
        or forces.shape != (n, 10, 2, 3)
        or n < 1
        or any(not np.isfinite(x).all() for x in (qpos, qvel, forces))
        or np.any(forces < 0)
    ):
        raise ValueError("complete finite shared-world substep history required")
    events = np.argwhere(np.any(forces[:, :, 1, :] > 1.0, axis=2))
    if not len(events):
        return dict(observed_receiver_contact=False, reconstructed_contacts=[])
    control, substep = map(int, events[0])
    if substep:
        q, v = qpos[control, substep - 1], qvel[control, substep - 1]
    elif control:
        q, v = qpos[control - 1, -1], qvel[control - 1, -1]
    else:
        q, v = initial["qpos"], initial["qvel"]
    if q.shape != (model.nq,) or v.shape != (model.nv,):
        raise ValueError("exact preceding native state required")
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = q, v
    mujoco.mj_forward(model, data)
    ball = int(model.body("soccer_ball").id)
    pelvis = int(model.body("b/pelvis").id)
    feet = [int(model.body("b/" + side + "_ankle_roll_link").id) for side in ("left", "right")]
    recorded = forces[control, substep, 1]
    contacts = []
    matched = set()
    for contact in data.contact:
        body_a, body_b = [int(model.geom_bodyid[g]) for g in contact.geom]
        other = body_b if body_a == ball else body_a if body_b == ball else -1
        if other <= 0:
            continue
        ancestor = other
        while ancestor not in (0, pelvis):
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != pelvis:
            continue
        part = feet.index(other) if other in feet else 2
        if recorded[part] <= 1.0:
            continue
        matched.add(part)
        point = np.asarray(contact.pos).copy()
        normal = np.asarray(contact.frame[:3]).copy()
        contacts.append(
            dict(
                receiver_body=model.body(other).name,
                category=("left_foot", "right_foot", "other_body")[part],
                contact_point_world_m=point.tolist(),
                gap_m=float(contact.dist),
                **contact_point_relative_velocity(model, data, point, normal, ball, other),
            )
        )
    expected = set(np.flatnonzero(recorded > 1.0).tolist())
    return dict(
        observed_receiver_contact=True,
        first_contact_control=control,
        first_contact_substep=substep,
        recorded_category_max_force_n=recorded.tolist(),
        category_matches=sorted(matched),
        unmatched_recorded_categories=sorted(expected - matched),
        reconstructed_contacts=contacts,
        recorded_force_not_assigned_to_individual_geom=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SOURCE", "AUDIT"))
    p.add_argument("--review", type=Path, required=True)
    for name in ("review-hash", "source-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.pair) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.review.is_symlink()
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("all independently reviewed sources and unused external output required")
    review = json.loads(a.review.read_text())
    if (
        review.get("status") != "RECEIVER_LEARNED_PAIR_PHYSICAL_REPLAY_REVIEWED"
        or review.get("producer_head") != a.source_head
        or review.get("conditions") != 24
        or any(
            review.get(k) is not False
            for k in ("goal_complete", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("exact full physical comparison required")
    expected = {(r["context"], r["side"]): r for r in review["records"]}
    if len(expected) != 24:
        raise ValueError("24 unique reviewed conditions required")
    rows, seen, frozen = [], set(), []
    for root_name, audit_name in a.pair:
        root, audit = Path(root_name), Path(audit_name)
        ah = digest(audit)
        source, _, sh = report(
            root, audit, ah, "NEURAL_RELAY_RECEIVER_LEARNED_PAIR16_PENDING_OWN_REPLAY", worlds=16
        )
        declaration = json.loads((root / "declaration.json").read_text())
        block = declaration["development_context_block"]
        if (
            block in seen
            or block not in (0, 1, 2)
            or source["source_head"] != a.source_head
            or digest(root / "declaration.json") != source["declaration_hash"]
        ):
            raise ValueError("distinct exact source blocks required")
        seen.add(block)
        scene = root / "scene.xml"
        if scene.is_symlink():
            raise ValueError("owned audited scene required")
        scene_hash = digest(scene)
        model = mujoco.MjModel.from_xml_path(str(scene))
        for trial in source["records"]:
            condition = (trial["course"]["context"], trial["course"]["lateral_m"])
            arm = "candidate" if trial["course"]["arm"]["receiver_learned"] else "baseline"
            row = expected[condition]
            if (
                row["block"] != block
                or row[arm] != measured(root, trial)
                or row["historical_baseline_identical"] is not True
            ):
                raise ValueError("exact reviewed physical outcome required")
            physics, initial = bind(root, trial, "physics"), bind(root, trial, "initial")
            if physics is None or initial is None:
                raise ValueError("actual physics and initial state required")
            with (
                np.load(physics, allow_pickle=False) as raw,
                np.load(initial, allow_pickle=False) as start,
            ):
                effect = first_contact(model, dict(start), dict(raw))
            rows.append(
                dict(
                    context=condition[0],
                    side=condition[1],
                    block=block,
                    arm=arm,
                    termination=trial["termination"],
                    **effect,
                )
            )
        frozen.extend(((root / "complete.json", sh), (audit, ah), (scene, scene_hash)))
    if (
        seen != {0, 1, 2}
        or len(rows) != 48
        or len({(r["context"], r["side"], r["arm"]) for r in rows}) != 48
        or digest(a.review) != a.review_hash
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("all unchanged paired source histories required")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_FIRST_CONTACT_GEOMETRY_READBACK",
            source_head=a.expected_head,
            review_hash=a.review_hash,
            records=rows,
            prior_state_reconstructed=True,
            measured_force_per_geom=False,
            independent_new_physics=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
