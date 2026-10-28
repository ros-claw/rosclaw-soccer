"""Measure foot and shin clearance before actual sampled receiver contacts."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "diagnosis", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("diagnosis-hash", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.diagnosis.is_symlink()
        or digest(a.diagnosis) != a.diagnosis_hash
    ):
        raise ValueError("exact physical diagnosis and unused external output required")
    diagnosis = json.loads(a.diagnosis.read_text())
    if diagnosis.get("status") != "LEARNED_RECEIVER_PILOT_FIRST_CONTACT_DIAGNOSED" or any(
        diagnosis.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
    ):
        raise ValueError("replayed contact diagnosis required")
    root, audit = a.pilot / "source", a.pilot / "audit/complete.json"
    source, _, sh = report(
        root,
        audit,
        digest(audit),
        "NEURAL_RELAY_RECEIVER_LEARNED_EXPLORATION32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    scene = root / "scene.xml"
    if (
        sh != diagnosis["source_hash"]
        or source["source_head"] != a.producer_head
        or scene.is_symlink()
        or digest(scene) != diagnosis["scene_hash"]
    ):
        raise ValueError("exact original replayed source and scene required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    ball_body = model.body("soccer_ball").id
    ball = np.flatnonzero(model.geom_bodyid == ball_body)
    feet = [
        i
        for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
            ("b/left_foot", "b/right_foot")
        )
        and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("_collision")
    ]
    shins = [model.geom("b/left_shin").id, model.geom("b/right_shin").id]
    if len(ball) != 1 or len(feet) != 14 or len(set(shins)) != 2:
        raise ValueError("exact ball, foot and shin collision geometry required")
    indexed = {
        (
            r["course"]["context"],
            r["course"]["lateral_m"],
            r["course"]["arm"]["exploration_seed"],
        ): r
        for r in source["records"]
    }
    rows = []
    for row in diagnosis["records"]:
        if not row["contact"]["observed_receiver_contact"]:
            continue
        key = (row["context"], row["side"], row["seed"])
        trial = indexed[key]
        path, initial_path = bind(root, trial, "physics"), bind(root, trial, "initial")
        if path is None or initial_path is None:
            raise ValueError("complete original physics and initial state required")
        contact_control = row["contact"]["first_contact_control"]
        states = []
        with (
            np.load(path, allow_pickle=False) as raw,
            np.load(initial_path, allow_pickle=False) as initial,
        ):
            for lead in (10, 5, 1, 0):
                control = contact_control - lead
                if control < 0:
                    continue
                q = initial["qpos"] if control == 0 else raw["qpos"][control - 1, -1]
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                fd = min(
                    float(mujoco.mj_geomDistance(model, data, int(ball[0]), int(g), 4.0, None))
                    for g in feet
                )
                sd = min(
                    float(mujoco.mj_geomDistance(model, data, int(ball[0]), int(g), 4.0, None))
                    for g in shins
                )
                if not np.isfinite((fd, sd)).all():
                    raise ValueError("finite sampled contact clearances required")
                states.append(
                    dict(
                        lead_controls=lead,
                        foot_clearance_m=fd,
                        shin_clearance_m=sd,
                        shin_minus_foot_m=sd - fd,
                    )
                )
        rows.append(
            dict(
                context=key[0],
                side=key[1],
                seed=key[2],
                first_categories=sorted(
                    {c["category"] for c in row["contact"]["reconstructed_contacts"]}
                ),
                states=states,
            )
        )
    summary = {}
    for group, category in (("foot_first", "other_body"), ("shin_first", "other_body")):
        selected = [
            r for r in rows if (category not in r["first_categories"]) == (group == "foot_first")
        ]
        summary[group] = {}
        for lead in (10, 5, 1, 0):
            observations = [s for r in selected for s in r["states"] if s["lead_controls"] == lead]
            summary[group][str(lead)] = dict(
                episodes=len(observations),
                median_foot_clearance_m=float(
                    np.median([s["foot_clearance_m"] for s in observations])
                )
                if observations
                else None,
                median_shin_clearance_m=float(
                    np.median([s["shin_clearance_m"] for s in observations])
                )
                if observations
                else None,
                median_shin_minus_foot_m=float(
                    np.median([s["shin_minus_foot_m"] for s in observations])
                )
                if observations
                else None,
            )
    if (
        len(rows) != 16
        or digest(root / "complete.json") != sh
        or digest(a.diagnosis) != a.diagnosis_hash
        or digest(scene) != diagnosis["scene_hash"]
    ):
        raise ValueError("complete unchanged native contact episodes required")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_PRECONTACT_FOOT_SHIN_CLEARANCE_DIAGNOSED",
            source_head=a.expected_head,
            source_hash=sh,
            diagnosis_hash=a.diagnosis_hash,
            native_timestep_s=float(model.opt.timestep),
            lead_is_control_entry_before_contact=True,
            selected_by_future_contact=True,
            physics_observation_only=True,
            causal_effect_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
            summary=summary,
            records=rows,
        ),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
