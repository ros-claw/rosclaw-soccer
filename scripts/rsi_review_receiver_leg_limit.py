"""Review actual projected-leg relay outcomes after independent own replay."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.paired_precontact_authority import precontact_tracking_error
from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "prior", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "prior-hash", "prior-audit-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    root, audit = a.source / "source", a.source / "audit/complete.json"
    old, old_audit = a.prior / "source", a.prior / "audit/complete.json"
    source, _, sh = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_LEG_TARGET_PROJECTION16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    prior, _, ph = report(
        old,
        old_audit,
        a.prior_audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    if sh != a.source_hash or ph != a.prior_hash:
        raise ValueError("exact pinned source and previous independent replay required")
    declaration = json.loads((root / "declaration.json").read_text())
    if (
        declaration.get("receiver_leg_projection_is_unlearned") is not True
        or declaration.get("receiver_actor_previous_action_remains_raw") is not True
        or declaration.get("development_context_block") != 0
    ):
        raise ValueError("exact explicit projection and action-memory contract required")
    model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = {
        (
            r["course"]["context"],
            r["course"]["lateral_m"],
            r["course"]["arm"]["receiver_leg_projection"],
        ): r
        for r in source["records"]
    }
    baseline = {
        (r["course"]["context"], r["course"]["lateral_m"]): r
        for r in prior["records"]
        if r["course"]["arm"]["receiver_lateral_offset_m"] == 0.12
    }
    if len(rows) != 16 or len(baseline) != 8:
        raise ValueError("complete unique paired conditions required")
    records = []
    for key in sorted(baseline):
        parent, projected = rows[(*key, False)], rows[(*key, True)]
        if not artifact_equal(root, parent, old, baseline[key], "initial") or not artifact_equal(
            root, parent, root, projected, "initial"
        ):
            raise ValueError("paired worlds must share their actual initial state")
        identity = artifact_equal(root, parent, old, baseline[key], "physics")
        values = {}
        for name, r in (("baseline", parent), ("projected", projected)):
            physical, velocity = bind(root, r, "physics"), bind(root, r, "receiver_velocity")
            if physical is None or velocity is None:
                raise ValueError("actual trajectory and projection trace required")
            with (
                np.load(physical, allow_pickle=False) as raw,
                np.load(velocity, allow_pickle=False) as v,
            ):
                d = v["LIMIT_delta"]
                if (
                    d.shape != (len(v["control"]), 29)
                    or not np.isfinite(d).all()
                    or np.any(d[:, 12:] != 0)
                ):
                    raise ValueError("finite actual 12-leg-axis projection history required")
                changed = np.any(d != 0, axis=1)
                if name == "baseline" and changed.any():
                    raise ValueError("disabled projection must exactly preserve parent")
                values[name] = dict(
                    outcome=strict_pass_receive(
                        raw["ball_forces"],
                        raw["body_failed"],
                        handoff=r["result"]["handoff_executed"],
                    ),
                    precontact_tracking=precontact_tracking_error(
                        raw["target_b"], raw["qpos"], raw["ball_forces"], joint_indices=layout.qi
                    ),
                    changed_controls=int(changed.sum()),
                    max_projection_rad=float(np.abs(d).max()),
                    precise_healthy_return=r["result"]["return_0p1m_and_healthy"],
                )
        records.append(
            dict(context=key[0], side=key[1], zero_projection_matches_prior=identity, **values)
        )
    summary = {
        name: dict(
            strict_clean=sum(r[name]["outcome"]["strict_both_clean_handoff"] for r in records),
            healthy=sum(r[name]["outcome"]["both_bodies_healthy"] for r in records),
            foot_after_pass=sum(
                r[name]["outcome"]["passer_foot_precedes_receiver"] for r in records
            ),
            nonfoot=sum(
                r[name]["outcome"]["receiver_nonfoot_substep"] is not None for r in records
            ),
            precise=sum(r[name]["precise_healthy_return"] for r in records),
            changed_worlds=sum(r[name]["changed_controls"] > 0 for r in records),
        )
        for name in ("baseline", "projected")
    }
    if any(
        digest(p) != h
        for p, h in (
            (root / "complete.json", sh),
            (audit, a.audit_hash),
            (old / "complete.json", ph),
            (old_audit, a.prior_audit_hash),
        )
    ):
        raise ValueError("source and replay evidence changed during review")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_LEG_PROJECTION_REPLAY_REVIEWED",
            source_head=a.expected_head,
            source_hash=sh,
            audit_hash=a.audit_hash,
            prior_hash=ph,
            zero_identity_count=sum(r["zero_projection_matches_prior"] for r in records),
            summary=summary,
            records=records,
            projection_unlearned=True,
            raw_action_memory_preserved=True,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
