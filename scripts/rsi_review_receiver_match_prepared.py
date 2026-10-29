"""Compare all replayed pursuit vs velocity-matching shared-world conditions."""

import argparse
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def physical_outcome(root, row, horizon, model, layout):
    path = bind(root, row, "physics")
    if path is None:
        raise ValueError("actual native physics required")
    with np.load(path, allow_pickle=False) as d:
        force = d["ball_forces"][:horizon]
        failed = d["body_failed"][:horizon]
        handoff = row["handoff"]
        witnessed = bool(handoff is not None and handoff["control"] < horizon)
        outcome = strict_pass_receive(force, failed, handoff=witnessed)
        event = outcome["actual_receiver_foot_substep"]
        v = d["qvel"][:horizon].reshape(-1, model.nv)
        relative = (
            None
            if event is None
            else float(
                np.linalg.norm(
                    v[event, layout.bv : layout.bv + 3] - v[event, layout.rv : layout.rv + 3]
                )
            )
        )
        return dict(
            handoff=witnessed,
            a_failed=bool(failed[:, 0].any()),
            b_failed=bool(failed[:, 1].any()),
            first_foot_relative_root_speed_mps=relative,
            relative_speed_not_foot_surface_speed=True,
            **outcome,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=4,
        action="append",
        required=True,
        metavar=("SOURCE", "AUDIT", "PRIOR", "PRIOR_AUDIT"),
    )
    p.add_argument("--audit-hash", action="append", required=True)
    p.add_argument("--prior-audit-hash", action="append", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.group) != 3 or len(a.audit_hash) != 3 or len(a.prior_audit_hash) != 3:
        raise ValueError("all declared paired blocks and exact replay pins required")
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    records, pins, seen = [], [], set()
    for group, ah, ph in zip(a.group, a.audit_hash, a.prior_audit_hash, strict=True):
        root, audit, prior_root, prior_audit = map(Path, group)
        src, _, sh = report(
            root,
            audit,
            ah,
            "NEURAL_RELAY_RECEIVER_VELOCITY_MATCH_PREPARED16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        prior, _, oldhash = report(
            prior_root,
            prior_audit,
            ph,
            "NEURAL_RELAY_RETURN_PLAYER_PREPARE16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        context = tuple(sorted({r["course"]["context"] for r in src["records"]}))
        if context not in ((2, 3, 5, 7), (8, 9, 11, 13), (15, 16, 17, 18)) or context in seen:
            raise ValueError("all unique predeclared development blocks required")
        seen.add(context)
        scene = root / "scene.xml"
        scene_hash = digest(scene)
        if scene.is_symlink() or digest(prior_root / "scene.xml") != scene_hash:
            raise ValueError("exact independently replayed shared scene required")
        model = mujoco.MjModel.from_xml_path(str(scene))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        for i in range(0, 16, 2):
            old, new, expected = src["records"][i], src["records"][i + 1], prior["records"][i + 1]
            if (
                old["course"]["context"] != new["course"]["context"]
                or old["course"]["lateral_m"] != new["course"]["lateral_m"]
                or old["course"]["arm"]["receiver_adaptive_intercept"] is not False
                or new["course"]["arm"]["receiver_adaptive_intercept"] is not True
                or old["artifacts"]["initial"]["hash"] != new["artifacts"]["initial"]["hash"]
                or old["artifacts"]["physics"]["hash"] != expected["artifacts"]["physics"]["hash"]
            ):
                raise ValueError(
                    "exact paired initial conditions and prepared-role baseline required"
                )
            oldv, newv = bind(root, old, "receiver_velocity"), bind(root, new, "receiver_velocity")
            if oldv is None or newv is None:
                raise ValueError("actual receiver command traces required")
            with np.load(oldv, allow_pickle=False) as a0, np.load(newv, allow_pickle=False) as a1:
                n = min(len(a0["command"]), len(a1["command"]))
                changed = np.any(a0["command"][:n] != a1["command"][:n], axis=1)
                first = int(np.flatnonzero(changed)[0]) if changed.any() else None
            common = min(old["controls"], new["controls"])
            records.append(
                dict(
                    context=old["course"]["context"],
                    side=old["course"]["lateral_m"],
                    first_command_divergence=first,
                    common_controls=common,
                    baseline_termination=old["termination"],
                    candidate_termination=new["termination"],
                    baseline_full=physical_outcome(root, old, old["controls"], model, layout),
                    candidate_full=physical_outcome(root, new, new["controls"], model, layout),
                    baseline_common=physical_outcome(root, old, common, model, layout),
                    candidate_common=physical_outcome(root, new, common, model, layout),
                )
            )
        if digest(scene) != scene_hash:
            raise ValueError("audited physics scene changed during review")
        pins.append(dict(source=sh, audit=ah, prior=oldhash, prior_audit=ph, scene=scene_hash))
    terms = Counter((r["baseline_termination"], r["candidate_termination"]) for r in records)
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RECEIVER_VELOCITY_MATCH_REPLAYED_DEVELOPMENT_ONLY",
            source_head=a.expected_head,
            conditions=len(records),
            source_pins=pins,
            records=records,
            termination_pairs=[
                dict(baseline=k[0], candidate=k[1], count=v) for k, v in sorted(terms.items())
            ],
            same_scene_verified_by_own_replay=True,
            baseline_matches_prior_physics=True,
            command_change_is_unlearned=True,
            possession_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
