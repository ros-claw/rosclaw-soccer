"""Compare independently replayed CFtrack shared-world development pairs."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.strict_receive_screen import screen_receive
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report

PHYSICAL_FIELDS = (
    "qpos",
    "qvel",
    "torque",
    "ball_forces",
    "body_failed",
    "time_before",
    "target_b",
)


def compare_prefix(left, right, fields, controls):
    if type(controls) is not int or controls < 0:
        raise ValueError("explicit nonnegative shared physical prefix required")
    for field in fields:
        if (
            field not in left
            or field not in right
            or len(left[field]) < controls
            or len(right[field]) < controls
        ):
            raise ValueError("complete named shared prefix required")
        if not np.array_equal(left[field][:controls], right[field][:controls]):
            raise ValueError("actual paired prefix differs")


def own_report(source, audit, source_hash, audit_hash, *, early_cftrack=False):
    source_path, audit_path = source / "complete.json", audit / "complete.json"
    if (
        any(x.is_symlink() for x in (source_path, audit_path))
        or digest(source_path) != source_hash
        or digest(audit_path) != audit_hash
    ):
        raise ValueError("exact source and own replay receipt required")
    if (source / "failed.json").exists() or (audit / "failed.json").exists():
        raise ValueError("failed campaigns cannot be promoted as complete")
    original, replay = (json.loads(x.read_text()) for x in (source_path, audit_path))
    if (
        original.get("status")
        != (
            "SHARED_CFTRACK_EARLY_16_PENDING_OWN_REPLAY"
            if early_cftrack
            else "SHARED_CFTRACK_16_PENDING_OWN_REPLAY"
        )
        or replay.get("status")
        != (
            "SHARED_CFTRACK_EARLY_ALL_OWN_HISTORY_MATCH"
            if early_cftrack
            else "SHARED_CFTRACK_ALL_OWN_HISTORY_MATCH"
        )
        or replay.get("producer_hash") != source_hash
        or original.get("source_head") != replay.get("source_head")
        or original.get("external_head") != replay.get("external_head")
        or len(original.get("records", [])) != 16
        or len(replay.get("records", [])) != 16
        or any(
            doc.get(k) is not False
            for doc in (original, replay)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("complete matching unpromoted 16-world own-history replay required")
    declarations = []
    for root, item in ((source, original), (audit, replay)):
        declaration, scene = root / "declaration.json", root / "scene.xml"
        if (
            declaration.is_symlink()
            or scene.is_symlink()
            or digest(declaration) != item.get("declaration_hash")
            or digest(scene) != item.get("scene_hash")
        ):
            raise ValueError("source and replay declarations and scenes must be receipt-bound")
        declarations.append(json.loads(declaration.read_text()))
    if (
        declarations[0].get("input_file_hashes") != declarations[1].get("input_file_hashes")
        or original["scene_hash"] != replay["scene_hash"]
        or any(d.get("early_cftrack", False) is not early_cftrack for d in declarations)
    ):
        raise ValueError("exact original source assets and compiled scene required")
    for i, (x, y) in enumerate(zip(original["records"], replay["records"], strict=True)):
        if (
            x["index"] != i
            or y["index"] != i
            or any(
                x[k] != y[k]
                for k in (
                    "context",
                    "lateral_m",
                    "cftrack",
                    "termination",
                    "controls",
                    "native_steps",
                    "handoff",
                )
            )
        ):
            raise ValueError("same exact source/replay outcomes required")
        for name in x["artifacts"]:
            xp, yp = bind(source, x, name), bind(audit, y, name)
            if (xp is None) != (yp is None):
                raise ValueError("same source/replay artifact presence required")
            if xp is not None:
                with np.load(xp, allow_pickle=False) as a, np.load(yp, allow_pickle=False) as b:
                    if set(a.files) != set(b.files) or any(
                        not np.array_equal(a[k], b[k]) for k in a.files
                    ):
                        raise ValueError("independent own physical/neural histories differ")
    return original


def outcome(root, row, horizon, model, b):
    with np.load(bind(root, row, "physics"), allow_pickle=False) as z:
        forces = z["ball_forces"][:horizon].reshape(-1, 2, 3)
        q = z["qpos"][:horizon].reshape(-1, model.nq)
        v = z["qvel"][:horizon].reshape(-1, model.nv)
        failed = z["body_failed"][:horizon]
        screen = screen_receive(
            forces, q, v, failed, ball_q=b.bq, ball_v=b.bv, root_q=b.rq, root_v=b.rv
        )
        foot = np.flatnonzero((forces[:, 1, :2] > 10).any(axis=1))
        other = np.flatnonzero(forces[:, 1, 2] > 1)
        handoff = row["handoff"]
        strike, a_foot, a_other = None, None, None
        if handoff is not None and 0 < handoff["control"] < horizon:
            first = handoff["control"] * 10
            origin = q[first - 1, b.bq : b.bq + 2]
            direction = np.asarray(handoff["return_goal"][:2]) - origin
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("finite actual return direction required")
            hits = np.flatnonzero(
                (forces[first:, 1, :2] > 10).any(axis=1)
                & (v[first:, b.bv : b.bv + 2] @ (direction / norm) >= 1)
            )
            if len(hits):
                strike = first + int(hits[0])
                af = np.flatnonzero((forces[strike:, 0, :2] > 10).any(axis=1))
                ao = np.flatnonzero(forces[strike:, 0, 2] > 1)
                a_foot = strike + int(af[0]) if len(af) else None
                a_other = strike + int(ao[0]) if len(ao) else None
        return dict(
            controls=horizon,
            handoff=handoff is not None and handoff["control"] < horizon,
            a_failed=bool(failed[:, 0].any()),
            b_failed=bool(failed[:, 1].any()),
            first_b_foot_substep=int(foot[0]) if len(foot) else None,
            first_b_nonfoot_substep=int(other[0]) if len(other) else None,
            directed_b_strike_substep=strike,
            teammate_a_foot_substep=a_foot,
            teammate_a_nonfoot_substep=a_other,
            screen=screen,
            teammate_possession_proven=False,
            nonfoot_is_not_legality=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "audit", "prior", "prior-audit", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    for flag in ("source-hash", "audit-hash", "prior-audit-hash", "expected-head"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--early-cftrack", action="store_true")
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    current = own_report(
        a.source, a.audit, a.source_hash, a.audit_hash, early_cftrack=a.early_cftrack
    )
    prior, _, ph = report(
        a.prior,
        a.prior_audit,
        a.prior_audit_hash,
        "NEURAL_RELAY_RECEIVER_COORDINATED_INTERCEPT16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    scene = a.source / "scene.xml"
    scene_hash = digest(scene)
    if scene.is_symlink() or digest(a.prior / "scene.xml") != scene_hash:
        raise ValueError("same source/qualified baseline scene required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for i in range(0, 16, 2):
        base, candidate, old = current["records"][i], current["records"][i + 1], prior["records"][i]
        if (
            base["cftrack"]
            or not candidate["cftrack"]
            or base["context"] != candidate["context"]
            or base["lateral_m"] != candidate["lateral_m"]
            or base["context"] != old["course"]["context"]
            or base["lateral_m"] != old["course"]["lateral_m"]
        ):
            raise ValueError("matched original baseline and intervention required")
        if a.early_cftrack:
            prefix = min(
                base["controls"] if base["handoff"] is None else base["handoff"]["control"],
                candidate["controls"]
                if candidate["handoff"] is None
                else candidate["handoff"]["control"],
            )
        else:
            if (base["handoff"] is None) != (candidate["handoff"] is None):
                raise ValueError("intervention cannot change readiness before switching")
            prefix = base["controls"] if base["handoff"] is None else base["handoff"]["control"]
            if candidate["handoff"] is not None and candidate["handoff"]["control"] != prefix:
                raise ValueError("identical switch boundary required")
        with (
            np.load(bind(a.source, base, "physics"), allow_pickle=False) as x,
            np.load(bind(a.source, candidate, "physics"), allow_pickle=False) as y,
            np.load(bind(a.prior, old, "physics"), allow_pickle=False) as z,
        ):
            compare_prefix(x, y, PHYSICAL_FIELDS, prefix)
            compare_prefix(x, z, PHYSICAL_FIELDS, min(base["controls"], old["controls"]))
            if base["controls"] != old["controls"]:
                raise ValueError("original baseline physical horizon differs")
        common = min(base["controls"], candidate["controls"])
        rows.append(
            dict(
                context=base["context"],
                side=base["lateral_m"],
                prefix_controls=prefix,
                baseline_termination=base["termination"],
                cftrack_termination=candidate["termination"],
                baseline_full=outcome(a.source, base, base["controls"], model, layout),
                cftrack_full=outcome(a.source, candidate, candidate["controls"], model, layout),
                baseline_common=outcome(a.source, base, common, model, layout),
                cftrack_common=outcome(a.source, candidate, common, model, layout),
                cftrack_neural_controls=0
                if candidate["artifacts"]["cftrack"] is None
                else candidate["controls"] - candidate["handoff"]["control"],
            )
        )
    if digest(scene) != scene_hash:
        raise ValueError("scene changed during review")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="SHARED_CFTRACK_EARLY_REPLAYED_DEVELOPMENT_ONLY"
            if a.early_cftrack
            else "SHARED_CFTRACK_REPLAYED_DEVELOPMENT_ONLY",
            source_hash=a.source_hash,
            audit_hash=a.audit_hash,
            prior_hash=ph,
            prior_audit_hash=a.prior_audit_hash,
            scene_hash=scene_hash,
            code_head=a.expected_head,
            rows=rows,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
