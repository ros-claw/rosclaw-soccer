"""Review replayed ankle-frozen and historic teacher interventions in SIM."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.strict_receive_screen import screen_receive
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_receiver_generation_pair import team_followup
from scripts.rsi_review_receiver_match_prepared import physical_outcome
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def teacher_metrics(root, row, *, frozen):
    with (
        np.load(bind(root, row, "receiver_velocity"), allow_pickle=False) as z,
        np.load(bind(root, row, "physics"), allow_pickle=False) as physics,
    ):
        delta = z["FOOT_applied_delta"]
        n = len(delta)
        if (
            delta.shape != (n, 29)
            or not np.isfinite(delta).all()
            or not np.array_equal(z["control"], np.arange(n))
            or not np.array_equal(z["FOOT_target"], physics["target_b"][:n])
        ):
            raise ValueError("actual sequential executed teacher targets required")
        changes = np.max(np.abs(np.diff(np.vstack((np.zeros((1, 29)), delta)), axis=0)), axis=1)
        if not np.allclose(changes, z["FOOT_delta_jump_rad"], rtol=0, atol=1e-12):
            raise ValueError("teacher jump must match actual residual history")
        if frozen and np.any(delta[:, [4, 5, 10, 11]] != 0):
            raise ValueError("ankle corrections must remain exactly frozen")
        return dict(
            controls=n,
            active_controls=int(z["FOOT_active"].sum()),
            actual_nonzero_controls=int(np.any(delta != 0, axis=1).sum()),
            maximum_residual_rad=float(np.abs(delta).max()),
            maximum_residual_slew_rad=float(changes.max()),
            slew_exceeding_controls=int((changes > 0.025 + 1e-12).sum()),
            total_joint_target_slew_not_bounded_by_this_measure=True,
        )


def screen(root, row, model, layout):
    with np.load(bind(root, row, "physics"), allow_pickle=False) as z:
        return screen_receive(
            z["ball_forces"].reshape(-1, 2, 3),
            z["qpos"].reshape(-1, model.nq),
            z["qvel"].reshape(-1, model.nv),
            z["body_failed"],
            ball_q=layout.bq,
            ball_v=layout.bv,
            root_q=layout.rq,
            root_v=layout.rv,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "audit", "prior", "prior-audit", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    for flag in ("audit-hash", "prior-audit-hash", "expected-head"):
        p.add_argument("--" + flag, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    source, _, sh = report(
        a.source,
        a.audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_ANKLE_FREEZE16_PENDING_OWN_REPLAY",
        worlds=16,
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
        raise ValueError("same original simulation model required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    la = NativeVelocityLayout(
        model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
    )
    lb = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for i in range(0, 16, 2):
        base, frozen = source["records"][i : i + 2]
        old_base, old_teacher = prior["records"][i : i + 2]
        if (
            base["course"] != old_base["course"]
            or frozen["course"] != old_teacher["course"]
            or base["artifacts"]["initial"]["hash"] != frozen["artifacts"]["initial"]["hash"]
            or base["artifacts"]["physics"]["hash"] != old_base["artifacts"]["physics"]["hash"]
        ):
            raise ValueError("identical conditions and replayed baseline required")
        common = min(base["controls"], frozen["controls"], old_teacher["controls"])
        cases = {}
        for label, root, row in (
            ("baseline", a.source, base),
            ("old_teacher", a.prior, old_teacher),
            ("frozen", a.source, frozen),
        ):
            cases[label] = dict(
                termination=row["termination"],
                controls=row["controls"],
                full=physical_outcome(root, row, row["controls"], model, lb),
                common=physical_outcome(root, row, common, model, lb),
                screen=screen(root, row, model, lb),
                teammate=team_followup(root, row, common, model, la),
                teacher=teacher_metrics(root, row, frozen=label == "frozen")
                if label != "baseline"
                else None,
            )
        rows.append(
            dict(
                context=base["course"]["context"],
                side=base["course"]["lateral_m"],
                common_controls=common,
                **cases,
            )
        )
    if digest(scene) != scene_hash:
        raise ValueError("scene changed during review")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="ANKLE_FREEZE_REPLAYED_DEVELOPMENT_ONLY",
            source_hash=sh,
            audit_hash=a.audit_hash,
            prior_hash=ph,
            prior_audit_hash=a.prior_audit_hash,
            scene_hash=scene_hash,
            code_head=a.expected_head,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
            screen_is_not_possession=True,
            rows=rows,
        ),
    )


if __name__ == "__main__":
    main()
