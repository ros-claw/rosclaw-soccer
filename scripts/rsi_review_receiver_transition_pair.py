"""Own-replayed B successor transitions versus exact previous child baseline."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import onnx

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.paid_teacher_policy import strict_parameters
from rosclaw_soccer.rsi.transition_feasibility import transition_feasibility
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_receiver_generation_pair import observed_body_margins, team_followup
from scripts.rsi_review_receiver_match_prepared import physical_outcome
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def verify_before_handoff(root, direct, smooth):
    """Require identical actual physics and private B histories before intervention."""
    if direct["artifacts"]["initial"]["hash"] != smooth["artifacts"]["initial"]["hash"]:
        raise ValueError("identical initial physical state required")
    x, y = direct["handoff"], smooth["handoff"]
    if (x is None) != (y is None):
        raise ValueError("intervention cannot change pre-handoff trigger")
    if x is None:
        for key in ("physics", "receiver_velocity", "receiver_shadow"):
            if direct["artifacts"][key]["hash"] != smooth["artifacts"][key]["hash"]:
                raise ValueError("no-handoff paths must remain identical")
        return None
    if (
        any(x[key] != y[key] for key in ("control", "return_goal", "kick_origin"))
        or x["boundary"]["hash"] != y["boundary"]["hash"]
    ):
        raise ValueError("identical causal handoff state required")
    count = x["control"]
    for key in ("physics", "receiver_velocity", "receiver_shadow"):
        with (
            np.load(bind(root, direct, key), allow_pickle=False) as old,
            np.load(bind(root, smooth, key), allow_pickle=False) as new,
        ):
            if set(old.files) != set(new.files) or any(
                len(old[k]) < count
                or len(new[k]) < count
                or not np.array_equal(old[k][:count], new[k][:count])
                for k in old.files
            ):
                raise ValueError("pre-handoff actual physical and neural histories must match")
    return count


def transition_target_metrics(root, row, codec=None):
    if row["handoff"] is None:
        return None
    step = row["handoff"]["control"]
    with np.load(bind(root, row, "physics"), allow_pickle=False) as z:
        targets = z["target_b"]
        if step < 1 or step >= len(targets):
            raise ValueError("actual previous and successor targets required")
        jumps = np.max(
            np.abs(np.diff(targets[step - 1 : min(len(targets), step + 10)], axis=0)), axis=1
        )
        metrics = dict(
            first_target_jump_rad=float(jumps[0]),
            transition_max_target_jump_rad=float(jumps.max()),
            observed_transition_controls=len(jumps),
        )
        if codec is not None:
            offset, scale, order = codec
            with np.load(bind(root, row, "receiver_return_nn"), allow_pickle=False) as nn:
                if not len(nn["control"]) or int(nn["control"][0]) != step:
                    raise ValueError("reference must belong to actual first handoff control")
                reference = offset + nn["NN_reference_nominal"][0][order] * scale
            bound = transition_feasibility(
                targets[step - 1], reference, reference_radius=0.20002, max_step=0.0
            )
            minimum = bound["maximum_minimum_jump_rad"]
            if jumps[0] + 1e-6 < minimum:
                raise ValueError("executed first jump contradicts reference-envelope geometry")
            joint = bound["limiting_joint_index"]
            metrics.update(
                reference_envelope_radius_rad=0.20002,
                unavoidable_first_jump_lower_bound_rad=minimum,
                limiting_joint=G1_DDS_JOINT_NAMES[joint],
                lower_bound_is_not_a_safety_certificate=True,
            )
        return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--prior-audit", type=Path, required=True)
    p.add_argument("--prior-audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    source, _, sh = report(
        a.source,
        a.audit,
        a.audit_hash,
        "NEURAL_RELAY_CHILD_SUCCESSOR_TRANSITION16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    prior, _, ph = report(
        a.prior,
        a.prior_audit,
        a.prior_audit_hash,
        "NEURAL_RELAY_CHILD_VELOCITY_GENERATION_PAIR16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    policies = [
        Path(path)
        for path in source["input_file_hashes"]
        if path.endswith("/ckp/policy_30000.onnx")
    ]
    if len(policies) != 1 or digest(policies[0]) != source["input_file_hashes"][str(policies[0])]:
        raise ValueError("one hash-bound original reference policy required")
    graph = onnx.load(policies[0], load_external_data=False)
    names, params = strict_parameters({x.key: x.value for x in graph.metadata_props})
    order = np.array([names.index(name) for name in G1_DDS_JOINT_NAMES])
    codec = (params["default_joint_pos"][order], params["action_scale"][order], order)
    scene = a.source / "scene.xml"
    scene_hash = digest(scene)
    if scene.is_symlink() or digest(a.prior / "scene.xml") != scene_hash:
        raise ValueError("identical independently replayed scene required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    la = NativeVelocityLayout(
        model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
    )
    lb = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for i in range(0, 16, 2):
        old, new, prior_child = (
            source["records"][i],
            source["records"][i + 1],
            prior["records"][i + 1],
        )
        if (
            old["course"]["context"] != new["course"]["context"]
            or old["course"]["lateral_m"] != new["course"]["lateral_m"]
            or old["course"]["arm"]["receiver_child_candidate"] is not True
            or new["course"]["arm"]["receiver_child_candidate"] is not True
            or old["course"]["arm"]["receiver_successor_smooth"] is not False
            or new["course"]["arm"]["receiver_successor_smooth"] is not True
            or old["artifacts"]["physics"]["hash"] != prior_child["artifacts"]["physics"]["hash"]
        ):
            raise ValueError("matched fixed candidate and exact v56 child baseline required")
        handoff = verify_before_handoff(a.source, old, new)
        for arm in (old, new):
            with np.load(bind(a.source, arm, "receiver_velocity"), allow_pickle=False) as trace:
                if not trace["candidate_actor_selected"].all():
                    raise ValueError("both worlds must actually execute the same candidate actor")
        common = min(old["controls"], new["controls"])
        rows.append(
            dict(
                context=old["course"]["context"],
                side=old["course"]["lateral_m"],
                handoff_control=handoff,
                direct_controls=old["controls"],
                smooth_controls=new["controls"],
                common_controls=common,
                direct_termination=old["termination"],
                smooth_termination=new["termination"],
                direct_full=physical_outcome(a.source, old, old["controls"], model, lb),
                smooth_full=physical_outcome(a.source, new, new["controls"], model, lb),
                direct_common=physical_outcome(a.source, old, common, model, lb),
                smooth_common=physical_outcome(a.source, new, common, model, lb),
                direct_body_margins=observed_body_margins(a.source, old, common, la, lb),
                smooth_body_margins=observed_body_margins(a.source, new, common, la, lb),
                direct_team=team_followup(a.source, old, common, model, la),
                smooth_team=team_followup(a.source, new, common, model, la),
                # The direct arm does not record reference_nominal. Do not
                # invent it from the paired arm after trajectories diverge.
                direct_target=transition_target_metrics(a.source, old),
                smooth_target=transition_target_metrics(a.source, new, codec),
            )
        )
    if (
        digest(scene) != scene_hash
        or digest(policies[0]) != source["input_file_hashes"][str(policies[0])]
    ):
        raise ValueError("scene or reference policy changed during review")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="REPLAYED_RECEIVER_TRANSITION_DEVELOPMENT_ONLY",
            source_hash=sh,
            audit_hash=a.audit_hash,
            prior_hash=ph,
            prior_audit_hash=a.prior_audit_hash,
            code_head=a.expected_head,
            scene_hash=scene_hash,
            rows=rows,
            unlearned_intervention=True,
            team_pass_proven=False,
            physical_retention_proven=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
