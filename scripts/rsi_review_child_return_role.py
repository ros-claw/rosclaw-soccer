"""Review replayed causal team return-role pairs against exact learned child."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.team_return_interaction import return_interaction
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal, handoff_equal


def assert_causal_prefix(root, baseline, candidate):
    boundary = candidate["passer_velocity_recovery_boundary"]
    controls = min(baseline["controls"], candidate["controls"])
    cutoff = boundary["control"] if boundary is not None else controls
    if type(cutoff) is not int or not 0 <= cutoff <= controls:
        raise ValueError("valid observed role-switch boundary required")
    for name in (
        "physics",
        "passer_nn",
        "receiver_velocity",
        "receiver_shadow",
        "receiver_return_nn",
    ):
        left, right = bind(root, baseline, name), bind(root, candidate, name)
        if name == "receiver_return_nn" and left is None and right is None:
            continue
        if left is None or right is None:
            raise ValueError("complete paired own-state prefix required")
        with np.load(left, allow_pickle=False) as x, np.load(right, allow_pickle=False) as y:
            if set(x.files) != set(y.files):
                raise ValueError("matching own-state prefix fields required")
            if "control" in x:
                if (
                    x["control"].ndim != 1
                    or y["control"].ndim != 1
                    or np.any(np.diff(x["control"]) <= 0)
                    or np.any(np.diff(y["control"]) <= 0)
                ):
                    raise ValueError("ordered explicit control clocks required")
                left_limit = int(np.searchsorted(x["control"], cutoff))
                right_limit = int(np.searchsorted(y["control"], cutoff))
            else:
                left_limit = right_limit = cutoff
            for key in x.files:
                left_count, right_count = (
                    min(left_limit, len(x[key])),
                    min(right_limit, len(y[key])),
                )
                if left_count != right_count or not np.array_equal(
                    x[key][:left_count], y[key][:right_count]
                ):
                    raise ValueError("physical or neural history changed before role switch")
    return cutoff


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=8,
        action="append",
        required=True,
        metavar=(
            "SEED",
            "BLOCK",
            "SOURCE",
            "AUDIT",
            "AUDIT_HASH",
            "PRIOR",
            "PRIOR_AUDIT",
            "PRIOR_AUDIT_HASH",
        ),
    )
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.group) != 9 or a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("nine exact groups and unused external output required")
    rows, pins, provenance, seen, actor_by_seed = [], [], [], set(), {}
    for ss, bs, source_name, audit_name, ah, prior_name, prior_audit_name, ph in a.group:
        seed, block = int(ss), int(bs)
        if seed not in (92401, 92402, 92403) or block not in (0, 1, 2) or (seed, block) in seen:
            raise ValueError("distinct seed and block required")
        seen.add((seed, block))
        root, prior, audit, prior_audit = map(
            Path, (source_name, prior_name, audit_name, prior_audit_name)
        )
        source, _, sh = report(
            root, audit, ah, "NEURAL_RELAY_CHILD_RETURN_ROLE16_PENDING_OWN_REPLAY", worlds=16
        )
        parent, _, psh = report(
            prior,
            prior_audit,
            ph,
            "NEURAL_RELAY_RECEIVER_PRECONTACT_CONTINUATION16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        decs = [json.loads((r / "declaration.json").read_text()) for r in (root, prior)]
        if (
            any(d.get("development_context_block") != block for d in decs)
            or decs[0]["receiver_checkpoint_hash"] != decs[1]["receiver_checkpoint_hash"]
            or any(
                digest(r / "declaration.json") != s["declaration_hash"]
                for r, s in ((root, source), (prior, parent))
            )
        ):
            raise ValueError("exact same child and development block required")
        checkpoint_hash = decs[0]["receiver_checkpoint_hash"]
        if seed in actor_by_seed and actor_by_seed[seed] != checkpoint_hash:
            raise ValueError("each seed group must use one exact actor")
        actor_by_seed[seed] = checkpoint_hash
        audit_candidate = Path(decs[0]["receiver_candidate_audit"])
        if (
            audit_candidate.is_symlink()
            or digest(audit_candidate) != decs[0]["receiver_candidate_audit_hash"]
        ):
            raise ValueError("exact candidate readback required")
        candidate_audit = json.loads(audit_candidate.read_text())
        if candidate_audit.get("checkpoint_hash") != decs[0]["receiver_checkpoint_hash"]:
            raise ValueError("candidate audit does not bind the actor")
        scene, old_scene = root / "scene.xml", prior / "scene.xml"
        scene_hash = digest(scene)
        if scene.is_symlink() or old_scene.is_symlink() or digest(old_scene) != scene_hash:
            raise ValueError("exact matching native scene required")
        model = mujoco.MjModel.from_xml_path(str(scene))
        layout = NativeVelocityLayout(
            model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
        )
        pins.extend(
            (
                (root / "complete.json", sh),
                (prior / "complete.json", psh),
                (audit, ah),
                (prior_audit, ph),
                (root / "declaration.json", source["declaration_hash"]),
                (prior / "declaration.json", parent["declaration_hash"]),
                (audit_candidate, decs[0]["receiver_candidate_audit_hash"]),
                (scene, scene_hash),
                (old_scene, scene_hash),
            )
        )
        provenance.append(
            dict(
                seed=seed,
                block=block,
                source_hash=sh,
                audit_hash=ah,
                prior_hash=psh,
                prior_audit_hash=ph,
                scene_hash=scene_hash,
                checkpoint_hash=decs[0]["receiver_checkpoint_hash"],
            )
        )
        for i in range(0, 16, 2):
            base, role = source["records"][i : i + 2]
            old = parent["records"][i + 1]
            a0, a1 = base["course"], role["course"]
            if (
                (a0["context"], a0["lateral_m"]) != (a1["context"], a1["lateral_m"])
                or (a0["context"], a0["lateral_m"])
                != (old["course"]["context"], old["course"]["lateral_m"])
                or a0["arm"]["passer_velocity_recovery"] is not False
                or a1["arm"]["passer_velocity_recovery"] is not True
                or not a0["arm"]["receiver_learned"]
                or not a1["arm"]["receiver_learned"]
            ):
                raise ValueError("matched child and paired role arms required")
            for name in old["artifacts"]:
                if not artifact_equal(prior, old, root, base, name):
                    raise ValueError("no-role child baseline history changed")
            if not handoff_equal(prior, old, root, base) or any(
                old[k] != base[k] for k in ("termination", "controls", "native_steps", "result")
            ):
                raise ValueError("no-role child baseline outcome changed")
            cutoff = assert_causal_prefix(root, base, role)
            failures, interactions = [], []
            for trial in (base, role):
                path = bind(root, trial, "physics")
                if path is None:
                    raise ValueError("native trajectory required")
                pins.append((path, trial["artifacts"]["physics"]["hash"]))
                with np.load(path, allow_pickle=False) as z:
                    failures.append(z["body_failed"].copy())
                    h = trial["handoff"]
                    if h is None:
                        interactions.append(None)
                    else:
                        direction = np.asarray(h["return_goal"][:2]) - np.asarray(
                            h["kick_origin"][:2]
                        )
                        interactions.append(
                            return_interaction(
                                z["qpos"].reshape(-1, model.nq),
                                z["qvel"].reshape(-1, model.nv),
                                z["ball_forces"].reshape(-1, 2, 3),
                                z["body_failed"],
                                start=h["control"] * 10,
                                direction=direction,
                                ball_q=layout.bq,
                                ball_v=layout.bv,
                                player_q=layout.rq,
                                player_v=layout.rv,
                            )
                        )
            n = min(len(failures[0]), len(failures[1]))
            rows.append(
                dict(
                    seed=seed,
                    block=block,
                    context=a0["context"],
                    side=a0["lateral_m"],
                    prefix_controls=cutoff,
                    role_trigger=role["passer_velocity_recovery_boundary"] is not None,
                    horizons=[len(x) for x in failures],
                    common_healthy=[not bool(x[:n].any()) for x in failures],
                    observed_horizon_healthy=[not bool(x.any()) for x in failures],
                    interaction=interactions,
                )
            )
    if (
        len(rows) != 72
        or len(seen) != 9
        or len(set(actor_by_seed.values())) != 3
        or any(digest(path) != h for path, h in pins)
    ):
        raise ValueError("all complete unchanged paired histories required")
    summaries = {}
    for seed in (92401, 92402, 92403):
        group = [r for r in rows if r["seed"] == seed]
        summaries[str(seed)] = dict(
            conditions=len(group),
            triggers=sum(r["role_trigger"] for r in group),
            healthy_common=[sum(r["common_healthy"][j] for r in group) for j in (0, 1)],
            healthy_observed_horizon=[
                sum(r["observed_horizon_healthy"][j] for r in group) for j in (0, 1)
            ],
            rescues=sum(not r["common_healthy"][0] and r["common_healthy"][1] for r in group),
            new_failures=sum(r["common_healthy"][0] and not r["common_healthy"][1] for r in group),
            unequal_horizons=sum(r["horizons"][0] != r["horizons"][1] for r in group),
            interactions=[
                dict(
                    directed_strikes=sum(
                        r["interaction"][j] is not None
                        and r["interaction"][j]["directed_strike_step"] is not None
                        for r in group
                    ),
                    foot_contacts=sum(
                        r["interaction"][j] is not None
                        and r["interaction"][j]["first_A_foot_step"] is not None
                        for r in group
                    ),
                    nonfoot_contacts=sum(
                        r["interaction"][j] is not None
                        and r["interaction"][j]["first_A_nonfoot_step"] is not None
                        for r in group
                    ),
                    near_200ms=sum(
                        r["interaction"][j] is not None
                        and r["interaction"][j]["near_healthy_longest_substeps"] >= 100
                        for r in group
                    ),
                    near_1s=sum(
                        r["interaction"][j] is not None
                        and r["interaction"][j]["near_healthy_longest_substeps"] >= 500
                        for r in group
                    ),
                )
                for j in (0, 1)
            ],
        )
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="CHILD_RETURN_ROLE_PAIRED_DEVELOPMENT_NOT_PROMOTED",
            source_head=a.expected_head,
            provenance=provenance,
            summaries=summaries,
            records=rows,
            baseline_child_exact=True,
            causal_prefix_exact=True,
            near_window_not_possession=True,
            unlearned_role=True,
            reused_development_contexts=True,
            own_replay_not_independent_simulator=True,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
