"""Compare replayed gated continuation, its exact parent and zero baseline."""

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
from scripts.rsi_review_whole_receiver_identity import artifact_equal, handoff_equal


def paired_health(first: np.ndarray, second: np.ndarray) -> dict:
    if (
        first.ndim != 2
        or second.ndim != 2
        or first.shape[1:] != (2,)
        or second.shape[1:] != (2,)
        or first.dtype != np.bool_
        or second.dtype != np.bool_
        or not len(first)
        or not len(second)
    ):
        raise ValueError("complete two-body boolean failure traces required")
    n = min(len(first), len(second))
    return dict(
        common_controls=n,
        horizons=[len(first), len(second)],
        healthy=[not bool(first[:n].any()), not bool(second[:n].any())],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=8,
        action="append",
        required=True,
        metavar=(
            "SEED",
            "BLOCK",
            "CHILD",
            "CHILD_AUDIT",
            "CHILD_AUDIT_HASH",
            "PARENT",
            "PARENT_AUDIT",
            "PARENT_AUDIT_HASH",
        ),
    )
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.group) != 9 or a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("nine complete paired groups and unused external output required")
    rows, pins, provenance, seen = [], [], [], set()
    for (
        ss,
        bs,
        child_root,
        child_audit,
        child_hash,
        parent_root,
        parent_audit,
        parent_hash,
    ) in a.group:
        seed, block = int(ss), int(bs)
        if seed not in (92401, 92402, 92403) or block not in (0, 1, 2) or (seed, block) in seen:
            raise ValueError("three unique seeds and blocks required")
        seen.add((seed, block))
        roots = [Path(parent_root), Path(child_root)]
        audits = [Path(parent_audit), Path(child_audit)]
        audit_hashes = [parent_hash, child_hash]
        parent, _, parent_source_hash = report(
            roots[0],
            audits[0],
            audit_hashes[0],
            "NEURAL_RELAY_RECEIVER_PRECONTACT_PAIR16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        child, _, child_source_hash = report(
            roots[1],
            audits[1],
            audit_hashes[1],
            "NEURAL_RELAY_RECEIVER_PRECONTACT_CONTINUATION16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        declarations = [r / "declaration.json" for r in roots]
        decs = [json.loads(d.read_text()) for d in declarations]
        for d, dec, source in zip(declarations, decs, (parent, child), strict=True):
            if (
                d.is_symlink()
                or digest(d) != source["declaration_hash"]
                or dec["development_context_block"] != block
            ):
                raise ValueError("exact distinct declared physical block required")
        child_audit_path = Path(decs[1]["receiver_candidate_audit"])
        if (
            child_audit_path.is_symlink()
            or digest(child_audit_path) != decs[1]["receiver_candidate_audit_hash"]
        ):
            raise ValueError("exact candidate lineage audit required")
        lineage = json.loads(child_audit_path.read_text())
        parent_candidate_audit = Path(decs[0]["receiver_candidate_audit"])
        if (
            lineage.get("parent_hash") != decs[0]["receiver_checkpoint_hash"]
            or lineage.get("checkpoint_hash") != decs[1]["receiver_checkpoint_hash"]
            or parent_candidate_audit.is_symlink()
            or digest(parent_candidate_audit) != decs[0]["receiver_candidate_audit_hash"]
            or json.loads(parent_candidate_audit.read_text()).get("seed") != seed
        ):
            raise ValueError("actual continuation must descend from compared parent")
        scene = roots[1] / "scene.xml"
        if scene.is_symlink():
            raise ValueError("owned native scene required")
        scene_hash = digest(scene)
        if digest(roots[0] / "scene.xml") != scene_hash:
            raise ValueError("same shared physical scene required")
        model = mujoco.MjModel.from_xml_path(str(scene))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        pins.extend(
            (
                (roots[0] / "complete.json", parent_source_hash),
                (roots[1] / "complete.json", child_source_hash),
                (audits[0], audit_hashes[0]),
                (audits[1], audit_hashes[1]),
                (declarations[0], parent["declaration_hash"]),
                (declarations[1], child["declaration_hash"]),
                (scene, scene_hash),
                (roots[0] / "scene.xml", scene_hash),
                (child_audit_path, decs[1]["receiver_candidate_audit_hash"]),
                (parent_candidate_audit, decs[0]["receiver_candidate_audit_hash"]),
            )
        )
        provenance.append(
            dict(
                seed=seed,
                block=block,
                child_source_hash=child_source_hash,
                parent_source_hash=parent_source_hash,
                child_audit_hash=child_hash,
                parent_audit_hash=parent_hash,
                scene_hash=scene_hash,
            )
        )
        for i in range(0, 16, 2):
            original_base, original_parent = parent["records"][i : i + 2]
            base, candidate = child["records"][i : i + 2]
            courses = [r["course"] for r in (original_base, original_parent, base, candidate)]
            key = (courses[0]["context"], courses[0]["lateral_m"])
            if any((c["context"], c["lateral_m"]) != key for c in courses) or [
                c["arm"]["receiver_learned"] for c in courses
            ] != [False, True, False, True]:
                raise ValueError("exact paired course and arm order required")
            for name in base["artifacts"]:
                if not artifact_equal(roots[0], original_base, roots[1], base, name):
                    raise ValueError("old and new fixed baseline histories must match")
            if any(
                original_base[k] != base[k]
                for k in ("termination", "controls", "native_steps", "result")
            ) or not handoff_equal(roots[0], original_base, roots[1], base):
                raise ValueError("fixed baseline outcome and private handoff must match")
            failures, screens = [], []
            for root, trial in (
                (roots[1], base),
                (roots[0], original_parent),
                (roots[1], candidate),
            ):
                physical = bind(root, trial, "physics")
                if physical is None:
                    raise ValueError("native physics required")
                pins.append((physical, trial["artifacts"]["physics"]["hash"]))
                with np.load(physical, allow_pickle=False) as z:
                    failures.append(z["body_failed"].copy())
                    screens.append(
                        screen_receive(
                            z["ball_forces"].reshape(-1, 2, 3),
                            z["qpos"].reshape(-1, model.nq),
                            z["qvel"].reshape(-1, model.nv),
                            z["body_failed"],
                            ball_q=layout.bq,
                            ball_v=layout.bv,
                            root_q=layout.rq,
                            root_v=layout.rv,
                        )
                    )
            rows.append(
                dict(
                    seed=seed,
                    block=block,
                    context=key[0],
                    side=key[1],
                    baseline_vs_child=paired_health(failures[0], failures[2]),
                    parent_vs_child=paired_health(failures[1], failures[2]),
                    handoffs=[
                        r["result"]["handoff_executed"] for r in (base, original_parent, candidate)
                    ],
                    precise_healthy_returns=[
                        r["result"]["return_0p1m_and_healthy"]
                        for r in (base, original_parent, candidate)
                    ],
                    screens=screens,
                )
            )
    if len(rows) != 72 or len(seen) != 9 or any(digest(path) != h for path, h in pins):
        raise ValueError("complete unchanged paired physical evidence required")
    summaries = {}
    for seed in (92401, 92402, 92403):
        group = [r for r in rows if r["seed"] == seed]
        measures = {}
        for key in ("baseline_vs_child", "parent_vs_child"):
            pairs = [r[key] for r in group]
            measures[key] = dict(
                first_healthy=sum(x["healthy"][0] for x in pairs),
                child_healthy=sum(x["healthy"][1] for x in pairs),
                rescues=sum(not x["healthy"][0] and x["healthy"][1] for x in pairs),
                new_failures=sum(x["healthy"][0] and not x["healthy"][1] for x in pairs),
                unequal_horizons=sum(x["horizons"][0] != x["horizons"][1] for x in pairs),
            )
        summaries[str(seed)] = dict(
            conditions=len(group),
            comparisons=measures,
            handoffs=[sum(r["handoffs"][i] for r in group) for i in (0, 1, 2)],
            precise_healthy_returns=[
                sum(r["precise_healthy_returns"][i] for r in group) for i in (0, 1, 2)
            ],
            screens=[
                {
                    status: sum(r["screens"][i]["status"] == status for r in group)
                    for status in (
                        "NO_PASSER_FOOT",
                        "NO_RECEIVER_FOOT",
                        "POST_CONTACT_CENSORED",
                        "SCREEN_REJECTED",
                        "SCREEN_CANDIDATE",
                    )
                }
                for i in (0, 1, 2)
            ],
        )
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="PRECONTACT_CONTINUATION_PAIRED_DEVELOPMENT_NOT_PROMOTED",
            source_head=a.expected_head,
            provenance=provenance,
            summaries=summaries,
            records=rows,
            old_baseline_all_artifacts_identical=True,
            own_replay_not_independent_simulator=True,
            reused_development_contexts=True,
            screen_not_possession=True,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
