"""Review all timed receiver pairs with own-history replay, without promotion."""

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=5,
        action="append",
        required=True,
        metavar=("SEED", "BLOCK", "SOURCE", "AUDIT", "AUDIT_HASH"),
    )
    p.add_argument("--expected-head", required=True)
    p.add_argument("--precontact", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.group) != 9 or a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("nine complete paired groups and unused external output required")
    results, pins, provenance, seen = [], [], [], set()
    for seed_s, block_s, root_s, audit_s, audit_hash in a.group:
        seed, block = int(seed_s), int(block_s)
        if seed not in (92401, 92402, 92403) or block not in (0, 1, 2) or (seed, block) in seen:
            raise ValueError("three unique seeds and three unique development blocks required")
        seen.add((seed, block))
        root, audit = Path(root_s), Path(audit_s)
        source, _, source_hash = report(
            root,
            audit,
            audit_hash,
            "NEURAL_RELAY_RECEIVER_PRECONTACT_PAIR16_PENDING_OWN_REPLAY"
            if a.precontact
            else "NEURAL_RELAY_RECEIVER_FUNCTIONAL_TIMED_PAIR16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        decl_path = root / "declaration.json"
        decl = json.loads(decl_path.read_text())
        if (
            decl_path.is_symlink()
            or digest(decl_path) != source["declaration_hash"]
            or decl.get("development_context_block") != block
            or decl.get("protocol")
            != (
                "soccer.neural_relay_receiver_precontact_pair.development.v41"
                if a.precontact
                else "soccer.neural_relay_receiver_functional_timed_pair.development.v40"
            )
        ):
            raise ValueError("exact declared source protocol required")
        candidate_audit = Path(decl["receiver_candidate_audit"])
        if (
            candidate_audit.is_symlink()
            or digest(candidate_audit) != decl["receiver_candidate_audit_hash"]
            or json.loads(candidate_audit.read_text()).get("seed") != seed
        ):
            raise ValueError("declared candidate audit must match the requested seed")
        scene = root / "scene.xml"
        if scene.is_symlink():
            raise ValueError("physical scene must not be a symlink")
        scene_hash = digest(scene)
        provenance.append(
            dict(
                seed=seed,
                block=block,
                source_root=str(root),
                source_hash=source_hash,
                audit_path=str(audit),
                audit_hash=audit_hash,
                declaration_hash=source["declaration_hash"],
                scene_hash=scene_hash,
                candidate_audit_hash=decl["receiver_candidate_audit_hash"],
            )
        )
        model = mujoco.MjModel.from_xml_path(str(scene))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        pins.extend(
            (
                (root / "complete.json", source_hash),
                (audit, audit_hash),
                (decl_path, source["declaration_hash"]),
                (scene, scene_hash),
                (candidate_audit, decl["receiver_candidate_audit_hash"]),
            )
        )
        for idx in range(0, 16, 2):
            base, candidate = source["records"][idx : idx + 2]
            ca, cb = base["course"], candidate["course"]
            if (
                ca["arm"]["receiver_learned"] is not False
                or cb["arm"]["receiver_learned"] is not True
                or (ca["context"], ca["lateral_m"]) != (cb["context"], cb["lateral_m"])
            ):
                raise ValueError("matching fixed baseline and candidate order required")
            screens, failures, lengths = [], [], []
            for row in (base, candidate):
                physical = bind(root, row, "physics")
                if physical is None:
                    raise ValueError("complete physics required")
                pins.append((physical, row["artifacts"]["physics"]["hash"]))
                with np.load(physical, allow_pickle=False) as z:
                    failed = z["body_failed"].copy()
                    lengths.append(len(failed))
                    failures.append(failed)
                    screens.append(
                        screen_receive(
                            z["ball_forces"].reshape(-1, 2, 3),
                            z["qpos"].reshape(-1, model.nq),
                            z["qvel"].reshape(-1, model.nv),
                            failed,
                            ball_q=layout.bq,
                            ball_v=layout.bv,
                            root_q=layout.rq,
                            root_v=layout.rv,
                        )
                    )
            n = min(lengths)
            healthy = [not bool(x[:n].any()) for x in failures]
            results.append(
                dict(
                    seed=seed,
                    block=block,
                    context=ca["context"],
                    side=ca["lateral_m"],
                    common_controls=n,
                    horizons=lengths,
                    healthy_common=healthy,
                    handoff=[
                        base["result"]["handoff_executed"],
                        candidate["result"]["handoff_executed"],
                    ],
                    precise_healthy_return=[
                        base["result"]["return_0p1m_and_healthy"],
                        candidate["result"]["return_0p1m_and_healthy"],
                    ],
                    endpoint_screens=screens,
                )
            )
    if len(results) != 72 or any(digest(path) != h for path, h in pins):
        raise ValueError("all paired stable source trajectories required")
    summaries = {}
    for seed in (92401, 92402, 92403):
        rows = [r for r in results if r["seed"] == seed]
        pairs = [r["healthy_common"] for r in rows]
        summaries[str(seed)] = dict(
            conditions=len(rows),
            healthy_baseline=sum(x for x, _ in pairs),
            healthy_candidate=sum(y for _, y in pairs),
            rescues=sum(not x and y for x, y in pairs),
            new_failures=sum(x and not y for x, y in pairs),
            unequal_horizons=sum(r["horizons"][0] != r["horizons"][1] for r in rows),
            handoffs=[sum(r["handoff"][i] for r in rows) for i in (0, 1)],
            precise_healthy_returns=[
                sum(r["precise_healthy_return"][i] for r in rows) for i in (0, 1)
            ],
            screens=[
                {
                    status: sum(r["endpoint_screens"][i]["status"] == status for r in rows)
                    for status in (
                        "NO_PASSER_FOOT",
                        "NO_RECEIVER_FOOT",
                        "POST_CONTACT_CENSORED",
                        "SCREEN_REJECTED",
                        "SCREEN_CANDIDATE",
                    )
                }
                for i in (0, 1)
            ],
        )
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="PRECONTACT_RECEIVER_PAIR_REPLAYED_DEVELOPMENT_NOT_PROMOTED"
            if a.precontact
            else "TIMED_RECEIVER_PAIR_REPLAYED_DEVELOPMENT_NOT_PROMOTED",
            source_head=a.expected_head,
            summaries=summaries,
            provenance=provenance,
            records=results,
            own_replay_not_independent_simulator=True,
            reused_development_contexts=True,
            screen_not_possession=True,
            physical_retention_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
