"""Review paired own-replayed early motor transitions against historical physics."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from rosclaw_soccer.rsi.receiving_original_readback import original_reception_readback
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal, handoff_equal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=4,
        action="append",
        required=True,
        metavar=("SOURCE", "AUDIT", "OLD", "OLD_AUDIT"),
    )
    for name in ("source-head", "old-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.group) != 3 or a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("three complete independent paired studies and unused output required")
    rows, seen, frozen = [], set(), []
    blocks = ((2, 3, 5, 7), (8, 9, 11, 13), (15, 16, 17, 18))
    for source_name, audit_name, old_name, old_audit_name in a.group:
        root, audit, old, old_audit = map(Path, (source_name, audit_name, old_name, old_audit_name))
        ah, oh = digest(audit), digest(old_audit)
        source, _, sh = report(
            root, audit, ah, "NEURAL_RELAY_PRECONTACT_FULLBODY16_PENDING_OWN_REPLAY", worlds=16
        )
        previous, _, ph = report(
            old,
            old_audit,
            oh,
            "NEURAL_RELAY_WHOLE_RECEIVER_IDENTITY16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        if source["source_head"] != a.source_head or previous["source_head"] != a.old_head:
            raise ValueError("exact frozen physical producers required")
        scene = root / "scene.xml"
        scene_hash = digest(scene)
        model = mujoco.MjModel.from_xml_path(str(scene))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        current = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["precontact_fullbody"],
            ): r
            for r in source["records"]
        }
        parents = {
            (r["course"]["context"], r["course"]["lateral_m"]): r
            for r in previous["records"]
            if r["course"]["arm"]["receiver_whole"] is False
        }
        if len(current) != 16 or len(parents) != 8:
            raise ValueError("complete two-arm source and old baseline required")
        block = next(
            (i for i, group in enumerate(blocks) if set(c for c, _ in parents) == set(group)), None
        )
        if block is None or block in seen:
            raise ValueError("distinct original blocks required")
        seen.add(block)
        for key, prior in sorted(parents.items()):
            base, candidate = current[(*key, False)], current[(*key, True)]
            same = all(
                base[k] == prior[k] for k in ("termination", "controls", "native_steps", "result")
            ) and handoff_equal(root, base, old, prior)
            same = same and all(
                artifact_equal(root, base, old, prior, name) for name in prior["artifacts"]
            )
            outcomes = {}
            for name, row in (("baseline", base), ("candidate", candidate)):
                path = bind(root, row, "physics")
                if path is None:
                    raise ValueError("full physical contact history required")
                with np.load(path, allow_pickle=False) as f:
                    value = strict_pass_receive(
                        f["ball_forces"],
                        f["body_failed"],
                        handoff=row["result"]["handoff_executed"],
                    )
                    readback = original_reception_readback(
                        f["qpos"],
                        f["qvel"],
                        f["ball_forces"],
                        f["body_failed"],
                        receiver_q=layout.rq,
                        receiver_v=layout.rv,
                        ball_q=layout.bq,
                        ball_v=layout.bv,
                        excluded_contact_ever=any(
                            name.startswith("b/")
                            for name in row["result"]["excluded_contact_bodies"]
                        ),
                    )
                if (
                    readback["reconstruction_exact"]
                    and readback["first_ready_control"] != row["original_receiving_ready_control"]
                ):
                    raise ValueError(
                        "independent physical original reception reconstruction differs"
                    )
                value["original_readback"] = readback
                value["original"] = measured(root, row)
                value["motor_transition_executed"] = row["motor_transition_executed"]
                value["original_receiving_ready_control"] = row["original_receiving_ready_control"]
                value["transition_control"] = (
                    row["handoff"]["control"] if row["handoff"] is not None else None
                )
                if value["original_handoff"] != (
                    value["original_receiving_ready_control"] is not None
                ):
                    raise ValueError(
                        "original 15-control reception must not be inferred from switch"
                    )
                outcomes[name] = value
            t = outcomes["candidate"]["transition_control"]
            r = outcomes["candidate"]["original_receiving_ready_control"]
            early = t is not None and (r is None or t < r)
            rows.append(
                dict(
                    block=block,
                    context=key[0],
                    side=key[1],
                    historical_baseline_identical=same,
                    candidate_early_transition=early,
                    **outcomes,
                )
            )
        frozen.extend(
            (
                (root / "complete.json", sh),
                (audit, ah),
                (old / "complete.json", ph),
                (old_audit, oh),
                (scene, scene_hash),
            )
        )
    if seen != {0, 1, 2} or len(rows) != 24 or any(digest(path) != value for path, value in frozen):
        raise ValueError("all 24 unchanged independently replayed pairs required")
    summary = {}
    for name in (
        "original_handoff",
        "strict_receiver_clean_handoff",
        "strict_both_clean_handoff",
        "both_bodies_healthy",
    ):
        pairs = [(r["baseline"][name], r["candidate"][name]) for r in rows]
        summary[name] = dict(
            baseline=sum(a for a, _ in pairs),
            candidate=sum(b for _, b in pairs),
            rescues=sum(not a and b for a, b in pairs),
            losses=sum(a and not b for a, b in pairs),
        )
    summary["early_transitions"] = sum(r["candidate_early_transition"] for r in rows)
    summary["precise_old_success_losses"] = sum(
        r["baseline"]["original"]["return_0p1m_and_healthy"]
        and not r["candidate"]["original"]["return_0p1m_and_healthy"]
        for r in rows
    )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="PRECONTACT_FULLBODY_PAIR_REPLAY_REVIEWED"
            if all(r["historical_baseline_identical"] for r in rows)
            else "PRECONTACT_FULLBODY_BASELINE_MISMATCH",
            source_head=a.expected_head,
            producer_head=a.source_head,
            conditions=24,
            summary=summary,
            records=rows,
            early_transition_is_not_reception=True,
            teacher_not_learned=True,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
