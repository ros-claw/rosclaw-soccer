"""Compare actual teammate-delivered states with an earlier legacy skill domain."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiving_legacy_proximity import legacy_proximity
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True)
    for name in ("review", "legacy-anchor", "legacy-qualified", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "review-hash",
        "legacy-anchor-hash",
        "legacy-qualified-hash",
        "source-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    pins = (
        (a.review, a.review_hash),
        (a.legacy_anchor, a.legacy_anchor_hash),
        (a.legacy_qualified, a.legacy_qualified_hash),
    )
    if (
        len(a.pair) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in pins)
    ):
        raise ValueError("exact prior physical study, legacy evidence and unused output required")
    review, legacy, qualified = (json.loads(p.read_text()) for p, _ in pins)
    if (
        review.get("status") != "RECEIVER_LEARNED_PAIR_PHYSICAL_REPLAY_REVIEWED"
        or review.get("producer_head") != a.source_head
        or review.get("conditions") != 24
        or qualified.get("status") != "FRESH_VELOCITY_SAFE_NEARSIDE_QUALIFIED_FOR_NEXT_CHAIN_GATE"
        or qualified.get("v237_report_hash") != legacy.get("report_hash")
        or qualified.get("activation_radius") != 0.017
        or qualified.get("minimum_relative_vx_feature") != -0.48
        or any(qualified.get(k) is not False for k in ("promotion_authorized", "video_authorized"))
    ):
        raise ValueError("unpromoted, correctly bound legacy and paired evidence required")
    anchors = np.asarray(legacy["activation_states"], dtype=np.float64)
    expected = {(x["context"], x["side"]): x for x in review["records"]}
    rows, frozen, blocks = [], [], set()
    for root_name, audit_name in a.pair:
        root, audit = Path(root_name), Path(audit_name)
        ah = digest(audit)
        source, _, sh = report(
            root, audit, ah, "NEURAL_RELAY_RECEIVER_LEARNED_PAIR16_PENDING_OWN_REPLAY", worlds=16
        )
        dec = json.loads((root / "declaration.json").read_text())
        block = dec["development_context_block"]
        if (
            block in blocks
            or block not in (0, 1, 2)
            or source["source_head"] != a.source_head
            or digest(root / "declaration.json") != source["declaration_hash"]
        ):
            raise ValueError("exact independently replayed source block required")
        blocks.add(block)
        scene = root / "scene.xml"
        if scene.is_symlink():
            raise ValueError("owned source scene required")
        scene_hash = digest(scene)
        model = mujoco.MjModel.from_xml_path(str(scene))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        for row in source["records"]:
            if row["course"]["arm"]["receiver_learned"]:
                continue
            key = (row["course"]["context"], row["course"]["lateral_m"])
            if (
                key not in expected
                or expected[key]["block"] != block
                or expected[key]["historical_baseline_identical"] is not True
                or measured(root, row) != expected[key]["baseline"]
            ):
                raise ValueError("actual source outcome must match reviewed historical baseline")
            initial, raw, velocity = (
                bind(root, row, k) for k in ("initial", "physics", "receiver_velocity")
            )
            if initial is None or raw is None or velocity is None:
                raise ValueError("exact original actual histories required")
            with (
                np.load(initial, allow_pickle=False) as i,
                np.load(raw, allow_pickle=False) as f,
                np.load(velocity, allow_pickle=False) as v,
            ):
                n = len(v["control"])
                if not 1 <= n <= len(f["qpos"]) or not np.array_equal(v["control"], np.arange(n)):
                    raise ValueError("continuous actual receiver controls required")
                q = np.concatenate((i["qpos"][None], f["qpos"][: n - 1, -1]), axis=0)
                dq = np.concatenate((i["qvel"][None], f["qvel"][: n - 1, -1]), axis=0)
                cq = np.concatenate(
                    (
                        q[:, layout.rq : layout.rq + 7],
                        q[:, layout.qi],
                        q[:, layout.bq : layout.bq + 7],
                    ),
                    axis=1,
                )
                cv = np.concatenate(
                    (
                        dq[:, layout.rv : layout.rv + 6],
                        dq[:, layout.vi],
                        dq[:, layout.bv : layout.bv + 6],
                    ),
                    axis=1,
                )
                result = legacy_proximity(cq, cv, v["contact_flags_before"], anchors)
            rows.append(dict(context=key[0], side=key[1], termination=row["termination"], **result))
        frozen.extend(((root / "complete.json", sh), (audit, ah), (scene, scene_hash)))
    if (
        blocks != {0, 1, 2}
        or len(rows) != 24
        or {(x["context"], x["side"]) for x in rows} != set(expected)
        or any(digest(path) != h for path, h in (*pins, *frozen))
    ):
        raise ValueError("all exact unchanged physical and legacy sources required")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="LEGACY_RECEIVE_PROXIMITY_DIAGNOSIS_NOT_TRANSFER",
            source_head=a.expected_head,
            review_hash=a.review_hash,
            legacy_anchor_hash=a.legacy_anchor_hash,
            legacy_qualified_hash=a.legacy_qualified_hash,
            conditions=len(rows),
            records=rows,
            uses_hypothetical_phase=True,
            different_foundation_and_world=True,
            independent_new_physics=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
