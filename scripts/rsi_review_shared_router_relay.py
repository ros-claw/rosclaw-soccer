"""Review exact own-replayed paired router physical trajectories, SIM only."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiving_handoff_diagnosis import diagnose_no_handoff
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_shared_cftrack_relay import PHYSICAL_FIELDS, compare_prefix, outcome


def bind_world(root, row, name):
    item = row["artifacts"].get(name)
    if item is None:
        return None
    path = root / item["path"]
    if (
        path.is_symlink()
        or path.resolve().parent != (root / f"trial{row['index']}").resolve()
        or digest(path) != item["hash"]
    ):
        raise ValueError("exact confined world artifact required")
    return path


def verify_own_handoff(shadow, returned, *, handoff, controls, width):
    """Check recorded own recurrent continuity, independent of the producer."""
    if (
        type(handoff) is not int
        or type(controls) is not int
        or not 0 < handoff < controls
        or width not in (128, 256)
    ):
        raise ValueError("bounded actual handoff and known model width required")
    length = controls - handoff
    for prefix in ("h", "c"):
        before = returned["NN_" + prefix + "_before"]
        after = returned["NN_" + prefix + "_after"]
        shadow_after = shadow[prefix + "_after"]
        if (
            before.shape != (length, 2, 1, width)
            or after.shape != before.shape
            or shadow_after.shape != (handoff, 2, 1, width)
            or not np.array_equal(before[0], shadow_after[-1])
            or not np.array_equal(before[1:], after[:-1])
        ):
            raise ValueError("same player's shadow and return recurrent history must connect")
    obs, action = returned["NN_obs"], returned["NN_action"]
    if (
        obs.shape != (length, 160)
        or action.shape != (length, 29)
        or not np.array_equal(obs[1:, 125:154], action[:-1])
    ):
        raise ValueError("actual committed action must enter next return observation")


def verify_previous_target_feedback(shadow, returned, targets, *, handoff, offset, scale, order):
    """Independently decode each actual executed HG target into policy action units."""
    obs = shadow["obs"]
    if obs.ndim != 2 or obs.shape[1] != 160 or len(obs) != handoff or targets.shape[0] < handoff:
        raise ValueError("complete aligned shadow and applied targets required")
    if any(x.shape != (29,) for x in (offset, scale, order)) or (scale <= 0).any():
        raise ValueError("qualified original target codec required")
    actual = np.empty((handoff, 29), dtype=np.float32)
    actual[0] = 0
    actual[1:, order] = ((targets[: handoff - 1] - offset) / scale).astype(np.float32)
    if not np.array_equal(obs[:, 125:154], actual):
        raise ValueError("shadow observation does not contain its own last actual target")
    if returned is not None:
        first = np.empty(29, dtype=np.float32)
        first[order] = ((targets[handoff - 1] - offset) / scale).astype(np.float32)
        if not np.array_equal(returned["NN_obs"][0, 125:154], first):
            raise ValueError("handoff observation does not contain its own last actual target")


def optimistic_handoff_diagnosis(physics, layout):
    """Upper bound assuming all >1 N contact groups were eligible body parts."""
    diagnosis = diagnose_no_handoff(
        physics["qpos"],
        physics["qvel"],
        physics["ball_forces"],
        physics["body_failed"],
        receiver_q=layout.rq,
        receiver_v=layout.rv,
        ball_q=layout.bq,
        ball_v=layout.bv,
        excluded_contact=False,
    )
    # Force groups are not anatomical witnesses. Never claim the assumption is fact.
    diagnosis.pop("excluded_contact_ever")
    diagnosis.pop("predicate_reconstruction_exact")
    diagnosis.pop("no_handoff_reconstruction_consistent")
    return dict(
        assumption="optimistic_all_observed_contact_groups_are_eligible",
        actual_excluded_contact_unknown=True,
        upper_bound_insufficient_for_15_controls=diagnosis["best_consecutive_candidate_controls"]
        < 15,
        diagnostic=diagnosis,
    )


def admit(source, replay, source_hash, replay_hash):
    paths = (source / "complete.json", replay / "complete.json")
    if (
        any(p.is_symlink() for p in paths)
        or digest(paths[0]) != source_hash
        or digest(paths[1]) != replay_hash
        or (source / "failed.json").exists()
        or (replay / "failed.json").exists()
    ):
        raise ValueError("exact successful source and own replay required")
    old, new = [json.loads(p.read_text()) for p in paths]
    if (
        old["status"] != "ROUTER_SHARED_RELAY_SOURCE_PENDING_REPLAY"
        or new["status"] != "ROUTER_SHARED_RELAY_OWN_REPLAY_EXACT"
        or new["source_hash"] != source_hash
        or old["source_head"] != new["source_head"]
        or old["worlds"] != 16
        or new["worlds"] != 16
        or len(old["records"]) != 16
        or len(new["records"]) != 16
        or any(
            r.get(k) is not False
            for r in (old, new)
            for k in ("hardware_authorized", "promotion_authorized", "team_game_verified")
        )
    ):
        raise ValueError("matching unpromoted 16-world replay required")
    declarations = []
    for root, report in ((source, old), (replay, new)):
        path = root / "declaration.json"
        if path.is_symlink() or digest(path) != report["declaration_hash"]:
            raise ValueError("source-bound declarations required")
        declarations.append(json.loads(path.read_text()))
    if (
        declarations[0]["input_file_hashes"] != declarations[1]["input_file_hashes"]
        or declarations[0]["protocol_hash"] != declarations[1]["protocol_hash"]
    ):
        raise ValueError("source/replay input commitments differ")
    for i, (x, y) in enumerate(zip(old["records"], new["records"], strict=True)):
        if (
            x["index"] != i
            or y["index"] != i
            or {k: v for k, v in x.items() if k != "artifacts"}
            != {k: v for k, v in y.items() if k != "artifacts"}
            or set(x["artifacts"]) != set(y["artifacts"])
        ):
            raise ValueError("source/replay world metadata differs")
        for name in x["artifacts"]:
            xp, yp = bind_world(source, x, name), bind_world(replay, y, name)
            with np.load(xp, allow_pickle=False) as a, np.load(yp, allow_pickle=False) as b:
                if set(a.files) != set(b.files) or any(
                    not np.array_equal(a[k], b[k]) for k in a.files
                ):
                    raise ValueError("source/replay actual histories differ")
    return old, declarations[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "replay", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source-hash", "replay-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external review required")
    source, declaration = admit(a.source, a.replay, a.source_hash, a.replay_hash)
    pins = declaration["input_file_hashes"]
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1 or digest(scenes[0]) != pins[str(scenes[0])]:
        raise ValueError("exact source-bound standard body required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/", "b/"))
    )
    import onnx

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.paid_teacher_policy import strict_parameters

    policies = [
        Path(x) for x in pins if x.endswith("/paid-reference-source-20261021/ckp/policy_30000.onnx")
    ]
    if len(policies) != 1 or digest(policies[0]) != pins[str(policies[0])]:
        raise ValueError("exact source-bound policy target codec required")
    metadata = onnx.load(policies[0], load_external_data=False).metadata_props
    names, params = strict_parameters({x.key: x.value for x in metadata})
    order = np.array([names.index(name) for name in G1_DDS_JOINT_NAMES])
    offset, scale = params["default_joint_pos"][order], params["action_scale"][order]
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for i in range(0, 16, 2):
        base, candidate = source["records"][i : i + 2]
        if (
            base["arm"] != "dual"
            or candidate["arm"] != "router"
            or any(base[k] != candidate[k] for k in ("context", "lateral_m"))
        ):
            raise ValueError("fixed paired roles and contexts required")
        ba, ca = base["handoff"], candidate["handoff"]
        if (ba is None) != (ca is None) or (ba and ba["control"] != ca["control"]):
            raise ValueError("same causal readiness required")
        prefix = ba["control"] if ba else min(base["controls"], candidate["controls"])
        with (
            np.load(bind_world(a.source, base, "physics"), allow_pickle=False) as x,
            np.load(bind_world(a.source, candidate, "physics"), allow_pickle=False) as y,
        ):
            compare_prefix(x, y, PHYSICAL_FIELDS, prefix)
        common = min(base["controls"], candidate["controls"])
        if ba is not None:
            for item, width in ((base, 256), (candidate, 128)):
                with (
                    np.load(
                        bind_world(a.source, item, "receiver_shadow"), allow_pickle=False
                    ) as shadow,
                    np.load(
                        bind_world(a.source, item, "receiver_return_nn"), allow_pickle=False
                    ) as returned,
                ):
                    verify_own_handoff(
                        shadow,
                        returned,
                        handoff=ba["control"],
                        controls=item["controls"],
                        width=width,
                    )
        for item in (base, candidate):
            handoff = item["handoff"]["control"] if item["handoff"] else item["controls"]
            with (
                np.load(
                    bind_world(a.source, item, "receiver_shadow"), allow_pickle=False
                ) as shadow,
                np.load(bind_world(a.source, item, "physics"), allow_pickle=False) as physics,
            ):
                return_path = bind_world(a.source, item, "receiver_return_nn")
                if return_path is None:
                    verify_previous_target_feedback(
                        shadow,
                        None,
                        physics["target_b"],
                        handoff=handoff,
                        offset=offset,
                        scale=scale,
                        order=order,
                    )
                else:
                    with np.load(return_path, allow_pickle=False) as returned:
                        verify_previous_target_feedback(
                            shadow,
                            returned,
                            physics["target_b"],
                            handoff=handoff,
                            offset=offset,
                            scale=scale,
                            order=order,
                        )
        no_handoff_bound = None
        if ba is None and base["controls"] == 300:
            with np.load(bind_world(a.source, base, "physics"), allow_pickle=False) as physics:
                no_handoff_bound = optimistic_handoff_diagnosis(physics, layout)
        rows.append(
            dict(
                context=base["context"],
                lateral_m=base["lateral_m"],
                prefix_controls=prefix,
                base_termination=base["termination"],
                candidate_termination=candidate["termination"],
                base_full=outcome(a.source, base, base["controls"], model, layout),
                candidate_full=outcome(a.source, candidate, candidate["controls"], model, layout),
                base_common=outcome(
                    a.source, base, common, model, layout, proximity_start_control=prefix
                ),
                candidate_common=outcome(
                    a.source, candidate, common, model, layout, proximity_start_control=prefix
                ),
                candidate_neural_controls=candidate["controls"] - ba["control"] if ba else 0,
                no_handoff_optimistic_bound=no_handoff_bound,
            )
        )
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="PAIRED_ROUTER_REPLAYED_DEVELOPMENT_ONLY",
            source_hash=a.source_hash,
            replay_hash=a.replay_hash,
            code_head=a.expected_head,
            source_head=source["source_head"],
            protocol_hash=declaration["protocol_hash"],
            compiled_scene_hash=compiled_model_hash(model),
            rows=rows,
            hardware_authorized=False,
            promotion_authorized=False,
            team_game_verified=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
