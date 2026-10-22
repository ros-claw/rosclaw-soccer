"""Admission and raw-response derivation for the learned SIM teacher pilot."""

import json

import numpy as np

from scripts.rsi_native_context_recovery_common import digest


def admit(root, source_hash, audit_root, audit_hash):
    sp, ap = root / "complete.json", audit_root / "complete.json"
    for path, h in ((sp, source_hash), (ap, audit_hash)):
        if path.is_symlink() or (path.parent / "failed.json").exists() or digest(path) != h:
            raise ValueError("completed exact source and whole72 own-native proof required")
    s, own = [json.loads(p.read_text()) for p in (sp, ap)]
    if (
        s["status"] != "TARGETED_PASS_CEM72_COLLECTED_PENDING_OWN_REPLAY_AND_TEAM_TRANSFER"
        or own["status"] != "TARGETED_PASS_ALL72_OWN_NATIVE_AND_CEM_REPLAY_MATCH"
        or own["producer_hash"] != source_hash
        or own["context"] != s["context"]
        or len(s["records"]) != 72
        or len(own["records"]) != 72
        or own["counts"]["native"] != s["counts"]["native_steps"]
        or own["counts"]["neural"] != s["counts"]["actor_calls"]
        or any(
            r[k] is not False
            for r in (s, own)
            for k in ("goal_complete", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("whole72 unpromoted own-history closure required")
    pins = dict(own["input_file_hashes"])
    pins[str(ap)] = audit_hash
    loaded = []
    for name in ("declaration.json", "preflight.json"):
        path = root / name
        if path.is_symlink() or pins.get(str(path)) != digest(path):
            raise ValueError("own-reconstructed and hash-bound source metadata required")
        loaded.append(json.loads(path.read_text()))
    for claim, proof in zip(s["records"], own["records"], strict=True):
        if (claim["tag"], claim["task"], claim["passing_result"]) != (
            proof["tag"],
            proof["task"],
            proof["passing_result"],
        ):
            raise ValueError("all source outcomes must be independently reproduced")
        for art in claim["artifacts"].values():
            path = root / art["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (root / claim["tag"]).resolve()
                or pins.get(str(path)) != art["hash"]
                or digest(path) != art["hash"]
            ):
                raise ValueError("all exact own-replayed raw trajectories required")
    return s, loaded[0], loaded[1], pins


def response_from_blobs(blobs, result, goal):
    index = result["closest_goal_substep"]
    if index is None:
        return None
    count = len(blobs["kick"]["qpos"]) * 10
    if index < count:
        point = blobs["kick"]["qpos"][index // 10, index % 10, 36:39]
    else:
        point = blobs["stand"]["qpos"][(index - count) // 10, (index - count) % 10, 36:39]
    initial = blobs["kick"]["initial_qpos"][36:39]
    forward = goal[:2] - initial[:2]
    if np.linalg.norm(forward) < 0.5:
        raise ValueError("nondegenerate declared delivery ray required")
    forward = forward / np.linalg.norm(forward)
    lateral = np.array([-forward[1], forward[0]])
    delta = point - goal
    return np.array(
        [delta[:2] @ lateral, delta[:2] @ forward, delta[2], result["arrival_speed_mps"] - 2.8],
        np.float64,
    )


def plan_task(root, source, task):
    from rosclaw_soccer.rsi.pass_response_model import feasible_first_rank, response_proposals

    rows = [r for r in source["records"] if r["task"] == task]
    safe = []
    excluded = []
    unique = {}
    duplicates = []
    for row in rows:
        result = row["passing_result"]
        if not (
            result["whole_body_healthy"]
            and result["recovery_observed"]
            and result["clean_foot"]
            and result["closest_goal_substep"] is not None
        ):
            excluded.append(dict(tag=row["tag"], reason="UNSAFE_OR_NO_OBSERVED_FOOT_DELIVERY"))
            continue
        blobs = {}
        for key, art in row["artifacts"].items():
            with np.load(root / art["path"], allow_pickle=False) as z:
                blobs[key] = {"qpos": z["qpos"].copy(), "initial_qpos": z["initial_qpos"].copy()}
        response = response_from_blobs(blobs, result, np.array(row["target"]))
        latent = np.array(row["latent"], np.float64)
        key = latent.tobytes()
        if key in unique:
            if not np.array_equal(unique[key][2], response):
                raise ValueError("same physical input has contradictory repeated response")
            duplicates.append(row["tag"])
            continue
        unique[key] = (row, latent, response)
        safe.append(row)
    start = (
        max(safe, key=lambda r: (feasible_first_rank(r["passing_result"]), -rows.index(r)))
        if safe
        else rows[0]
    )
    if not safe:
        fit = dict(status="NO_SAFE_OBSERVED_DELIVERY", candidates=[])
        actual = None
    else:
        actual = unique[np.array(start["latent"], np.float64).tobytes()][2]
        fit = response_proposals(
            np.stack([v[1] for v in unique.values()]),
            np.stack([v[2] for v in unique.values()]),
            start=np.array(start["latent"]),
            start_response=actual,
        )
    return dict(
        task=task,
        start_tag=start["tag"],
        start_latent=start["latent"],
        actual_start_response=None if actual is None else actual.tolist(),
        source_unique_safe_tags=[v[0]["tag"] for v in unique.values()],
        duplicate_execution_tags=duplicates,
        excluded_from_regression=excluded,
        all_source_failures_preserved=True,
        fit=fit,
    )
