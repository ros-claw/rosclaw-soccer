"""Fit existing local response models to ALL8 own-verified two-body TRAIN tasks.

No motor weights, physics, Fresh access or task regrading. Keep every unsafe/
failed trajectory; only safe actual observed delivery rows enter regression.
Predictions require a NEW bounded native simulation plus independent replay.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("pilot-root", "shards-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("pilot-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument(
        "--anchor-rule",
        choices=("source-selection", "joint-normalized"),
        default="source-selection",
    )
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_self_teacher_search_common import calibrations, rank, verify_search

    pp = a.pilot_root / "complete.json"
    if (
        a.output.exists()
        or pp.is_symlink()
        or digest(pp) != a.pilot_hash
        or (a.pilot_root / "failed.json").exists()
    ):
        raise ValueError("unused fit output and exact whole pilot closure required")
    pilot = json.loads(pp.read_text())
    if (
        pilot["status"] != "SELF_TEACHER_CEM_FOUR_STATE_PILOT_CLOSED_NOT_NEURAL_TRAINING"
        or pilot["distinct_initial_states"] != 4
        or pilot["goal_tasks"] != 8
        or len(pilot["proofs"]) != 4
        or pilot["optimizer_updates"] != 0
        or pilot["hardware_authorized"] is not False
        or pilot["promotion_authorized"] is not False
    ):
        raise ValueError("whole unpromoted actual teacher pilot required")
    pins = {str(pp): a.pilot_hash}
    sources = []
    for proof, state in zip(pilot["proofs"], (20, 23, 26, 29), strict=True):
        root = a.shards_root / f"state{state}" / "source"
        ownroot = root.parent / "own"
        paths = [
            root / "complete.json",
            ownroot / "complete.json",
            root / "declaration.json",
            root / "scene.xml",
        ]
        if (
            proof["state_id"] != state
            or [digest(x) for x in paths[:2]] != [proof["source_hash"], proof["own_hash"]]
            or any(x.is_symlink() for x in paths)
            or any((r / "failed.json").exists() for r in (root, ownroot))
        ):
            raise ValueError("all4 exact source/own proofs required")
        s, o, d = [json.loads(x.read_text()) for x in paths[:3]]
        if (
            s["status"] != "WHOLE_LEARNED_SELF_TEACHER_SEARCH_COLLECTED_PENDING_ALL68_OWN"
            or o["status"] != "WHOLE_LEARNED_SELF_TEACHER_ALL68_OWN_NATIVE_MATCH"
            or o["producer_hash"] != proof["source_hash"]
            or o["counts"] != s["counts"]
            or s["declaration_hash"] != digest(paths[2])
            or s["scene_hash"] != digest(paths[3])
            or o["records"]
            != [{k: v for k, v in r.items() if k != "artifacts"} for r in s["records"]]
            or o["all_CEM_decisions_reconstructed"] is not True
        ):
            raise ValueError("complete actual own-state source replay required")
        calroot = Path(d["calibration_root"])
        cp = calroot / "complete.json"
        if digest(cp) != d["calibration_hash"]:
            raise ValueError("exact inherited calibration required")
        cal = json.loads(cp.read_text())
        verify_search(s, cal)
        pins.update({str(x): digest(x) for x in [*paths, cp]})
        sources.append((state, root, s, calroot, cal))
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.pass_response_model import response_proposals
    from rosclaw_soccer.rsi.two_body_pass_response import (
        measured_response,
        new_distinct_proposals,
        refinement_rank,
    )

    plans = []
    for state, root, s, calroot, cal in sources:
        model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
        body = model.body("soccer_ball").id
        joint = int(model.body_jntadr[body])
        qi = int(model.jnt_qposadr[joint])
        vi = int(model.jnt_dofadr[joint])
        geoms = np.flatnonzero(model.geom_bodyid == body)
        if (
            model.body_jntnum[body] != 1
            or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
            or len(geoms) != 1
            or model.geom_type[geoms[0]] != mujoco.mjtGeom.mjGEOM_SPHERE
            or abs(model.geom_size[geoms[0], 0] - 0.11) > 1e-12
            or abs(model.body_mass[body] - 0.43) > 1e-12
        ):
            raise ValueError("same actual compiled free standard ball required")
        for task in (0, 1):
            allrows = [
                (calroot, f"cal-t{task}-p{r['student_prefix_controls']}", r)
                for r in calibrations(cal, state, task)
            ] + [(root, r["tag"], r) for r in s["records"] if r["search_task"] == task]
            selected = s["selections"][task]["selected"]
            unique = {}
            excluded = []
            duplicates = []
            executed = []
            for folder, tag, row in allrows:
                z = np.array(row["course"]["latent"], np.float64)
                executed.append(z)
                o = row["outcome"]
                if not (
                    o["whole_body_healthy"]
                    and o["passer_full100NNrecovery"]
                    and o["passer_clean_foot"]
                    and not o["B_excluded_contact"]
                    and o["incoming"]["incoming_error_observed"]
                ):
                    excluded.append(dict(tag=tag, reason="UNSAFE_OR_NO_OBSERVED_DELIVERY"))
                    continue
                art = {}
                for name in ("initial", "physical"):
                    entry = row["artifacts"][name]
                    path = folder / entry["path"]
                    if (
                        path.is_symlink()
                        or not path.resolve().is_relative_to(folder.resolve())
                        or digest(path) != entry["hash"]
                    ):
                        raise ValueError("confined exact actual ball primitives required")
                    pins[str(path)] = entry["hash"]
                    art[name] = path
                with np.load(art["initial"], allow_pickle=False) as x:
                    initial = x["qpos"][qi : qi + 3].copy()
                with np.load(art["physical"], allow_pickle=False) as x:
                    if x["qpos"].shape[1:] != (10, model.nq) or x["qvel"].shape[1:] != (
                        10,
                        model.nv,
                    ):
                        raise ValueError("actual native2ms layout required")
                    pos = x["qpos"].reshape(-1, model.nq)[:, qi : qi + 3].copy()
                    vel = x["qvel"].reshape(-1, model.nv)[:, vi : vi + 3].copy()
                response = measured_response(
                    pos, vel, initial, np.array(row["course"]["target"]), o
                )
                if response is None:
                    raise ValueError("safe admitted delivery lost actual measurement")
                z[z == 0] = 0.0
                key = z.tobytes()
                if key in unique:
                    if not np.array_equal(unique[key]["response"], response):
                        raise ValueError(
                            "identical actual control has contradictory repeated response"
                        )
                    duplicates.append(tag)
                    continue
                unique[key] = dict(tag=tag, row=row, latent=z, response=response)
            start = next((v for v in unique.values() if v["tag"] == selected), None)
            if a.anchor_rule == "joint-normalized" and unique:
                start = max(unique.values(), key=lambda v: refinement_rank(v["row"]["outcome"]))
            if start is None and unique:
                start = max(unique.values(), key=lambda v: rank(v["row"]))
            if start is None:
                fit = dict(status="NO_SAFE_ACTUAL_DATA", candidates=[])
            else:
                fit = response_proposals(
                    np.stack([v["latent"] for v in unique.values()]),
                    np.stack([v["response"] for v in unique.values()]),
                    start=start["latent"],
                    start_response=start["response"],
                )
            fresh, duplicate_proposals = new_distinct_proposals(fit["candidates"], executed)
            plans.append(
                dict(
                    state_id=state,
                    task=task,
                    source_root=str(root),
                    source_hash=digest(root / "complete.json"),
                    source_own_hash=next(
                        p["own_hash"] for p in pilot["proofs"] if p["state_id"] == state
                    ),
                    start_tag=None if start is None else start["tag"],
                    start_latent=None if start is None else start["latent"].tolist(),
                    actual_start_response=None if start is None else start["response"].tolist(),
                    source_unique_safe_tags=[v["tag"] for v in unique.values()],
                    excluded_from_regression=excluded,
                    duplicate_execution_tags=duplicates,
                    fit=fit,
                    new_distinct_candidates=fresh,
                    duplicate_model_proposals=duplicate_proposals,
                    source_failures_removed=0,
                )
            )
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="TWO_BODY_SELF_TEACHER_RESPONSE_MODELS_FIT_NOT_PHYSICALLY_VALIDATED",
            fit_protocol=(
                "soccer.two_body_response_fit.v2"
                if a.anchor_rule == "joint-normalized"
                else "soccer.two_body_response_fit.v1"
            ),
            anchor_rule=a.anchor_rule,
            source_head=a.expected_head,
            input_file_hashes=pins,
            pilot_hash=a.pilot_hash,
            plans=plans,
            models_fit=sum(len(p["fit"].get("models", [])) for p in plans),
            new_control_proposals=sum(len(p["new_distinct_candidates"]) for p in plans),
            physical_steps=0,
            student_optimizer_updates=0,
            Fresh_cases_read=0,
            predictions_are_physical_evidence=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    print(
        json.dumps(
            [
                dict(
                    state_id=p["state_id"],
                    task=p["task"],
                    status=p["fit"]["status"],
                    safe_unique=len(p["source_unique_safe_tags"]),
                    new_candidates=len(p["new_distinct_candidates"]),
                )
                for p in plans
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
