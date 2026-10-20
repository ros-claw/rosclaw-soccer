"""Review all nine forecasts and both actually executed endpoints, no NN replay."""


def admit_closed(unit, root, expected_head):
    import json
    import subprocess
    from dataclasses import asdict

    from rosclaw_soccer.training.expanded_entry_courses import expanded_entry_courses
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_native_context_recovery_common import digest

    terminal = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "-p",
                "MainPID",
                "-p",
                "SubState",
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
            ],
            text=True,
        ).splitlines()
    )
    if terminal != dict(MainPID="0", SubState="exited", Result="success", ExecMainStatus="0"):
        raise ValueError("producer must actually be closed; never restart it")
    declaration = json.loads((root / "declaration.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    bank = expanded_entry_courses()
    bank_hash = preflight_receiving_courses(bank)
    shard = complete["shard"]
    if (
        declaration["schema"] != "soccer.rsi.bilateral_expanded40.v1"
        or complete["source_head"] != expected_head
        or declaration["source_head"] != expected_head
        or shard not in range(4)
        or declaration["shard"] != shard
        or declaration["bank_hash"] != bank_hash
        or complete["bank_hash"] != bank_hash
        or declaration["courses"] != [asdict(c) | dict(course=list(c.course)) for c in bank]
        or len(complete["results"]) != 10
        or complete["neural_optimizer_updates"] != 0
        or any(root.rglob("failure.json"))
    ):
        raise ValueError("complete fixed ten-course expanded shard required")
    expected = [c for c in bank if c.course_index % 4 == shard]
    pins = dict(declaration["input_file_hashes"])
    for item, row in zip(expected, complete["results"], strict=True):
        folder = root / f"course{item.course_index:02d}"
        if row != json.loads((folder / "complete.json").read_text()) or row["course"] != asdict(
            item
        ) | dict(course=list(item.course)):
            raise ValueError("exact ordered original40 condition required")
        if (
            len(row["candidates"]) != 9
            or row["forecast_native_steps"] != 25200
            or row["endpoint_native_steps"] != (3000 if row["shared_endpoint"] else 5800)
            or row["neural_optimizer_updates"] != 0
        ):
            raise ValueError("exact candidate/shared endpoint accounting required")
        for receipt in row["candidates"] + [row["baseline"], row["selected"]]:
            p = (folder / receipt["artifact"]).resolve()
            if not p.is_relative_to(folder.resolve()) or digest(p) != receipt["artifact_hash"]:
                raise ValueError("expanded raw artifact identity differs")
    if complete["forecast_native_steps"] != 252000 or complete["endpoint_native_steps"] != sum(
        r["endpoint_native_steps"] for r in complete["results"]
    ):
        raise ValueError("whole shard actual/forecast sum differs")
    pins.update({str(p): digest(p) for p in root.rglob("*") if p.is_file()})
    return declaration, complete, pins


def replay_case(model, folder, row, *, poses, affordance, evidence_hash, world_hash, progress=None):
    import json

    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.physics.sonic_navigation_forecast import _clone_body
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.rsi.bilateral_entry import (
        choose_bilateral_candidate,
        contact_foot_context,
        observed_contact_foot,
    )
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.full_entry_forecast import feedback_entry_laws, score_full_entry
    from rosclaw_soccer.rsi.full_task_branch import integration_hash
    from rosclaw_soccer.rsi.recorded_entry_review import (
        replay_prefix,
        replay_suffix,
        validate_recording,
    )
    from scripts.rsi_native_context_recovery_common import digest

    def load(path, frames, expected=None):
        if expected is not None and digest(path) != expected:
            raise ValueError("raw case digest differs")
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        validate_recording(data, frames=frames, nbody=model.nbody)
        return data

    # Verify complete shapes before this case's body allocation. Sources/world are pinned by caller.
    prefix = load(folder / "own-prefix20.npz", 20)
    futures = []
    for i, candidate in enumerate(row["candidates"]):
        if (
            candidate != json.loads((folder / f"candidate{i}.json").read_text())
            or candidate["index"] != i
        ):
            raise ValueError("ordered candidate receipt differs")
        futures.append(load(folder / candidate["artifact"], 280, candidate["artifact_hash"]))
    endpoints = {}
    for key in ("baseline", "selected"):
        receipt = row[key]
        endpoints[key] = load(folder / receipt["artifact"], 300, receipt["artifact_hash"])
        for k in prefix:
            np.testing.assert_array_equal(prefix[k], endpoints[key][k][:20])
    pd = diagnose_entry_prefix(model, prefix)
    if pd != row["prefix_diagnostics"]:
        raise ValueError("common prefix safety/attribution differs")

    def score(raw):
        value = score_full_entry(model, raw)
        value["original_suffix_qualified"] = value["qualified"]
        value["common_prefix_physical_safe"] = pd["prefix_physical_safe"]
        if not pd["prefix_physical_safe"]:
            value["qualified"], value["cost"] = False, None
            value["physical_prefix_and_suffix_safe"] = False
        return value

    kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
    body = G1TargetEpisode(
        model,
        world_hash=world_hash,
        kp=kp,
        kd=kd,
        course=tuple(row["course"]["course"]),
        defaults=G1SonicRunupController.default_angles.copy(),
        record_all_ball_contacts=True,
    )
    if progress is None:
        progress = {}
    progress.update(completed_steps=0, active_body=body, active_origin_steps=0)
    replay_prefix(body, prefix, affordance)
    progress.update(completed_steps=200, active_body=None)
    total = 200
    donor = integration_hash(model, body.data)
    laws = feedback_entry_laws()

    def replay(target, raw, candidate):
        source, entry, context = contact_foot_context(
            poses, affordance, candidate["contact_reference_foot"]
        )
        if context != candidate["contact_foot_context"]:
            raise ValueError("foot reference/entry provenance differs")
        progress.update(active_body=target, active_origin_steps=target.integrated_steps)
        replay_suffix(
            target,
            raw,
            law=laws[candidate["law_index"]],
            offset_root_m=tuple(candidate["offset_root_m"]),
            affordance=entry,
            source_poses=source,
            source_evidence_hash=evidence_hash,
            report=candidate["report"],
        )
        if target.review() != candidate["report"]["body"]:
            raise ValueError("full native body counters/review differ")
        progress.update(completed_steps=progress["completed_steps"] + 2800, active_body=None)

    for candidate, future in zip(row["candidates"], futures, strict=True):
        if candidate["donor_body_hash"] != donor:
            raise ValueError("future must originate at OWN actual prefix")
        branch = _clone_body(body)
        replay(branch, future, candidate)
        total += 2800
        if integration_hash(model, body.data) != donor:
            raise ValueError("native review changed donor")
        if (
            score(future) != candidate["score"]
            or observed_contact_foot(model, future, global_start_frame=20)
            != candidate["observed_contact_foot"]
        ):
            raise ValueError("candidate original score or physical contact side differs")
    selected = choose_bilateral_candidate(row["candidates"])
    eligible = [i for i, c in enumerate(row["candidates"][:6]) if c["score"]["qualified"]]
    baseline = (
        min(eligible, key=lambda i: (row["candidates"][i]["score"]["cost"], i)) if eligible else 0
    )
    if (
        row["selected"]["selected_index"] != selected
        or row["baseline"]["selected_index"] != baseline
        or row["shared_endpoint"] != (baseline == selected)
    ):
        raise ValueError("same gate/cost/shared endpoint choice differs")
    if baseline == selected and row["baseline"] != row["selected"]:
        raise ValueError("shared endpoint cannot invent second execution")
    order = ["selected"] if baseline == selected else ["baseline", "selected"]
    for key in order:
        index = baseline if key == "baseline" else selected
        receipt = row[key]
        raw = endpoints[key]
        suffix = {k: v[20:] for k, v in raw.items()}
        actual = body if key == "selected" else _clone_body(body)
        replay(actual, suffix, row["candidates"][index])
        total += 2800
        for k in suffix:
            np.testing.assert_array_equal(suffix[k], futures[index][k])
        if (
            score(suffix) != receipt["score"]
            or receipt["report"] != row["candidates"][index]["report"]
            or receipt["intended_reference_foot"]
            != row["candidates"][index]["contact_reference_foot"]
            or receipt["observed_contact_foot"]
            != observed_contact_foot(model, raw, global_start_frame=0)
        ):
            raise ValueError("actual native endpoint/reference/score/foot differs")
        if key == "baseline" and integration_hash(model, body.data) != donor:
            raise ValueError("baseline clone mutated selected donor")
    old, new = row["baseline"]["score"], row["selected"]["score"]
    flags = dict(
        old_teacher_success_lost=old["qualified"] and not new["qualified"],
        old_teacher_failure_rescued=not old["qualified"] and new["qualified"],
        added_unsafe_vs_old_teacher=old["physical_prefix_and_suffix_safe"]
        and not new["physical_prefix_and_suffix_safe"],
        causal_first_touch_rescue=not old["qualified"]
        and new["qualified"]
        and pd["post_boundary_control_can_cause_first_touch"],
    )
    if (
        any(row[k] != value for k, value in flags.items())
        or total != row["forecast_native_steps"] + row["endpoint_native_steps"]
    ):
        raise ValueError("full paired rescue/retention/native accounting differs")
    return dict(
        course=row["course"],
        selected_score=new,
        baseline_score=old,
        additional_native_steps=total,
        shared_endpoint=baseline == selected,
        candidate_count=9,
        all_forecast_and_actual_native_fields_bit_exact=True,
        reference_action_mapping_and_contact_side_verified=True,
        prefix_safety_and_first_touch_attribution_verified=True,
        neural_math_replayed=False,
        **flags,
    )
