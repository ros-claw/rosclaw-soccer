"""Explicit model/physics provenance and old champion retention for SIM development."""

import json

from scripts.rsi_native_context_recovery_common import digest


def merge_pins(pins, incoming):
    for key, value in incoming.items():
        if key in pins and pins[key] != value:
            raise ValueError("conflicting stable/plastic provenance")
        pins[key] = value


def admit_development_gates(paths, hashes, model_hash, pins):
    """Both complete development banks must qualify BEFORE a new local exam."""
    if paths is None or hashes is None or len(paths) != 2 or len(hashes) != 2:
        raise ValueError("two independently audited development banks required")
    summaries = []
    for perturbed, path, expected in zip((False, True), paths, hashes, strict=True):
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact independent development audit required")
        report = json.loads(path.read_text())
        if (
            report.get("status") != "MOTION_SYNERGY_TEACHER_RAW_CEM_FUTURES_AND_FULL_EXAM_VERIFIED"
            or report.get("stable_plastic_development") is not True
            or report.get("expanded_public_validation") is not True
            or report.get("perturbed_public_validation", False) != perturbed
            or report.get("independent_local_validation", False)
            or report.get("new_physical_steps") != 0
            or report.get("actual_worlds") != 80
            or report.get("regressions") != []
            or report.get("champion_regressions") != []
            or len(report.get("results", [])) != 40
            or [r["course_index"] for r in report["results"]] != list(range(40))
            or sum(r["C"]["task_pass"] for r in report["results"]) < 32
            or not all(r["C_healthy"] for r in report["results"])
            or model_hash not in report["input_file_hashes"].values()
        ):
            raise ValueError(
                "complete same-model development32/40 with zero old-skill loss required"
            )
        merge_pins(pins, report["input_file_hashes"])
        pins[str(path)] = expected
        summaries.append(
            dict(
                perturbed=perturbed,
                audit_hash=expected,
                champion_rescues=report["champion_rescues"],
            )
        )
    if not any(r["champion_rescues"] for r in summaries):
        raise ValueError("a real independently verified new motor gain required before new exam")
    return summaries


def admit_stable_plastic_sources(a, pins):
    for path, expected, status in (
        (
            a.plastic_qualification,
            a.plastic_qualification_hash,
            "G1_FEEDBACK_COMPLEMENT_REFERENCE_NN_FUTURES_EXACT",
        ),
        (
            a.retention_proof,
            a.retention_proof_hash,
            "STABLE_PLASTIC_BACKBONE_RESCUES_FULL_ACTUAL_TRAJECTORIES_EXACT",
        ),
    ):
        if path is None or path.is_symlink() or digest(path) != expected:
            raise ValueError("exact independently qualified stable/plastic inputs required")
        report = json.loads(path.read_text())
        if report["status"] != status:
            raise ValueError("nonzero feedback and old complete physical retention required")
        merge_pins(pins, report["input_file_hashes"])
        pins[str(path)] = expected
    if a.plastic_model is None or pins.get(str(a.plastic_model)) != a.plastic_model_hash:
        raise ValueError("explicit bound feedback model required")
    if digest(a.plastic_model) != a.plastic_model_hash:
        raise ValueError("feedback model changed")
    fitted = json.loads(a.plastic_model.read_text())
    if (
        fitted["status"] != "ACTUAL_SUCCESSFUL_MOTOR_FEEDBACK_COMPLEMENT_FITTED"
        or fitted["backbone_model_hash"] != a.synergy_model_hash
        or not fitted["prior_perturbation_bank_is_now_consumed_development"]
    ):
        raise ValueError("frozen backbone and declared consumed development lineage required")
    artifact = a.plastic_model.parent / fitted["artifact"]["path"]
    if (
        not artifact.resolve().is_relative_to(a.plastic_model.parent.resolve())
        or artifact.is_symlink()
        or pins.get(str(artifact)) != fitted["artifact"]["hash"]
        or digest(artifact) != fitted["artifact"]["hash"]
    ):
        raise ValueError("confined explicitly bound complement artifact required")
    if a.champion_root is None:
        raise ValueError("explicit same-state old40 champion reference required")
    cp = a.champion_root / "complete.json"
    if pins.get(str(cp)) != digest(cp):
        raise ValueError("old champion not bound by qualified feedback lineage")
    champion = json.loads(cp.read_text())
    if (
        champion["interface"] != "motion-synergy-full-window"
        or not champion["expanded_public_validation"]
        or champion.get("perturbed_public_validation", False) != a.perturb_40
        or [r["course_index"] for r in champion["results"]] != list(range(40))
    ):
        raise ValueError("same complete development40 champion protocol required")


def compare_champion(root, champion_root, pins):
    """Never hide old skill loss behind improvement relative to a weaker parent."""
    import numpy as np

    cp = champion_root / "complete.json"
    if pins.get(str(cp)) != digest(cp):
        raise ValueError("bound old champion required")
    old = json.loads(cp.read_text())
    regressions, rescues = [], []
    for i, item in enumerate(old["results"]):
        path = champion_root / item["arms"]["C"]["record"]
        if (
            not path.resolve().is_relative_to(champion_root.resolve())
            or pins.get(str(path)) != item["arms"]["C"]["hash"]
            or digest(path) != item["arms"]["C"]["hash"]
        ):
            raise ValueError("bound old actual champion control record required")
        previous = json.loads(path.read_text())
        current = json.loads((root / f"course-{i:02d}" / "C.json").read_text())
        if any(
            current[k] != previous[k]
            for k in ("course", "parameters", "experimental_fast_approach", "entry_control")
        ):
            raise ValueError("champion comparison requires exactly the same actual course")
        if any(
            previous["outcome"][k] and not current["outcome"][k] for k in ("task_pass", "safe")
        ) or (previous["healthy_full_horizon"] and not current["healthy_full_horizon"]):
            regressions.append(i)
        if (
            not previous["outcome"]["task_pass"]
            and current["outcome"]["task_pass"]
            and current["healthy_full_horizon"]
        ):
            rescues.append(i)
        events = current.get("controller_events", [])
        # Every previously selected old plan must remain EXACT, not merely successful.
        if previous.get("changed_controls", 0) or not (
            events and events[0].get("selected_basis") == "plastic"
        ):
            prior_path = path.parent / previous["artifact"]["path"]
            if pins.get(str(prior_path)) != previous["artifact"]["hash"]:
                raise ValueError("bound old whole native trajectory required")
            if digest(prior_path) != previous["artifact"]["hash"]:
                raise ValueError("old native trajectory changed")
            with (
                np.load(prior_path, allow_pickle=False) as before,
                np.load(
                    root / f"course-{i:02d}" / current["artifact"]["path"], allow_pickle=False
                ) as after,
            ):
                if set(before.files) != set(after.files) or any(
                    not np.array_equal(before[k], after[k]) for k in before.files
                ):
                    raise ValueError("old passing branch was replaced or stable execution drifted")
    return regressions, rescues


def verify_complement_fit(model_path, pins, backbone):
    """Reconstruct every inclusion and actual paired data interval, not fit statistics alone."""
    from pathlib import Path

    import numpy as np

    from rosclaw_soccer.rsi.feedback_motion_complement import fit_feedback_complement
    from rosclaw_soccer.sim.contracts import hash_json

    report = json.loads(model_path.read_text())
    sources = {}
    for name in report["input_file_hashes"]:
        path = Path(name)
        if not name.endswith("/complete.json"):
            continue
        raw = json.loads(path.read_text())
        if (
            raw.get("status") == "BOUNDED_SURFACE_CONTACT_PILOT_COMPLETE"
            and raw.get("interface") == "motion-synergy-full-window"
            and raw.get("expanded_public_validation") is True
        ):
            kind = "perturbed" if raw.get("perturbed_public_validation", False) else "nominal"
            if kind in sources:
                raise ValueError("unique old expanded physical source required")
            sources[kind] = (path.parent, raw)
    if set(sources) != {"nominal", "perturbed"}:
        raise ValueError("both consumed physical source banks required")
    records, intervals, samples = [], [], []
    for kind in ("nominal", "perturbed"):
        root, producer = sources[kind]
        for i, result in enumerate(producer["results"]):
            rows = {}
            for arm in ("B", "C"):
                ref = result["arms"][arm]
                path = root / ref["record"]
                if (
                    not path.resolve().is_relative_to(root.resolve())
                    or pins.get(str(path)) != ref["hash"]
                    or digest(path) != ref["hash"]
                ):
                    raise ValueError("bound actual model source record required")
                rows[arm] = json.loads(path.read_text())
            b, c = rows["B"], rows["C"]
            included = bool(
                c.get("changed_controls", 0)
                and c["outcome"]["task_pass"]
                and c["healthy_full_horizon"]
            )
            records.append(
                dict(
                    kind=kind,
                    course_index=i,
                    included=included,
                    C_task_pass=c["outcome"]["task_pass"],
                    C_healthy=c["healthy_full_horizon"],
                    source_C_hash=result["arms"]["C"]["hash"],
                )
            )
            if not included:
                continue
            entry = c["entry_control"]
            end = min(entry + 48, c["outcome"]["first_foot_contact_frame"] + 10, 300)
            if entry != b["entry_control"] or end <= entry:
                raise ValueError("paired motor learning interval differs")
            targets = {}
            for arm, row in rows.items():
                path = root / result["arms"][arm]["record"]
                artifact = path.parent / row["artifact"]["path"]
                if pins.get(str(artifact)) != digest(artifact):
                    raise ValueError("exact bound physical learning arrays required")
                with np.load(artifact, allow_pickle=False) as raw:
                    targets[arm] = raw["joint_targets"][entry:end].copy()
            samples.append(targets["C"] - targets["B"])
            intervals.append(
                dict(
                    kind=kind, source_context=i, start=entry, end_exclusive=end, samples=end - entry
                )
            )
    basis, fit = fit_feedback_complement(np.concatenate(samples).astype(np.float64), backbone)
    if (
        hash_json(records) != hash_json(report["all_context_records"])
        or hash_json(intervals) != hash_json(report["selected_actual_intervals"])
        or hash_json(fit) != hash_json(report["fit"])
    ):
        raise ValueError("feedback fit differs from all actual inclusion/exclusion evidence")
    artifact = model_path.parent / report["artifact"]["path"]
    if (
        pins.get(str(artifact)) != report["artifact"]["hash"]
        or digest(artifact) != report["artifact"]["hash"]
    ):
        raise ValueError("bound output complement artifact required")
    with np.load(artifact, allow_pickle=False) as raw:
        if not np.array_equal(basis, raw["basis_hg"]) or not np.array_equal(
            backbone, raw["frozen_backbone_basis_hg"]
        ):
            raise ValueError("plastic data model or frozen backbone differs")
    return dict(
        actual_states=len(intervals),
        actual_control_samples=sum(r["samples"] for r in intervals),
        original_context_clusters=len({r["source_context"] for r in intervals}),
    )
