"""Exact catalog2 six-course controls for a material-point mechanism ablation."""

import json
from pathlib import Path

from rosclaw_soccer.rsi.receding_scope_evidence import authenticate_receipt
from rosclaw_soccer.rsi.receding_teacher_scope import SOURCE_GROUPS, receding_jobs
from rosclaw_soccer.rsi.surface_context_expansion import require_previous_surface
from scripts.rsi_stage_receding_contact_replay import require_receding_endpoints


def surface_receding_jobs():
    old = {j["group"]: j for j in receding_jobs() if j["role"] == "teacher"}
    return [dict(old[g], job=f"surface-receding-g{g}") for g in SOURCE_GROUPS]


def load_baselines(cfg, source, sealed, digest, pins):
    """Require whole completed study/dense replay; never cherry-pick C6/C7.

    Full failures justify this new physical intervention, not a dataset. Prior
    material-point default-OFF proof is reused only with identical control code.
    """
    previous = cfg["receding_predecessor"]
    config_path = Path(previous["configuration"])
    old = json.loads(config_path.read_text())
    root = Path(old["output_root"])
    declaration = sealed(root / "declaration.json")
    complete = sealed(root / "complete.json")
    audit = sealed(Path(previous["readback"]))
    diagnostic = sealed(Path(previous["diagnostic_complete"]))
    metrics = sealed(Path(previous["metrics"]))
    reader = Path(previous["reader_source"])
    if (
        digest(config_path) != previous["configuration_file_hash"]
        or complete["report_hash"] != previous["complete_hash"]
        or audit["report_hash"] != previous["readback_hash"]
        or audit.get("reader_source_hash") != digest(reader)
        or diagnostic["report_hash"] != previous["diagnostic_complete_hash"]
        or diagnostic.get("schema") != "soccer.rsi.receding_contact_diagnostic_complete.v1"
        or diagnostic.get("upstream_readback_hash") != audit["report_hash"]
        or diagnostic.get("new_fixed_control_replay_substeps") != 30000
        or diagnostic.get("metrics_hash") != metrics["report_hash"]
        or metrics["report_hash"] != previous["metrics_hash"]
        or metrics.get("schema") != "soccer.rsi.contact_exit_metrics_readback.v1"
        or len(metrics.get("rows", [])) != 22
        or old.get("mesh_task_contact_world_hash") != cfg["mesh_task_contact_world_hash"]
        or old.get("portfolio_fit_hash") != cfg["portfolio_fit_hash"]
    ):
        raise ValueError("whole receding ten-body terminal and dense audit required")
    require_receding_endpoints(old, declaration, complete, audit)
    pins[str(config_path)], pins[str(reader)] = digest(config_path), digest(reader)
    inputs = {
        **declaration["prepared_pins"],
        **declaration["input_file_hashes"],
        **metrics["input_file_hashes"],
    }
    for name, h in inputs.items():
        if digest(Path(name)) != h:
            raise ValueError("whole upstream source/witness bytes changed")
        pins[name] = h
    teachers = {r["group"]: r for r in complete["results"] if r["role"] == "teacher"}
    controls = {r["group"]: r for r in complete["results"] if r["role"] == "control"}
    controls.update({r["receipt"]["group"]: r["receipt"] for r in complete["reused_endpoints"]})
    if set(teachers) != set(SOURCE_GROUPS) or set(controls) != set(SOURCE_GROUPS):
        raise ValueError("all six original teacher and None outcomes required")
    for receipt in (*teachers.values(), *controls.values()):
        authenticate_receipt(
            receipt,
            world_hash=cfg["mesh_task_contact_world_hash"],
            sealed=sealed,
            digest=digest,
            pins=pins,
        )
    for group in SOURCE_GROUPS[:2]:
        matches = [
            r
            for r in metrics["rows"]
            if r["source_receipt_hash"] == teachers[group]["receipt_hash"]
        ]
        if (
            len(matches) != 1
            or matches[0]["original_outcome"] != teachers[group]["outcome"]
            or teachers[group]["outcome"]["task_pass"] is not True
            or matches[0]["metrics"]["first_nonfoot_robot_contact_substep"] is not None
        ):
            raise ValueError("both retained C6/C7 successes require clean whole-body dense proof")
    # The upstream actual six bodies are the disabled controls for THIS exact
    # catalog2. The older independent full OFF proof additionally authenticates
    # the opt-in material-point implementation's unchanged source arithmetic.
    require_previous_surface(cfg, source, sealed, digest, pins)
    for relative in (
        "rsi/native_first_touch_episode.py",
        "rsi/first_touch_campaign_worker.py",
        "rsi/cpu_motor_transfer_evidence.py",
        "rsi/mesh_surface_swing_probe.py",
        "rsi/mesh_surface_swing_evidence.py",
        "rsi/mesh_surface_swing.py",
        "physics/convex_surface_distance.py",
    ):
        paths = [
            Path(project) / "src/rosclaw_soccer" / relative
            for project in (old["source_root"], source)
        ]
        if digest(paths[0]) != digest(paths[1]):
            raise ValueError("this mechanism ablation cannot change existing control arithmetic")
        pins.update({str(path): digest(path) for path in paths})
    for item, job in zip(cfg["baseline_receipts"], surface_receding_jobs(), strict=True):
        path = Path(item["path"])
        receipt = teachers[job["group"]]
        if (
            item["pair_id"] != job["pair_id"]
            or json.loads(path.read_text()) != receipt
            or digest(path) != item["file_hash"]
            or item["receipt_hash"] != receipt["receipt_hash"]
            or receipt["option_index"] != 2
        ):
            raise ValueError("all six exact catalog2 baselines required, including unsafe outcomes")
        pins[str(path)] = digest(path)
    return (
        {f"receding-g{g}": teachers[g] for g in SOURCE_GROUPS},
        {f"receding-g{g}": controls[g] for g in SOURCE_GROUPS},
    )
