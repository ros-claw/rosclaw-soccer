"""Fixed public cross-state screen scope, not a new control law or teacher gate."""

import json
from pathlib import Path


def context_jobs(contexts):
    if len(contexts) != 8 or [c.context_index for c in contexts] != list(range(8)):
        raise ValueError("whole eight original public contexts required")
    return [
        dict(
            job=f"surface-context-{c.context_index}",
            role="teacher",
            pair_id=f"context-{c.context_index}",
            context_index=c.context_index,
            group=c.source_group,
            option_index=4,
        )
        for c in contexts[2:]
    ]


def require_previous_surface(cfg, source, sealed, digest, pins):
    """Reuse completed OFF qualification only with byte-identical control sources.

    A successful evidence audit is not a teacher qualification. The prior seven
    outcomes are retained, including regressions, and are never rerun here.
    """
    previous = cfg["surface_predecessor"]
    config_path = Path(previous["configuration"])
    old_cfg = json.loads(config_path.read_text())
    complete = sealed(Path(old_cfg["output_root"]) / "complete.json")
    declaration = sealed(Path(old_cfg["output_root"]) / "declaration.json")
    audit = sealed(Path(previous["readback"]))
    reader = Path(previous["reader_source"])
    old_source = Path(old_cfg["source_root"])
    if (
        digest(config_path) != previous["configuration_file_hash"]
        or complete["report_hash"] != previous["complete_hash"]
        or complete["declaration_hash"] != declaration["report_hash"]
        or audit["report_hash"] != previous["readback_hash"]
        or audit.get("complete_hash") != complete["report_hash"]
        or audit.get("schema") != "soccer.rsi.mesh_surface_swing_terminal_readback.v1"
        or audit.get("reader_source_hash") != digest(reader)
        or audit.get("source_native_bodies") != 7
        or len(complete.get("results", [])) != 7
        or old_cfg.get("surface_profile", "original-seven") != "original-seven"
        or old_cfg.get("mesh_task_contact_world_hash") != cfg["mesh_task_contact_world_hash"]
        or old_cfg.get("portfolio_fit_hash") != cfg["portfolio_fit_hash"]
        or any(
            audit.get(k) is not True
            for k in (
                "whole_input_and_witness_files_verified",
                "shared_payload_logical_seals_rebuilt",
                "prior_independent_replays_authenticated",
                "all_original_pre_activation_prefixes_bit_equal",
                "disabled_complete_300_frame_noninterference_verified",
            )
        )
        or any(declaration.get(k) != v for k, v in old_cfg.items())
    ):
        raise ValueError("complete original seven-body audit with real default-OFF parity required")
    pins[str(config_path)], pins[str(reader)] = digest(config_path), digest(reader)
    for path in (
        Path(old_cfg["output_root"]) / "complete.json",
        Path(old_cfg["output_root"]) / "declaration.json",
        Path(previous["readback"]),
    ):
        pins[str(path)] = digest(path)
    for relative in (
        "rsi/native_first_touch_episode.py",
        "rsi/first_touch_campaign_worker.py",
        "rsi/cpu_motor_transfer_evidence.py",
        "rsi/mesh_surface_swing_probe.py",
        "rsi/mesh_surface_swing_evidence.py",
        "rsi/mesh_surface_swing.py",
        "physics/convex_surface_distance.py",
    ):
        old, new = [Path(root) / "src/rosclaw_soccer" / relative for root in (old_source, source)]
        if digest(old) != digest(new):
            raise ValueError("cross-state screen may not change prior qualified control arithmetic")
        pins[str(old)], pins[str(new)] = digest(old), digest(new)
    return audit["report_hash"]


def require_context_baselines(cfg, jobs, sealed, digest, pins):
    """Authenticate all twelve reused endpoints before any new model allocation."""
    path = Path(cfg["baseline_receipts"][0]["path"]).parent / "complete.json"
    complete = sealed(path)
    audit = sealed(Path(cfg["baseline_terminal_readback"]))
    reader = Path(cfg["baseline_reader_source"])
    if (
        audit["report_hash"] != cfg["baseline_terminal_readback_hash"]
        or audit.get("complete_hash") != complete["report_hash"]
        or audit.get("schema") != "soccer.rsi.mesh_option4_teacher_screen_terminal_readback.v1"
        or audit.get("reader_source_hash") != digest(reader)
        or audit.get("source_native_bodies") != 18
        or len(complete.get("results", [])) != 18
        or len(jobs) != 6
        or [j["context_index"] for j in jobs] != list(range(2, 8))
        or any(
            audit.get(k) is not True
            for k in (
                "whole_input_and_witness_files_verified",
                "shared_payload_logical_seals_rebuilt",
                "prior_independent_replays_authenticated",
            )
        )
    ):
        raise ValueError(
            "whole original 18-body terminal baseline study required before allocation"
        )
    for job, item in zip(jobs, cfg["baseline_receipts"], strict=True):
        for offset, role in ((0, "control"), (1, "teacher")):
            prior = complete["results"][2 * job["context_index"] + offset]
            if (
                prior["group"] != job["group"]
                or prior["job"] != role + "-" + job["pair_id"]
                or not any(
                    row["receipt_hash"] == prior["receipt_hash"]
                    and row["job"] == prior["job"]
                    and row["outcome"] == prior["outcome"]
                    for row in audit["rows"]
                )
                or (
                    offset == 1
                    and (
                        item["pair_id"] != job["pair_id"]
                        or item["receipt_hash"] != prior["receipt_hash"]
                        or item["file_hash"] != digest(Path(item["path"]))
                        or sealed(Path(item["path"]), "receipt_hash") != prior
                    )
                )
            ):
                raise ValueError(
                    "all original None and option4 endpoints must belong to baseline audit"
                )
    pins[str(reader)] = digest(reader)
    for source_path in (path, Path(cfg["baseline_terminal_readback"])):
        pins[str(source_path)] = digest(source_path)
    return complete
