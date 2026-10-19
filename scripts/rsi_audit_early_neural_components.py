"""All six measured Foundation histories/arithmetic and unlogged planner inputs.

Not an independent residual/IK/search audit, new physics, learning, or teacher.
Requires prior complete raw readback and whole recorded-target physical replay.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    if (
        cfg.get("profile") != "six-early-neural-components-v1"
        or cfg.get("mode") != "SIM_ONLY"
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_cases", 6),
                ("planner_call_budget", 90),
                ("executed_encoder_decoder_pairs", 1800),
                ("new_physical_steps", 0),
                ("new_optimizer_updates", 0),
            )
        )
        or cfg.get("additional_original_onnx_comparison_calls") is not True
        or any(
            cfg.get(k) is not False
            for k in (
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
                "automatic_retry",
                "whole_controller_qualification_claimed",
            )
        )
    ):
        raise ValueError("exact whole-six no-learning neural-component diagnostic required")


def prepare(path):
    cfg = json.loads(path.read_text())
    require_scope(cfg)
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        Path(__file__).resolve() != source / "scripts/rsi_audit_early_neural_components.py"
        or root.exists()
        or not time.time() < cfg["deadline_unix"] <= time.time() + 18000
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("frozen exact entry, unused output and bounded deadline required")
    for key in ("producer_unit", "readback_unit", "physical_unit"):
        require_terminal(cfg[key])
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts")
        for p in folder.rglob("*.py")
    }
    pins[str(path)] = digest(path)

    def pin(name, expected=None):
        p = Path(name)
        h = digest(p)
        if (
            p.is_symlink()
            or (expected is not None and h != expected)
            or (str(p) in pins and pins[str(p)] != h)
        ):
            raise ValueError("whole same ordinary source/input bytes required")
        pins[str(p)] = h

    def sealed(name, expected):
        pin(name, expected)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole logical evidence seal required")
        return d

    complete = sealed(
        Path(cfg["producer_root"]) / "complete.json", cfg["producer_complete_file_hash"]
    )
    readback = sealed(cfg["readback_path"], cfg["readback_file_hash"])
    physical = sealed(
        Path(cfg["physical_root"]) / "complete.json", cfg["physical_complete_file_hash"]
    )
    physical_decl = sealed(Path(cfg["physical_root"]) / "declaration.json", None)
    if (
        complete["schema"] != "soccer.rsi.early_impact_h300_complete.v1"
        or readback["schema"] != "soccer.rsi.early_impact_h300_readback.v1"
        or physical["schema"] != "soccer.rsi.recorded_early_h300_complete.v1"
        or readback["complete_hash"] != complete["report_hash"]
        or physical_decl["producer_complete_hash"] != complete["report_hash"]
        or physical_decl["readback_hash"] != readback["report_hash"]
        or physical["declaration_hash"] != physical_decl["report_hash"]
        or physical["actual_physical_steps"] != 18000
        or any(len(d["rows"]) != 6 for d in (complete, readback, physical))
    ):
        raise ValueError("same complete whole-six readback and recorded-physics proof required")
    for name, h in physical_decl["input_file_hashes"].items():
        pin(name, h)
    for i, (row, p, r) in enumerate(
        zip(complete["rows"], physical["rows"], readback["rows"], strict=True)
    ):
        if (
            row["index"] != i
            or p["index"] != i
            or r["index"] != i
            or p["source_case_hash"] != row["report_hash"]
            or r["actual_outcome"] != row["causal_rollout_outcome"]
            or p["physical_review"]["actual_physical_steps"] != 3000
        ):
            raise ValueError("all six original ordered physical bodies required")
        pin(row["arrays"]["path"], row["arrays"]["sha256"])
    md = Path(cfg["model_declaration"])
    if str(md) not in pins:
        raise ValueError("model declaration must belong to independently checked physical inputs")
    model = json.loads(md.read_text())["config"]
    modelroot = Path(model["model_root"])
    if set(model["model_file_hashes"]) != {
        "planner_sonic.onnx",
        "low_latency/model_encoder.onnx",
        "low_latency/model_decoder.onnx",
        "low_latency/observation_config.yaml",
    }:
        raise ValueError("complete original planner/Foundation asset binding required")
    for rel, h in model["model_file_hashes"].items():
        pin(modelroot / rel, h)
    return cfg, root, pins, complete, modelroot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg, root, pins, complete, modelroot = prepare(args.config)
    if args.preflight:
        print("SIX_EARLY_NEURAL_COMPONENTS_READY_NO_MODELS_OR_NEW_PHYSICS", flush=True)
        return
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 16 * 1024**3:
        raise ValueError(
            "16GiB available before planner and independent Foundation allocation required"
        )
    root.mkdir()
    declaration = dict(
        schema="soccer.rsi.six_early_neural_components_declaration.v1",
        config=cfg,
        input_file_hashes=pins,
        producer_complete_hash=complete["report_hash"],
        original_planner_event_logs_available=False,
        recorded_commands_consumed=True,
        whole_controller_qualification_claimed=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(root / "declaration.json", declaration)
    rows, current, phase = [], None, "ALLOCATION"
    attempted = completed = 0
    try:
        import onnxruntime as ort
        import torch

        from rosclaw_soccer.rsi.early_foundation_audit import audit_early_foundation
        from rosclaw_soccer.rsi.foundation_torch_capture_audit import TorchCallAuditor
        from rosclaw_soccer.rsi.unlogged_encoder_reference_audit import (
            audit_unlogged_encoder_reference,
        )

        torch.set_num_threads(1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        raw = ort.InferenceSession(
            str(modelroot / "planner_sonic.onnx"), options, providers=["CPUExecutionProvider"]
        )

        class Planner:
            def run(self, outputs, feed):
                nonlocal attempted, completed
                if attempted >= 90 or time.time() >= cfg["deadline_unix"]:
                    raise RuntimeError("fixed planner/deadline budget exhausted; no retry")
                attempted += 1
                result = raw.run(outputs, feed)
                completed += 1
                return result

        planner = Planner()
        auditor = TorchCallAuditor(modelroot)
        for i, row in enumerate(complete["rows"]):
            current, phase = i, "PLANNER_ENCODER_FEATURES"
            with np.load(row["arrays"]["path"], allow_pickle=False) as a:
                trace = dict(a)
            reference = audit_unlogged_encoder_reference(planner, trace)
            phase = "FOUNDATION_ARITHMETIC_AND_BODY_HISTORY"
            neural = audit_early_foundation(trace, auditor)
            result = dict(
                index=i, source_case_hash=row["report_hash"], reference=reference, neural=neural
            )
            result["report_hash"] = hash_json(result)
            write_once(root / f"case-{i}-complete.json", result)
            rows.append(result)
            print("EARLY_NEURAL_COMPONENT_CASE_COMPLETE", i, flush=True)
        if attempted != 90 or completed != 90 or any(digest(name) != h for name, h in pins.items()):
            raise ValueError("whole bounded planner calls and immutable source/inputs required")
        result = dict(
            schema="soccer.rsi.six_early_neural_components_complete.v1",
            library_versions=dict(
                onnxruntime=str(ort.__version__),
                torch=str(torch.__version__),
                numpy=str(np.__version__),
            ),
            declaration_hash=declaration["report_hash"],
            rows=rows,
            attempted_planner_calls=attempted,
            completed_planner_calls=completed,
            executed_encoder_decoder_pairs=1800,
            additional_original_onnx_comparison_calls=True,
            new_physical_steps=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            whole_controller_qualified=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(root / "complete.json", result)
    except BaseException as error:
        write_once(
            root / "failure.json",
            dict(
                current_case=current,
                phase=phase,
                completed_cases=len(rows),
                attempted_planner_calls=attempted,
                completed_planner_calls=completed,
                current_neural_partial_call_count="UNOBSERVED_BOUNDED_BY_300_PAIRS"
                if phase == "FOUNDATION_ARITHMETIC_AND_BODY_HISTORY"
                else 0,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
                new_physical_steps=0,
            ),
        )
        raise


if __name__ == "__main__":
    main()
