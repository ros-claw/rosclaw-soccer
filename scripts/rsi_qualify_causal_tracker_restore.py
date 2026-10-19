"""Six real-model causal tracker restorations; zero physics or actor optimization."""

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_impact_screen_inputs import prepare
from scripts.rsi_stage_release_contact_replay import digest


def require_scope(cfg):
    if (
        cfg.get("mode") != "SIM_ONLY"
        or cfg.get("profile") != "six-causal-tracker-restore-v1"
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_cases", 6),
                ("maximum_planner_run_calls", 60),
                ("current_encoder_decoder_pairs", 6),
                ("next_golden_encoder_decoder_pairs", 6),
                ("maximum_constructor_parity_probe_passes", 16),
                ("new_physical_steps", 0),
                ("new_closed_loop_policy_trials", 0),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "private_fresh_accessed",
                "training_authorized",
                "teacher_qualified",
                "actor_future_event_input_allowed",
            )
        )
    ):
        raise ValueError(
            "fixed six current-state restoration qualification, no physics/learning required"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config, scope_validator=require_scope)
    if Path(__file__).resolve() != p.source / "scripts/rsi_qualify_causal_tracker_restore.py":
        raise ValueError("exact frozen current-state qualification runner required")
    model_root = Path(p.cfg["model_root"])
    for relative, h in p.cfg["model_file_hashes"].items():
        if (
            relative
            not in (
                "planner_sonic.onnx",
                "low_latency/model_encoder.onnx",
                "low_latency/model_decoder.onnx",
                "low_latency/observation_config.yaml",
            )
            or digest(model_root / relative) != h
        ):
            raise ValueError("declared original ordinary model assets required")
        p.pins[str(model_root / relative)] = h
    if set(p.cfg["model_file_hashes"]) != {
        "planner_sonic.onnx",
        "low_latency/model_encoder.onnx",
        "low_latency/model_decoder.onnx",
        "low_latency/observation_config.yaml",
    }:
        raise ValueError("complete planner/encoder/decoder/observation assets required")
    if any(c["frame"] is None or c["frame"] >= 298 or (c["frame"] + 1) % 20 == 0 for c in p.cases):
        raise ValueError("all original next-frame non-replan boundaries required")
    if args.preflight:
        print("CURRENT_TRACKER_WHOLE_SIX_READY_NO_MODEL_OR_PHYSICS_ALLOCATION", flush=True)
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 8 * 1024**3:
        raise ValueError(
            "8GiB available RAM for one Foundation/planner-only qualification required"
        )
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.causal_tracker_restore_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        all_six_source_cases_retained=True,
        future_body_used_only_for_offline_next_frame_golden_comparison=True,
        complete_feedback_controller_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    runs, current_done, next_done = 0, 0, 0
    rows = []
    model = None
    current_case = None
    try:
        import onnxruntime as ort
        import torch

        from rosclaw_soccer.providers.g1.sonic_causal_capture_state import (
            WIDTHS,
            restore_current_tracker,
        )
        from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
        from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
        from rosclaw_soccer.rsi.sonic_locomotion_reference_audit import replay_locomotion_segments

        torch.set_num_threads(1)
        model = FrozenSonicG1Torch(model_root, variant="low_latency", device="cpu")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        planner = ort.InferenceSession(
            str(model_root / "planner_sonic.onnx"), options, providers=["CPUExecutionProvider"]
        )

        class CountedPlanner:
            def run(self, *a, **kw):
                nonlocal runs
                if runs >= 60 or time.time() >= p.cfg["deadline_unix"]:
                    raise RuntimeError("fixed planner-call/deadline exhausted")
                runs += 1
                return planner.run(*a, **kw)

        class ReferenceReady(Exception):
            pass

        for index, case in enumerate(p.cases):
            current_case = index
            frame = case["frame"]
            trace = case["trace"]
            commitment = json.loads(
                Path(case["receipt"]["witnesses"]["commitment"]["path"]).read_text()
            )
            if commitment["foundation_hash"] != model.qualification.qualification_hash:
                raise ValueError("exact original source-model qualification identity required")
            envelope = load_json_artifact(Path(case["receipt"]["witnesses"]["report"]["path"]))
            if envelope.get("schema") != "rosclaw.growth.shared_json_payload.v1":
                raise ValueError("original previously qualified report envelope required")
            events = envelope["stripped_document"]["default_locomotion_planner_events"]
            cached = None

            def observe(tick, reference, stop_frame=frame // 20 * 20):
                nonlocal cached
                if tick == stop_frame:
                    cached = reference.copy()
                    raise ReferenceReady

            before = runs
            with contextlib.suppress(ReferenceReady):
                replay_locomotion_segments(
                    CountedPlanner(), trace, events, 2, reference_observer=observe
                )
            if cached is None or runs - before != frame // 20 + 1:
                raise ValueError(
                    "only original past planner calls through current cached reference required"
                )
            names = {"canonical_qpos", "canonical_qvel"} | {
                "foundation_neural_" + k for k in WIDTHS
            }
            prefix = {name: trace[name][: frame + 1].copy() for name in names}
            restored = restore_current_tracker(
                model, cached_reference=cached, prefix=prefix, frame=frame
            )
            current_done += 1
            # OFFLINE golden test only, not a prediction from unknown future state.
            q, v = trace["canonical_qpos"][frame + 1], trace["canonical_qvel"][frame + 1]
            restored.observe(q, v)
            restored.update(frame + 1, q, v)
            captured = restored.neural_observation()
            for name in WIDTHS:
                if not np.array_equal(
                    captured[name].cpu().numpy(), trace["foundation_neural_" + name][frame + 1]
                ):
                    raise ValueError("actual next-golden neural inference differs: " + name)
            next_done += 1
            row = dict(
                index=index,
                job=case["job"],
                source_receipt_hash=case["receipt"]["receipt_hash"],
                current_frame=frame,
                past_planner_calls=runs - before,
                current_model_call_and_history_bit_exact=True,
                next_offline_golden_model_call_bit_exact=True,
                restored_prefix_ends_at_current_frame=True,
                physics_performed=False,
                complete_feedback_controller_qualified=False,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            row["report_hash"] = hash_json(row)
            write_once(p.root / f"case-{index}-complete.json", row)
            rows.append(row)
            print("CAUSAL_TRACKER_RESTORE_COMPLETE", index, "PAST_PLANS", runs - before, flush=True)
        if any(digest(name) != h for name, h in p.pins.items()):
            raise ValueError("original model/source/input changed")
        report = dict(
            schema="soccer.rsi.causal_tracker_restore_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            planner_run_calls=runs,
            current_encoder_decoder_pairs=current_done,
            next_offline_golden_encoder_decoder_pairs=next_done,
            constructor_parity_samples=model.parity["samples"],
            constructor_parity_probe_passes_from_pinned_source=16,
            all_six_original_cases_retained=len(rows) == 6,
            new_physical_steps=0,
            new_closed_loop_policy_trials=0,
            new_optimizer_updates=0,
            complete_feedback_controller_qualified=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        report["report_hash"] = hash_json(report)
        write_once(p.root / "complete.json", report)
    except BaseException as error:
        write_once(
            p.root / "failure.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                current_case=current_case,
                planner_run_calls_attempted=runs,
                current_pairs_completed=current_done,
                next_pairs_completed=next_done,
                constructor_returned=model is not None,
                constructor_probe_passes_not_instrumented_on_failure=True,
                new_physical_steps=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
