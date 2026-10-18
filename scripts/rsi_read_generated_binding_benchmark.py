"""Read original benchmark seals and pinned bytes, without reallocating models.

This is artifact readback, not an independent rerun of the numeric decoder or
physics. Timings qualify only local combined generation/binding preparation.
"""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def readback(root):
    pins = {}

    def read(path, sealed=False):
        pins[str(path)] = digest(path)
        value = json.loads(path.read_text())
        if sealed and value.get("report_hash") != hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        ):
            raise ValueError("original report seal differs")
        return value

    declaration = read(root / "declaration.json", True)
    complete = read(root / "complete.json", True)
    allocation = read(root / "allocated.json")
    if (
        complete["schema"] != "soccer.rsi.generated_episode_binding_real_model_benchmark.v1"
        or complete["declaration_hash"] != declaration["report_hash"]
        or declaration["seeds"] != [30300, 30301]
        or complete["compared_numeric_frames"] != 600
        or allocation["original_complete_constructor_seconds"]
        != complete["original_complete_constructor_seconds"]
        or allocation["canonical_mean_hash"] != declaration["expected_canonical_mean_hash"]
        or allocation["numeric_prototype_hash"] != declaration["expected_numeric_prototype_hash"]
        or any(
            complete[key] is not False
            for key in (
                "native_collector_optimization_enabled",
                "end_to_end_speedup_claimed",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(
            complete[key] != 0
            for key in (
                "new_native_bodies",
                "new_independent_physics_steps",
                "new_optimizer_updates",
            )
        )
    ):
        raise ValueError("original complete numeric-only benchmark identity required")
    for name, expected in declaration["input_file_hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError("original whole source/model/body/world bytes differ")
        pins[name] = expected
    rows = []
    for seed, row in zip([30300, 30301], complete["rows"], strict=True):
        if (
            row != read(root / f"qualified-{seed}.json", True)
            or row["seed"] != seed
            or row["order"] != declaration["order_by_seed"][str(seed)]
            or row["compared_frames"] != 300
            or row["complete_documents_equal"] is not True
            or row["actions_hidden_and_sampling_transitions_bit_equal"] is not True
        ):
            raise ValueError("complete ordered paired numeric evidence required")
        measurements = [read(root / f"measured-{seed}-{m}.json") for m in row["order"]]
        if measurements[0]["policy_hash"] != measurements[1]["policy_hash"]:
            raise ValueError("original whole policy identities differ")
        for method, measured in zip(row["order"], measurements, strict=True):
            seconds = measured["elapsed_seconds"]
            if (
                measured["seed"] != seed
                or measured["method"] != method
                or measured["whole_preview_hash"] != row["whole_preview_hash"]
                or not math.isfinite(seconds)
                or seconds <= 0
                or seconds != row["combined_generation_binding_seconds"][method]
            ):
                raise ValueError("original stage measurement binding differs")
        old, new = [
            row["combined_generation_binding_seconds"][m]
            for m in ("external_bind", "generated_bind")
        ]
        if row["local_combined_speedup"] != old / new:
            raise ValueError("local timing ratio differs")
        rows.append(row)
    if any(digest(Path(name)) != expected for name, expected in pins.items()):
        raise ValueError("input bytes changed during readback")
    result = dict(
        schema="soccer.rsi.generated_binding_benchmark_readback.v1",
        complete_hash=complete["report_hash"],
        rows=rows,
        all_original_source_model_body_world_and_stage_bytes_verified=True,
        verified_file_count=len(pins),
        original_complete_constructor_seconds=complete["original_complete_constructor_seconds"],
        numeric_decoder_reexecuted_here=False,
        raw_action_and_hidden_arrays_reconstructed_here=False,
        new_physical_steps=0,
        new_optimizer_updates=0,
        native_noninterference_established=False,
        end_to_end_speedup_claimed=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        reader_source_hash=digest(Path(__file__)),
    )
    result["report_hash"] = hash_json(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            args.unit,
            "-p",
            "LoadState",
            "-p",
            "MainPID",
            "-p",
            "ActiveState",
            "-p",
            "Result",
        ],
        text=True,
    )
    if args.output.exists() or set(state.splitlines()) != {
        "LoadState=loaded",
        "MainPID=0",
        "ActiveState=inactive",
        "Result=success",
    }:
        raise ValueError("unused output and loaded successful terminal source unit required")
    result = readback(args.root)
    write_once(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
