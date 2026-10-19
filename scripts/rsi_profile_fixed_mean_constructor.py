"""Bounded isolated constructor profile; never stop or modify the live experiment."""

import argparse
import cProfile
import faulthandler
import json
import pstats
import signal
import subprocess
import sys
import time
from pathlib import Path


class ProfileBudgetReached(BaseException):
    pass


def require_scope(cfg):
    if (
        cfg.get("mode") != "SIM_ONLY"
        or type(cfg.get("profile_seconds")) is not int
        or cfg["profile_seconds"] != 180
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "training_authorized",
                "stop_live_worker_authorized",
            )
        )
    ):
        raise ValueError("bounded180s isolated non-training constructor profile required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    require_scope(cfg)
    wrapper = Path(__file__).resolve().parent.parent
    source, core, root = (Path(cfg[k]) for k in ("profile_source_root", "core_root", "output_root"))
    for path, head in ((wrapper, cfg["wrapper_head"]), (source, cfg["profile_source_head"])):
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
            != head
            or subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=path, text=True
            ).strip()
        ):
            raise ValueError("exact clean original constructor and wrapper sources required")
    if root.exists() or time.time() >= cfg["deadline_unix"]:
        raise ValueError("unused bounded diagnostic output required")
    # Import scientific helpers only from the same source/Core as the live job.
    sys.path[:0] = [str(source / "src"), str(source), str(core / "src")]
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_stage_release_contact_replay import digest

    pins = {str(args.config): digest(args.config), str(Path(__file__)): digest(__file__)}
    declaration_path = Path(cfg["live_declaration"])
    declaration = json.loads(declaration_path.read_text())
    if digest(declaration_path) != cfg["live_declaration_file_hash"] or declaration[
        "report_hash"
    ] != hash_json({k: v for k, v in declaration.items() if k != "report_hash"}):
        raise ValueError("whole actual live experiment declaration required")
    mean = Path(cfg["mean_model_path"])
    if digest(mean) != declaration["input_file_hashes"].get(str(mean)):
        raise ValueError("same whole source mean bytes required")
    pins[str(mean)] = digest(mean)
    pins[str(declaration_path)] = digest(declaration_path)
    for folder in (source / "src", core / "src"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    if args.preflight:
        print("BOUNDED_CONSTRUCTOR_PROFILE_READY_NO_MODEL_OR_PHYSICS", flush=True)
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError("56GiB available RAM before a second bounded constructor required")
    root.mkdir()
    write_once(
        root / "declaration.json",
        dict(
            config=cfg,
            input_file_hashes=pins,
            physical_execution_authorized=False,
            learning_authorized=False,
            live_worker_signal_or_mutation_authorized=False,
        ),
    )
    profiler = cProfile.Profile()
    phase = "IMPORTING_CONSTRUCTOR"
    stopped = False
    error_info = None
    started = time.time()

    def timeout(_signal, _frame):
        raise ProfileBudgetReached()

    old_handler = signal.signal(signal.SIGALRM, timeout)
    old_timer = signal.setitimer(signal.ITIMER_REAL, 180)
    if old_timer != (0.0, 0.0):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        raise ValueError("constructor profiler must own an unused process timer")
    with (root / "periodic-python-stacks.log").open("xb") as stacks:
        faulthandler.dump_traceback_later(30, repeat=True, file=stacks)
        profiler.enable()
        try:
            from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
            from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
                RecurrentSamplingEpisodeFactory,
            )

            phase = "LOADING_SAME_MEAN_JSON"
            print(phase, flush=True)
            mean_value = load_json_artifact(mean)
            phase = "CONSTRUCTING_SAME_FULL_FACTORY"
            print(phase, flush=True)
            RecurrentSamplingEpisodeFactory(mean_value)
            phase = "FACTORY_CONSTRUCTOR_COMPLETED"
        except ProfileBudgetReached:
            stopped = True
        except BaseException as error:
            error_info = dict(type=type(error).__name__, message=str(error))
        finally:
            profiler.disable()
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            faulthandler.cancel_dump_traceback_later()
    stats = pstats.Stats(profiler)
    functions = []
    for (filename, line, name), (
        primitive,
        total,
        self_time,
        cumulative,
        _callers,
    ) in stats.stats.items():
        functions.append(
            dict(
                file=filename,
                line=line,
                name=name,
                primitive_calls=primitive,
                total_calls=total,
                self_cpu_seconds=self_time,
                cumulative_cpu_seconds=cumulative,
            )
        )
    functions.sort(key=lambda x: x["cumulative_cpu_seconds"], reverse=True)
    stable = all(digest(path) == h for path, h in pins.items())
    result = dict(
        schema="soccer.rsi.bounded_fixed_mean_constructor_profile.v1",
        phase=phase,
        profile_budget_reached=stopped,
        error=error_info,
        wall_seconds=time.time() - started,
        functions=functions,
        source_input_bytes_unchanged=stable,
        new_physical_steps=0,
        new_optimizer_updates=0,
        neural_validation_forward_calls_not_instrumented=True,
        original_live_worker_not_touched=True,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(root / "complete.json", result)
    print("CONSTRUCTOR_PROFILE_COMPLETE", phase, result["report_hash"], flush=True)
    if not stable or error_info is not None:
        raise RuntimeError("constructor profile source drift or actual constructor error")


if __name__ == "__main__":
    main()
