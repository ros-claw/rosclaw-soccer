"""Bounded real-model preview/binding benchmark, no new physical execution.

Compare the existing private seed preview + external-document bind with the
internally generated seed episode. Shared weights, independently owned causal
state, same recorded body inputs. Not a native or end-to-end speed certificate.
"""

import argparse
import gc
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, core, root = (Path(cfg[k]) for k in ("source_root", "core_root", "output_root"))
    if (
        root.exists()
        or cfg.get("mode") != "SIM_ONLY"
        or cfg.get("hardware_authorized") is not False
        or cfg.get("promotion_authorized") is not False
        or cfg.get("new_physical_budget") != 0
        or cfg.get("new_optimizer_updates") != 0
        or cfg.get("seeds") != [30300, 30301]
        or Path(__file__).resolve() != source / "scripts/rsi_benchmark_generated_episode_binding.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused source-pinned two-seed numeric-only benchmark required")
    pins = {str(p): digest(p) for p in (Path(__file__).resolve(), args.config)}
    for name in ("mean_model", "body_trace", "compiled_world"):
        path = Path(cfg[name])
        if digest(path) != cfg[name + "_file_hash"]:
            raise ValueError(
                "exact independently selected complete model/body/world bytes required"
            )
        pins[str(path)] = cfg[name + "_file_hash"]
    for project in (source, core):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    sys.path[:0] = [str(source / "src"), str(source), str(core / "src")]
    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
    from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
        RecurrentSamplingEpisodeFactory,
    )
    from rosclaw_soccer.rsi.recurrent_sampling_motor import TRACE_FIELDS
    from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
    from scripts.rsi_atomic_artifacts import write_once

    if not callable(getattr(RecurrentSamplingEpisodeFactory, "generate_and_bind_seed", None)):
        raise ValueError("explicit source-pinned numeric API required")
    if args.preflight:
        print("GENERATED_EPISODE_BINDING_PREFLIGHT_ZERO_MODEL_OR_WORLD_ALLOCATIONS", flush=True)
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3 or time.time() >= cfg["deadline_unix"]:
        raise RuntimeError(
            "56GiB available before full-model allocation and fixed deadline required"
        )
    root.mkdir()
    declaration = dict(
        cfg,
        schema="soccer.rsi.generated_episode_binding_benchmark_declaration.v1",
        input_file_hashes=pins,
        order_by_seed={
            "30300": ["external_bind", "generated_bind"],
            "30301": ["generated_bind", "external_bind"],
        },
        same_archived_body_inputs_not_counterfactual_physics=True,
        independent_numeric_weights_claimed=False,
        native_parity_not_established=True,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(root / "declaration.json", declaration)
    try:
        started = time.perf_counter()
        mean = load_json_artifact(Path(cfg["mean_model"]))
        if mean["model_hash"] != cfg["expected_model_hash"]:
            raise ValueError("complete unchanged real numerical mean required")
        loaded = time.perf_counter() - started
        started = time.perf_counter()
        factory = RecurrentSamplingEpisodeFactory(mean)
        allocated = time.perf_counter() - started
        del mean
        gc.collect()
        if (
            factory._mean_hash != cfg["expected_canonical_mean_hash"]
            or factory._prototype_hash != cfg["expected_numeric_prototype_hash"]
        ):
            raise ValueError("entire mean or original numerical prototype differs")
        write_once(
            root / "allocated.json",
            dict(
                load_seconds=loaded,
                original_complete_constructor_seconds=allocated,
                canonical_mean_hash=factory._mean_hash,
                numeric_prototype_hash=factory._prototype_hash,
            ),
        )
        print("GENERATED_BINDING_FULL_ORIGINAL_FACTORY_ALLOCATED", allocated, flush=True)
        with np.load(cfg["body_trace"], allow_pickle=False) as archive:
            trace = dict(archive)
        world = mujoco.MjModel.from_binary_path(cfg["compiled_world"])
        joints = [int(world.joint(n).id) for n in G1_DDS_JOINT_NAMES]
        limits = world.jnt_range[joints[:12]].copy()
        rows = []
        for seed, order in (
            (30300, ("external_bind", "generated_bind")),
            (30301, ("generated_bind", "external_bind")),
        ):
            if time.time() >= cfg["deadline_unix"]:
                raise RuntimeError("fixed benchmark deadline reached")
            policies, episodes, seconds = {}, {}, {}
            for method in order:
                started = time.perf_counter()
                if method == "external_bind":
                    policy = factory.preview_from_seed(seed=seed)
                    episode = factory.bind(policy)
                else:
                    policy, episode = factory.generate_and_bind_seed(seed=seed)
                seconds[method] = time.perf_counter() - started
                policies[method], episodes[method] = policy, episode
                whole_hash = hash_json(policy)
                if policy["policy_hash"] != hash_json(
                    {k: v for k, v in policy.items() if k != "policy_hash"}
                ):
                    raise ValueError("complete ordinary preview seal differs")
                write_once(
                    root / f"measured-{seed}-{method}.json",
                    dict(
                        seed=seed,
                        method=method,
                        elapsed_seconds=seconds[method],
                        policy_hash=policy["policy_hash"],
                        whole_preview_hash=whole_hash,
                    ),
                )
                print(
                    "GENERATED_BINDING_METHOD_MEASURED", seed, method, seconds[method], flush=True
                )
            if policies["external_bind"] != policies["generated_bind"]:
                raise ValueError("complete actor/critic/parent/preview changed")
            left, right = episodes["external_bind"], episodes["generated_bind"]
            if (
                left._memory is right._memory
                or left._behavior._memory is right._behavior._memory
                or np.shares_memory(left.hidden_state, right.hidden_state)
            ):
                raise ValueError("independently owned causal episode state required")
            if left._recurrent.next_index != 0 or right._recurrent.next_index != 0:
                raise ValueError("fresh initial recurrent indices required")
            previous, actions, hidden = np.zeros(12), [], []
            for frame in range(300):
                nominal = trace["pre_motor_joint_target_rad"][frame, 0]
                options = dict(
                    frame=frame,
                    nominal_target=nominal,
                    baseline=nominal[:12],
                    limits=limits,
                    previous=previous,
                    previous_contact_forces=trace["force_n"][frame - 1, 0]
                    if frame
                    else np.zeros(6),
                )
                a = left.delta_at_frame(policies["external_bind"], trace, **options)
                b = right.delta_at_frame(policies["generated_bind"], trace, **options)
                if not np.array_equal(a, b) or not np.array_equal(
                    left.hidden_state, right.hidden_state
                ):
                    raise ValueError("complete actual-model action/hidden scan differs")
                if any(
                    not np.array_equal(left.sampled_transition[k], right.sampled_transition[k])
                    for k in TRACE_FIELDS
                ):
                    raise ValueError("actual-model draw/log-density/transition differs")
                actions.append(a.copy())
                hidden.append(left.hidden_state.copy())
                previous = a
            factory._stable()
            row = dict(
                seed=seed,
                order=list(order),
                combined_generation_binding_seconds=seconds,
                local_combined_speedup=seconds["external_bind"] / seconds["generated_bind"],
                whole_preview_hash=hash_json(policies["external_bind"]),
                compared_frames=300,
                complete_documents_equal=True,
                actions_hidden_and_sampling_transitions_bit_equal=True,
                action_bytes_hash=hash_bytes(np.asarray(actions).tobytes()),
                hidden_bytes_hash=hash_bytes(np.asarray(hidden).tobytes()),
            )
            row["report_hash"] = hash_json(row)
            write_once(root / f"qualified-{seed}.json", row)
            rows.append(row)
            del policies, episodes, policy, episode, left, right
            gc.collect()
        if any(digest(Path(name)) != expected for name, expected in pins.items()):
            raise ValueError("source/model/body input changed during benchmark")
        result = dict(
            schema="soccer.rsi.generated_episode_binding_real_model_benchmark.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            original_complete_constructor_seconds=allocated,
            maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            compared_numeric_frames=600,
            shared_weights_between_numeric_paths=True,
            new_native_bodies=0,
            new_independent_physics_steps=0,
            new_optimizer_updates=0,
            native_collector_optimization_enabled=False,
            end_to_end_speedup_claimed=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(root / "complete.json", result)
        print("GENERATED_EPISODE_BINDING_BENCHMARK_COMPLETE", result["report_hash"], flush=True)
    except BaseException as error:
        write_once(
            root / "fatal.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
