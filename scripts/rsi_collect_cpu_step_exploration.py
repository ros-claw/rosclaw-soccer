"""Collect actual per-frame motor exploration and replayed evidence on CPU MuJoCo."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor, make_model
from rosclaw_soccer.rsi.contextual_first_touch_option import first_touch_reward
from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame, make_sampling_view
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_step_motor_pilot import COURSES
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once
from scripts.rsi_train_protected_online_motor_v308 import _head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "soccer-python",
        "scene",
        "model-root",
        "late-swing-policy",
        "core-root",
        "step-model",
        "consumed-bank",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    runner = source / "scripts/rsi_mujoco_motor_transfer.py"
    warm = json.loads(args.step_model.read_text())
    bank = _sealed(args.consumed_bank)
    if bank["partition"] != "TRAIN_CONSUMED" or not set(COURSES) <= {
        (r["seed"], r["lane"]) for r in bank["courses"]
    }:
        parser.error("CPU training may only execute previously consumed courses")
    views = [
        make_model(make_sampling_view(warm, seed=202610026 + c * 100 + i, std=0.1))
        for c in range(4)
        for i in range(8)
    ]
    greedy = make_model(warm)
    commitment = dict(
        schema="soccer.rsi.cpu_step_exploration_commitment.v1",
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        runner_hash=hash_bytes(runner.read_bytes()),
        warm_model_hash=warm["model_hash"],
        consumed_bank_hash=bank["report_hash"],
        courses=[list(c) for c in COURSES],
        sample_model_hashes=[v["model_hash"] for v in views],
        greedy_model_hash=greedy["model_hash"],
        samples_per_course=8,
        std_raw=0.1,
        partition="TRAIN_CONSUMED",
        backend="MuJoCo CPU",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "logs").mkdir()
    (args.output_root / "models").mkdir()
    write_once(args.output_root / "commitment.json", commitment)
    greedy_path = args.output_root / "models/greedy.json"
    write_once(greedy_path, greedy)
    for index, view in enumerate(views):
        write_once(args.output_root / "models" / f"sample-{index}.json", view)

    def execute(
        course: int, label: str, model_path: Path
    ) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
        seed, lane = COURSES[course]
        folder = args.output_root / f"seed{seed}-lane{lane}-{label}"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(source / "src"), str(args.core_root / "src")))
        env["OPENBLAS_NUM_THREADS"] = "1"
        command = [
            str(args.soccer_python),
            str(runner),
            "--scene",
            str(args.scene),
            "--model-root",
            str(args.model_root),
            "--late-swing-policy",
            str(args.late_swing_policy),
            "--step-model",
            str(model_path),
            "--consumed-bank",
            str(args.consumed_bank),
            "--seed",
            str(seed),
            "--lane",
            str(lane),
            "--output-root",
            str(folder),
        ]
        with (args.output_root / "logs" / f"seed{seed}-lane{lane}-{label}.log").open("x") as log:
            result = subprocess.run(
                command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if result.returncode or not (folder / "report.json").is_file():
            raise RuntimeError(f"CPU execution incomplete: {folder}")
        raw = _sealed(folder / "report.json")
        expected_model = json.loads(model_path.read_text())
        if (
            raw["step_model_hash"] != expected_model["model_hash"]
            or raw["source_hash"] != commitment["runner_hash"]
        ):
            raise ValueError("CPU physical execution used a different model or source")
        review = audit_cpu_transfer(folder, runner)
        write_once(folder / "review.json", review)
        outcome = {
            k: review[k]
            for k in (
                "first_contact_frame",
                "contact_body_indices",
                "clean_foot_only",
                "minimum_pelvis_z_m",
                "maximum_lateral_excursion_m",
                "high_quality",
                "forward_60_m",
                "lateral_60_m",
                "lateral_over_forward_60",
            )
        }
        outcome["reward"] = first_touch_reward(
            {**outcome, "max_lateral_excursion_m": outcome["maximum_lateral_excursion_m"]}
        )
        return folder, raw, review, outcome

    def worker(course: int) -> dict[str, Any]:
        _, raw, review, outcome = execute(course, "greedy", greedy_path)
        baseline = dict(
            report_hash=raw["report_hash"], review_hash=review["report_hash"], **outcome
        )
        observations, latents, densities, returns, scales, groups, records = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for sample in range(8):
            index = course * 8 + sample
            folder, raw, review, outcome = execute(
                course, f"sample-{sample}", args.output_root / "models" / f"sample-{index}.json"
            )
            decoder = CompiledStepMotor(raw["executed_motor_policy"])
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as body:
                for frame in range(30, 300):
                    x = features_at_frame(
                        body,
                        frame=frame,
                        nominal_target=body["pre_motor_joint_target_rad"][frame, 0],
                        previous=body["motor_delta_rad"][frame - 1, 0],
                        previous_contact_forces=body["force_n"][frame - 1, 0],
                    )
                    z, density = decoder.latent_sample(x, frame)
                    observations.append(x)
                    latents.append(z)
                    densities.append(density)
                    returns.append(terminal_return(outcome))
                    scales.append(0.1)
                    groups.append(index)
            records.append(
                dict(
                    seed=COURSES[course][0],
                    lane=COURSES[course][1],
                    sample=sample,
                    group=index,
                    folder=str(folder),
                    report_hash=raw["report_hash"],
                    review_hash=review["report_hash"],
                    outcome=outcome,
                )
            )
            print(
                f"CPU_EXPLORATION_EXECUTED course={course} sample={sample} "
                f"HQ={outcome['high_quality']}",
                flush=True,
            )
        return dict(
            baseline=baseline,
            records=records,
            arrays=dict(
                observation=np.asarray(observations),
                latent_action=np.asarray(latents),
                old_log_probability=np.asarray(densities),
                terminal_return=np.asarray(returns),
                std_raw=np.asarray(scales),
                trajectory_index=np.asarray(groups),
            ),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))
    if (
        _head(source) != commitment["source_commit"]
        or hash_bytes(runner.read_bytes()) != commitment["runner_hash"]
    ):
        raise ValueError("CPU exploration source drift")
    arrays: dict[str, Any] = {
        k: np.concatenate([r["arrays"][k] for r in results]) for k in results[0]["arrays"]
    }
    data_path = args.output_root / "rollouts.npz"
    np.savez_compressed(data_path, **arrays)
    manifest = dict(
        schema="soccer.rsi.audited_cpu_step_rollout_bank.v1",
        commitment=commitment,
        partition="TRAIN_CONSUMED",
        records=[r for result in results for r in result["records"]],
        baselines=[r["baseline"] for r in results],
        physical_rollout_count=32,
        physical_executions=36,
        frame_sample_count=len(arrays["observation"]),
        independent_contexts=4,
        data_hash=hash_bytes(data_path.read_bytes()),
        promotion_authorized=False,
        hardware_authorized=False,
    )
    manifest["report_hash"] = hash_json(manifest)
    write_once(args.output_root / "rollout_manifest.json", manifest)
    print(json.dumps(dict(report_hash=manifest["report_hash"], physical_executions=36)), flush=True)


if __name__ == "__main__":
    main()
