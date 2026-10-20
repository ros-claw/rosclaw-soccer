"""Read-only full three-seed categorical experiment authentication and verdicts."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "core-source", "plan", "worker-source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "expected-head",
        "expected-inputs-hash",
        "expected-plan-hash",
        "expected-anchor-hash",
    ):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, _ = admit(a)
    if digest(a.plan) != a.expected_plan_hash:
        raise ValueError("unchanged declared complete three-seed pipeline required")
    plan = json.loads(a.plan.read_text())
    if (
        plan["seeds"] != [17, 29, 43]
        or plan["training_steps_per_seed"] != 1000
        or plan["planned_native_steps"] != 744000
        or len(plan["jobs"]) != 16
        or subprocess.check_output(
            ["git", "-C", str(a.worker_source), "rev-parse", "HEAD"], text=True
        ).strip()
        != plan["source_head"]
        or subprocess.check_output(["git", "-C", str(a.worker_source), "status", "--porcelain"])
    ):
        raise ValueError("exact immutable sixteen-stage source declaration required")
    for folder in ("src", "scripts"):
        for old in (a.worker_source / folder).rglob("*.py"):
            if digest(old) != digest(Path.cwd() / old.relative_to(a.worker_source)):
                raise ValueError("review cannot change worker control semantics")
    pins[str(a.plan)] = a.expected_plan_hash
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            plan_hash=a.expected_plan_hash,
            new_native_steps=0,
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        jobs = plan["jobs"]
        for job in jobs:
            wait_for_terminal_collection(job["unit"] + ".service", wait=True)
        imports = {}
        for job in jobs:
            root = Path(job["output"])
            d = json.loads((root / "declaration.json").read_text())
            c = json.loads((root / "complete.json").read_text())
            stage_source = job.get("source_head", plan["source_head"])
            if (
                d["source_head"] != stage_source
                or c["source_head"] != stage_source
                or c["hardware_authorized"] is not False
                or c["promotion_authorized"] is not False
            ):
                raise ValueError("stage source identity differs")
            for k, h in d["input_file_hashes"].items():
                if k in pins and pins[k] != h:
                    raise ValueError("conflicting pipeline inputs")
                pins[k] = h
            for name in ("declaration.json", "complete.json"):
                pins[str(root / name)] = digest(root / name)
            imports[job["unit"]] = (root, d, c)
        check_pins(pins)
        summaries = review(a, spec, bd, pins, jobs, imports)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="ALL_THREE_CATEGORICAL_SEEDS_NATIVE_PRIMITIVES_AUTHENTICATED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                summaries=summaries,
                reviewed_native_steps=744000,
                new_native_steps=0,
                optimizer_steps=0,
                candidate_learning_updates_observed=3000,
                updates_not_reexecuted_by_this_review=True,
                independently_reintegrated_by_this_review=False,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        print("CATEGORICAL_THREE_SEEDS_REVIEWED", summaries, flush=True)
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(error=repr(error), automatic_retry_authorized=False, new_native_steps=0),
        )
        raise


def review(a, spec, bd, pins, jobs, imports):
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.full_entry_forecast import score_full_entry
    from rosclaw_soccer.rsi.grouped_actor_verdict import grouped_actor_verdict
    from rosclaw_soccer.rsi.optimizer_lineage_audit import exact_tensor_tree
    from scripts.rsi_motion_student_rollout import validate_motion_recording
    from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, exact_array
    from scripts.rsi_native_context_recovery_common import digest

    world = mujoco.MjModel.from_binary_path(spec["world"])
    summaries = {}
    teachers = [imports[j["unit"]] for j in jobs if j["role"] == "train"]
    if len(teachers) != 3 or sorted(d["seed"] for _, d, _ in teachers) != [17, 29, 43]:
        raise ValueError("three unique training seeds required")
    probe = [j for j in jobs if j["role"] == "initial_parity"]
    if len(probe) != 1:
        raise ValueError("one initial parity stage required")
    probe_root, _, probe_c = imports[probe[0]["unit"]]
    if (
        probe_c["zero_head_native_identity_verified"] is not True
        or probe_c["variant"] != "initial"
        or [r["course_index"] for r in probe_c["results"]] != [0, 10, 15, 16]
    ):
        raise ValueError("complete initial physical parity proof required")
    for job in jobs:
        root, d, c = imports[job["unit"]]
        if job["role"] == "train":
            if (
                c["status"] != "CATEGORICAL_MOTION_TRAINED_REQUIRES_NATIVE_EXAM"
                or c["optimizer_steps"] != 1000
                or c["validation_sequences"] != 0
                or c["anchor_hash"] != a.expected_anchor_hash
                or c["teacher_train_indices"] != [2, 6, 12, 22, 24, 32, 34, 36]
                or c["own_train_indices"] != [0, 10, 15, 16]
            ):
                raise ValueError("fixed twelve train memories and complete learning required")
            for name, h in (
                ("candidate.pt", c["checkpoint_hash"]),
                ("initial.pt", c["initial_checkpoint_hash"]),
            ):
                if digest(root / name) != h:
                    raise ValueError("checkpoint bytes changed")
                pins[str(root / name)] = h
            anchors = [
                Path(k)
                for k, h in d["input_file_hashes"].items()
                if h == a.expected_anchor_hash and k.endswith(".pt")
            ]
            if len(anchors) != 1:
                raise ValueError("one frozen neural parent required")
            anchor = torch.load(anchors[0], map_location="cpu", weights_only=True)["state_dict"]
            candidate = torch.load(root / "candidate.pt", map_location="cpu", weights_only=True)
            for k, v in anchor.items():
                exact_tensor_tree(candidate["state_dict"]["parent." + k], v, path=k)
            if (
                candidate["optimizer_steps"] != 1000
                or candidate["optimizer"]["param_groups"][0]["params"] != list(range(4))
                or any(float(s["step"]) != 1000 for s in candidate["optimizer"]["state"].values())
            ):
                raise ValueError("complete actual classifier Adam clocks required")
            continue
        if (
            c["status"] != "CATEGORICAL_NATIVE_AND_NEURAL_REPLAY_VERIFIED"
            or c["new_native_steps"] != 6000 * len(c["results"])
            or c["optimizer_steps"] != 0
            or d["indices"] != [r["course_index"] for r in c["results"]]
            or c["anchor_hash"] != a.expected_anchor_hash
        ):
            raise ValueError("exact whole shard accounting required")
        if job["role"] == "candidate_exam" and d["indices"] != list(range(40))[job["part"] :: 4]:
            raise ValueError("original whole40 strided partition required")
        seed = job["seed"] if job["role"] == "candidate_exam" else 17
        training = [t for _, td, t in teachers if td["seed"] == seed][0]
        expected_hash = training[
            "checkpoint_hash" if job["role"] == "candidate_exam" else "initial_checkpoint_hash"
        ]
        if d["candidate_hash"] != expected_hash or c["candidate_hash"] != expected_hash:
            raise ValueError("native shard not bound to its exact seed's checkpoint")
        for row in c["results"]:
            i = row["course_index"]
            folder = root / f"course{i:02d}"
            if (
                row["split"] != bd["rows"][i]["split"]
                or json.loads((folder / "complete.json").read_text()) != row
                or digest(folder / "native.npz") != row["native_hash"]
            ):
                raise ValueError("whole raw trajectory and course identity required")
            pins[str(folder / "complete.json")] = digest(folder / "complete.json")
            pins[str(folder / "native.npz")] = row["native_hash"]
            with np.load(folder / "native.npz", allow_pickle=False) as z:
                trace = {k: z[k].copy() for k in z.files}
            validate_motion_recording(trace, nbody=world.nbody)
            if job["role"] == "initial_parity":
                original = [
                    Path(k)
                    for k in pins
                    if k.endswith(f"/round0/evaluate{i % 4}/course{i:02d}/native.npz")
                ]
                if len(original) != 1:
                    raise ValueError("one original anchor parity trace required")
                with np.load(original[0], allow_pickle=False) as z:
                    for key in (
                        *PHYSICAL_FIELDS,
                        "student_motion_token",
                        "hidden_after",
                        "student_features",
                        "muscle_proprioception",
                        "consumed_action_isaac",
                        "value_estimate",
                    ):
                        exact_array(trace[key], z[key])
            if (
                score_full_entry(world, {k: v[20:] for k, v in trace.items()}) != row["score"]
                or diagnose_entry_prefix(world, {k: v[:20] for k, v in trace.items()})
                != row["prefix_diagnostics"]
            ):
                raise ValueError("raw native outcomes contradict reported result")
            proof = row["replay"]
            if (
                row["native_steps"] != 6000
                or proof["additional_native_steps"] != 3000
                or proof["student_neural_and_muscle_native_bit_exact"] is not True
                or proof["intention_network_calls"] != 900
                or proof["muscle_decoder_calls"] != 280
                or proof["teacher_encoder_calls"] != 0
                or proof["teacher_planner_calls"] != 0
                or proof["promotion_authorized"] is not False
                or proof["hardware_authorized"] is not False
            ):
                raise ValueError("independent full neural/muscle/native replay required")
    for seed in (17, 29, 43):
        shards = [j for j in jobs if j["role"] == "candidate_exam" and j["seed"] == seed]
        if sorted(j["part"] for j in shards) != [0, 1, 2, 3]:
            raise ValueError("four unique shards per seed required")
        rows = [r for j in shards for r in imports[j["unit"]][2]["results"]]
        summaries[str(seed)] = grouped_actor_verdict(rows)
    return summaries


if __name__ == "__main__":
    main()
