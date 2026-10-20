"""Authenticate all32x4 grouped on-policy episodes before allocating a learner."""

import hashlib
import json
from pathlib import Path


def load_grouped_motion(plan, bd, world, pins):
    import numpy as np

    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.full_entry_forecast import score_full_entry
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_student_rollout import validate_motion_recording
    from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, exact_array
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        plan["group_count"] != 4
        or plan["unique_contexts"] != 32
        or plan["actual_episode_executions"] != 128
        or plan["planned_native_steps"] != 768000
        or len(plan["jobs"]) != 16
        or len({j["unit"] for j in plan["jobs"]}) != 16
        or len({j["output"] for j in plan["jobs"]}) != 16
    ):
        raise ValueError("whole matched four-group declaration required")
    train = sorted(i for i, r in enumerate(bd["rows"]) if r["split"] == "train")
    if len(train) != 32:
        raise ValueError("original32 train contexts required")
    rows = {}
    for job in plan["jobs"]:
        wait_for_terminal_collection(job["unit"] + ".service", wait=True)
        root = Path(job["output"])
        d = json.loads((root / "declaration.json").read_text())
        c = json.loads((root / "complete.json").read_text())
        group = job["group"]
        if (
            group not in range(4)
            or job["part"] not in range(4)
            or job["indices"] != train[job["part"] :: 4]
            or d["indices"] != job["indices"]
            or d["group"] != group
            or c["group"] != group
            or d["episode_noise_seed_base"] != 536870912 + 10000 * group
            or d["behavior_checkpoint_hash"] != plan["anchor_checkpoint_hash"]
            or c["behavior_checkpoint_hash"] != plan["anchor_checkpoint_hash"]
            or d["source_head"] != plan["source_head"]
            or c["source_head"] != plan["source_head"]
            or c["status"] != "FOUR_GROUP_CURRENT_ACTOR_EPISODES_REPLAY_VERIFIED"
            or c["new_native_steps"] != 48000
            or c["optimizer_steps"] != 0
            or [r["course_index"] for r in c["result"]["results"]] != job["indices"]
        ):
            raise ValueError("closed same-actor/noise-group declaration differs")
        for k, h in d["input_file_hashes"].items():
            if k in pins and pins[k] != h:
                raise ValueError("conflicting input commitments")
            pins[k] = h
        for name in ("declaration.json", "complete.json"):
            pins[str(root / name)] = digest(root / name)
        for row in c["result"]["results"]:
            i = row["course_index"]
            if (
                i not in train
                or row["split"] != "train"
                or (i, group) in rows
                or row["seed"] != 536870912 + 10000 * group + i
            ):
                raise ValueError("unique training-only context/group/seed required")
            folder = root / f"course{i:02d}"
            if json.loads((folder / "complete.json").read_text()) != row:
                raise ValueError("per-episode receipt mismatch")
            for name, key in (("native.npz", "native_hash"), ("sampling.npz", "sampling_hash")):
                if digest(folder / name) != row[key]:
                    raise ValueError("raw episode bytes differ")
                pins[str(folder / name)] = row[key]
            pins[str(folder / "complete.json")] = digest(folder / "complete.json")
            with np.load(folder / "native.npz", allow_pickle=False) as z:
                trace = {k: z[k].copy() for k in z.files}
            validate_motion_recording(trace, nbody=world.nbody)
            with np.load(folder / "sampling.npz", allow_pickle=False) as z:
                sample = {
                    k: z[k].copy() for k in ("raw", "mean", "std", "log_prob", "rewards", "noise")
                }
            if any(not np.isfinite(v).all() or v.dtype != np.float32 for v in sample.values()):
                raise ValueError("complete finite actual noise/action/reward tapes required")
            if (
                any(sample[k].shape != (300, 64) for k in ("raw", "mean", "std", "noise"))
                or sample["log_prob"].shape != (300,)
                or sample["rewards"].shape != (300,)
                or float(sample["rewards"].sum(dtype=np.float64)) != row["sum_reward"]
            ):
                raise ValueError("full sampling/reward horizon differs")
            score = score_full_entry(world, {k: v[20:] for k, v in trace.items()})
            prefix = diagnose_entry_prefix(world, {k: v[:20] for k, v in trace.items()})
            if score != row["score"] or prefix != row["prefix_diagnostics"]:
                raise ValueError("primitive gate differs from declared outcome")
            proof = row["replay"]
            if (
                row["native_steps"] != 6000
                or proof["additional_native_steps"] != 3000
                or proof["student_neural_and_muscle_native_bit_exact"] is not True
                or row["regenerated_noise_and_prequant_likelihood_bit_exact"] is not True
                or proof["intention_network_calls"] != 600
                or proof["muscle_decoder_calls"] != 280
                or proof["teacher_encoder_calls"] != 0
                or proof["teacher_planner_calls"] != 0
                or proof["hardware_authorized"] is not False
                or proof["promotion_authorized"] is not False
            ):
                raise ValueError("independent complete neural/noise/muscle/native replay required")
            rows[i, group] = dict(
                features=trace["student_features"],
                sample=sample,
                prefix={k: v[:20].copy() for k, v in trace.items() if k in PHYSICAL_FIELDS},
                outcome=row,
            )
    if set(rows) != {(i, g) for i in train for g in range(4)}:
        raise ValueError("entire32x4 required, never remove failure groups")
    for i in train:
        for g in range(1, 4):
            for k in PHYSICAL_FIELDS:
                exact_array(rows[i, g]["prefix"][k], rows[i, 0]["prefix"][k])
            exact_array(rows[i, g]["features"][:21], rows[i, 0]["features"][:21])
            exact_array(rows[i, g]["sample"]["mean"][:21], rows[i, 0]["sample"]["mean"][:21])
        tapes = {
            hashlib.sha256(rows[i, g]["sample"]["noise"][20:].tobytes()).hexdigest()
            for g in range(4)
        }
        if len(tapes) != 4:
            raise ValueError("independent declared group noise tapes required")
    check_pins(pins)
    return dict(
        course_indices=train,
        features=np.stack([np.stack([rows[i, g]["features"] for g in range(4)]) for i in train]),
        **{
            k: np.stack([np.stack([rows[i, g]["sample"][k] for g in range(4)]) for i in train])
            for k in ("raw", "mean", "std", "log_prob", "rewards")
        },
        outcomes=[rows[i, g]["outcome"] for i in train for g in range(4)],
        input_file_hashes=pins,
        reviewed_native_steps=768000,
        new_native_steps=0,
    )
