"""Build only measured TRAIN precontact resets; no actor labels or gradient."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes-root", type=Path, required=True)
    p.add_argument("--episodes-hash", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("new output and exact frozen builder required")
    receipt = a.episodes_root / "complete.json"
    if digest(receipt) != a.episodes_hash:
        raise ValueError("actual completed TRAIN episodes required")
    d = json.loads(receipt.read_text())
    if (
        d.get("status") != "TRAIN8_NATIVE_CONTACT_EPISODES_NOT_LEARNED_SKILL"
        or d.get("independent_episodes") != 64
        or d.get("source_split") != "train"
        or d.get("representative_full_repeats_exact") is not True
        or len(d.get("results", [])) != 72
    ):
        raise ValueError("all64TRAINepisodes and8representative full repeats required")
    pins = dict(d["input_file_hashes"])
    pins[str(receipt)] = a.episodes_hash
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    import numpy as np

    from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
    from rosclaw_soccer.rsi.precontact_curriculum import select_precontact_frames

    starts_path = next(Path(k) for k in pins if k.endswith("/train-starts.npz"))
    with np.load(starts_path, allow_pickle=False) as z:
        starts = {k: z[k].copy() for k in z.files}
    rows = []
    selections = []
    source_counts = {s: 0 for s in TRAIN_MOTION_NAMES}
    for world in range(64):
        trial = next(r for r in d["results"] if r["world"] == world and r["repeat"] == 0)
        if trial["source"] != TRAIN_MOTION_NAMES[world % 8]:
            raise ValueError("source split or balanced world identity differs")
        path = a.episodes_root / f"world{world}-repeat0.npz"
        if digest(path) != trial["raw_hash"]:
            raise ValueError("actual raw episode bytes differ")
        pins[str(path)] = trial["raw_hash"]
        with np.load(path, allow_pickle=False) as z:
            raw = {k: z[k].copy() for k in z.files}
        if world < 8:
            twin = next(r for r in d["results"] if r["world"] == world and r["repeat"] == 1)
            tp = a.episodes_root / f"world{world}-repeat1.npz"
            if digest(tp) != twin["raw_hash"]:
                raise ValueError("representative repeat bytes differ")
            pins[str(tp)] = twin["raw_hash"]
            with np.load(tp, allow_pickle=False) as z:
                if set(z.files) != set(raw) or any(
                    not np.array_equal(v, z[k]) for k, v in raw.items()
                ):
                    raise ValueError("whole native NN repeat differs")
        selection = select_precontact_frames(
            source=trial["source"],
            foot_force=raw["foot_force"],
            nonfoot_force=raw["nonfoot_force"],
            ball_to_feet=raw["ball_to_feet"],
            native_qpos=raw["native_qpos_before"],
            expected_foot=int(starts["expected_foot"][world]),
            pelvis_qadr=7,
        )
        selections.append(dict(world=world, **selection))
        for case in selection["cases"]:
            f = case["frame"]
            row = dict(
                world=world,
                source_id=world % 8,
                reference_frame=f,
                lead_controls=case["lead_controls"],
                qpos=raw["native_qpos_before"][f],
                qvel=raw["native_qvel_before"][f],
                qacc_warmstart=raw["qacc_warmstart_before"][f],
                previous_action=raw["previous_action"][f],
                previous_previous_action=raw["previous_action"][f - 1] if f else np.zeros(29),
                phase_before=raw["phase_before"][f],
                observation=raw["observation"][f],
                goal_world=starts["goal_world"][world],
                origin=starts["origins"][world],
                expected_foot=starts["expected_foot"][world],
            )
            rows.append(row)
            source_counts[trial["source"]] += 1
    if not rows or any(v == 0 for v in source_counts.values()):
        raise ValueError(
            "all TRAIN sources need measured eligible reset states; no silent omissions"
        )
    check_pins(pins)
    a.output.mkdir()
    bank = a.output / "precontact-resets.npz"
    np.savez_compressed(bank, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    write_once(
        a.output / "complete.json",
        dict(
            status="TRAIN8_PRECONTACT_RESET_BANK_NOT_TRAINED_OR_RESTORE_QUALIFIED",
            input_file_hashes=pins,
            source_head=a.expected_head,
            ball_placement_protocol=d.get("ball_placement_protocol", "legacy-subset-full-v0"),
            bank_hash=digest(bank),
            states=len(rows),
            selections=selections,
            source_counts=source_counts,
            lead_controls=[10, 20, 40, 80],
            no_synthetic_physical_states=True,
            no_dev_in_gradients=True,
            expert_labels_provided=False,
            historical_contact_free=True,
            historical_contact_threshold_n=1.0,
            restore_and_fresh_trajectory_qualification_required=True,
            new_physics_steps=0,
            new_neural_calls=0,
            optimizer_updates=0,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
