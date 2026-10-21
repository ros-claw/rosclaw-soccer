"""Read-only comparison of two actual cold-restored episode20 PPO executions."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for key in ("left", "right", "source-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("unused paired proof output required")
    roots = [a.left, a.right]
    reports = [json.loads((x / "complete.json").read_text()) for x in roots]
    if a.left.resolve() == a.right.resolve():
        raise ValueError("independent executions required")
    pins = {}
    head = subprocess.check_output(
        ["git", "-C", str(a.source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(["git", "-C", str(a.source_root), "status", "--porcelain"]):
        raise ValueError("unchanged actually executed source required")
    for root, r in zip(roots, reports, strict=True):
        declaration = json.loads((root / "declaration.json").read_text())
        if (
            r["status"] != "NATIVE_OPTION_BOUNDARY_ATTEMPT_REQUIRES_PAIRED_COMPARISON"
            or r["resume_mode"] != "qualify21"
            or r["end_episode"] != 21
            or r["new_optimizer_updates"] != 16
            or len(r["results"]) != 1
            or r["results"][0]["episode"] != 20
            or r["counts"]["optimizer_completed"] != 336
            or declaration["source_head"] != head
        ):
            raise ValueError("completed actual single resumed episode and16updates required")
        pins.update(r["input_file_hashes"])
        pins[str(root / "complete.json")] = digest(root / "complete.json")
        pins[str(root / "declaration.json")] = digest(root / "declaration.json")
    if reports[0]["resume_checkpoint_hash"] != reports[1]["resume_checkpoint_hash"]:
        raise ValueError("same actual parent required")
    qualified = {}
    for relative in (
        "scripts/rsi_train_native_contact_course.py",
        "src/rosclaw_soccer/rsi/contact_option_resume.py",
        "src/rosclaw_soccer/rsi/contact_option_rollout.py",
        "src/rosclaw_soccer/rsi/contact_option_credit.py",
        "src/rosclaw_soccer/rsi/native_contact_ppo.py",
        "src/rosclaw_soccer/rsi/native_contact_course.py",
        "src/rosclaw_soccer/rsi/native_recovery_tail.py",
    ):
        path = a.source_root / relative
        h = digest(path)
        if any(r["input_file_hashes"].get(str(path)) != h for r in reports):
            raise ValueError("same before-import source commitment required")
        qualified[relative] = h
    check_pins(pins)
    import numpy as np
    import torch

    def compare(x, y, where):
        if type(x) is not type(y):
            raise ValueError("state type differs: " + where)
        if isinstance(x, torch.Tensor):
            if x.dtype != y.dtype or not torch.equal(x, y):
                raise ValueError("tensor differs: " + where)
        elif isinstance(x, dict):
            if set(x) != set(y):
                raise ValueError("state keys differ: " + where)
            for k in x:
                compare(x[k], y[k], where + "/" + str(k))
        elif isinstance(x, (tuple, list)):
            if len(x) != len(y):
                raise ValueError("state lengths differ: " + where)
            for i, (v, w) in enumerate(zip(x, y, strict=True)):
                compare(v, w, where + "/" + str(i))
        elif x != y:
            raise ValueError("value differs: " + where)

    for filename, key in (("rollout.npz", "raw_hash"), ("recovery.npz", "recovery_raw_hash")):
        arrays = []
        for root, r in zip(roots, reports, strict=True):
            path = root / "episode20" / filename
            if digest(path) != r["results"][0][key]:
                raise ValueError("actual raw identity changed")
            pins[str(path)] = digest(path)
            with np.load(path, allow_pickle=False) as z:
                arrays.append({k: z[k].copy() for k in z.files})
        if set(arrays[0]) != set(arrays[1]) or any(
            not np.array_equal(arrays[0][k], arrays[1][k]) for k in arrays[0]
        ):
            raise ValueError("actual cold-restored trajectory differs: " + filename)
    compare(reports[0]["results"], reports[1]["results"], "all episode metrics/losses")
    source_paths = [
        Path(f)
        for f, h in pins.items()
        if h == reports[0]["resume_checkpoint_hash"] and f.endswith("native-contact-episode20.pt")
    ]
    if len(source_paths) != 1:
        raise ValueError("one actual original pilot checkpoint required")
    original = torch.load(source_paths[0], map_location="cpu", weights_only=True)
    for episode in (20, 21):
        states = []
        for root, r in zip(roots, reports, strict=True):
            ck = next(x for x in r["checkpoints"] if x["episode"] == episode)
            path = root / ck["file"]
            if digest(path) != ck["hash"]:
                raise ValueError("safe checkpoint identity changed")
            pins[str(path)] = ck["hash"]
            states.append(torch.load(path, map_location="cpu", weights_only=True))
        compare(states[0], states[1], "full state episode" + str(episode))
        if episode == 20:
            for key in (
                "policy_state_dict",
                "optimizer_state_dict",
                "torch_rng",
                "cuda_rng",
                "counts",
            ):
                compare(states[0][key], original[key], "restored pilot/" + key)
    check_pins(pins)
    write_once(
        a.output,
        dict(
            status="NATIVE_OPTION_RESUMED_EPISODE_EXACT_NOT_TASK_PASS",
            input_file_hashes=pins,
            qualified_source_hashes=qualified,
            resume_checkpoint_hash=reports[0]["resume_checkpoint_hash"],
            all_state_and_trajectory_exact=True,
            actual_native_steps=sum(r["native_steps"] for r in reports),
            actual_new_optimizer_updates=32,
            new_comparison_physics=0,
            new_comparison_NN=0,
            independent_repetition_is_not_two_distinct_training_generations=True,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    print("paired actual resumed trajectory/model/Adam/RNG/update comparison exact", flush=True)


if __name__ == "__main__":
    main()
