"""Actual H300 native execution and independent replay of learned skill ownership."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "core-source", "router-training", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-inputs-hash", "router-unit"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--indices", nargs="+", type=int, required=True)
    a = p.parse_args()
    if len(a.indices) != len(set(a.indices)) or any(i not in range(40) for i in a.indices):
        raise ValueError("distinct original40 course indices required")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit, load_frozen_experts
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, cases = admit(a)
    wait_for_terminal_collection(a.router_unit, wait=True)
    tc = json.loads((a.router_training / "complete.json").read_text())
    td = json.loads((a.router_training / "declaration.json").read_text())
    if (
        tc["status"] != "MOTION_ROUTER_TRAINED_REQUIRES_EXAM"
        or tc["source_head"] != a.expected_head
        or tc["inputs_hash"] != a.expected_inputs_hash
        or tc["optimizer_steps"] != 2000
        or digest(a.router_training / "router.pt") != tc["checkpoint_hash"]
    ):
        raise ValueError("closed same-source ownership learner required")
    pins.update(td["input_file_hashes"])
    pins.update({str(f): digest(f) for f in a.router_training.iterdir() if f.is_file()})
    check_pins(pins)
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.motion_portfolio_native.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            inputs_hash=a.expected_inputs_hash,
            router_checkpoint_hash=tc["checkpoint_hash"],
            expert_ids=[e["id"] for e in spec["experts"]],
            indices=a.indices,
            skill_owner_selected_only_at_frame20=True,
            new_low_level_expert_training=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import torch

    from rosclaw_soccer.rsi.motion_capability_router import create_motion_capability_router
    from scripts.rsi_frozen_motion_portfolio import FrozenMotionPortfolio
    from scripts.rsi_motion_ppo_pilot import episodes
    from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, exact_array

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(a.router_training / "router.pt", map_location="cpu", weights_only=True)
    if checkpoint["inputs_hash"] != a.expected_inputs_hash or checkpoint[
        "expert_checkpoint_hashes"
    ] != [e["checkpoint_hash"] for e in spec["experts"]]:
        raise ValueError("trained router frozen memory roster differs")
    router = create_motion_capability_router(experts=5)
    router.load_state_dict(checkpoint["state_dict"], strict=True)
    router.eval().requires_grad_(False)
    portfolio = FrozenMotionPortfolio(router, load_frozen_experts(spec))
    a.world, a.model_root = Path(spec["world"]), Path(spec["model_root"])
    a.mode, a.seed = "evaluate", 92029
    result = episodes(a, portfolio, bd, pins)
    if len(portfolio.audits) != 2 * len(a.indices):
        raise ValueError("original and independent owner inference required for every course")
    for i in range(0, len(portfolio.audits), 2):
        left = {k: v for k, v in portfolio.audits[i].items() if k != "stream_index"}
        right = {k: v for k, v in portfolio.audits[i + 1].items() if k != "stream_index"}
        if left != right:
            raise ValueError("owner applicability and all prefix memories differ in replay")
    import hashlib

    import numpy as np

    bindings = []
    for offset, index in enumerate(a.indices):
        audit = portfolio.audits[2 * offset]
        owner = audit["selected_owner"]
        source_row, source_path = cases[owner][index]
        prefix_memories = []
        for memory in cases:
            with np.load(memory[index][1], allow_pickle=False) as z:
                prefix_memories.append(z["hidden_after"][:20].copy())
        expected_hash = hashlib.sha256(np.stack(prefix_memories, axis=1).tobytes()).hexdigest()
        if expected_hash != audit["prefix_all_expert_memory_sha256"]:
            raise ValueError("private warm memories differ from frozen source policies")
        folder = a.output / f"course{index:02d}"
        with (
            np.load(source_path, allow_pickle=False) as old,
            np.load(folder / "native.npz", allow_pickle=False) as new,
        ):
            for key in (
                *PHYSICAL_FIELDS,
                "student_features",
                "muscle_proprioception",
                "consumed_action_isaac",
            ):
                exact_array(new[key], old[key])
            for key in ("student_motion_token", "hidden_after", "value_estimate"):
                exact_array(new[key][20:], old[key][20:])
        score = result["results"][offset]["score"]
        for key in (
            "qualified",
            "first_foot_contact_global_frame",
            "forward_to_task_end_m",
            "lateral_to_task_end_m",
            "maximum_ball_vx_mps",
            "minimum_pelvis_z_m",
            "minimum_upright",
            "maximum_nonfoot_force_n",
        ):
            if score[key] != source_row["score"][key]:
                raise ValueError("original selected-memory task criterion changed")
        binding = dict(
            course_index=index,
            selected_memory=spec["experts"][owner]["id"],
            source_native_hash=digest(source_path),
            actual_physics_and_post_prefix_policy_source_bit_exact=True,
        )
        write_once(folder / "memory-binding.json", binding)
        bindings.append(binding)
    check_pins(pins)
    result.update(
        source_head=a.expected_head,
        owner_inference_and_prefix_memory_verified=True,
        portfolio_audits=portfolio.audits,
        actual_intention_network_calls_per_episode=380,
        selected_memory_source_bindings=bindings,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    write_once(a.output / "complete.json", result)


if __name__ == "__main__":
    main()
