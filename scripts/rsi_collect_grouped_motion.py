"""Fresh four-noise groups under ONE fixed actor; no critic-derived advantage."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "core-source", "policy", "audit", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "expected-head",
        "expected-inputs-hash",
        "expected-policy-hash",
        "expected-audit-hash",
        "audit-unit",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--group", type=int, choices=range(4), required=True)
    p.add_argument("--indices", type=int, nargs="+", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, _ = admit(a)
    wait_for_terminal_collection(a.audit_unit)
    if (
        digest(a.audit / "complete.json") != a.expected_audit_hash
        or digest(a.policy) != a.expected_policy_hash
    ):
        raise ValueError("exact fully audited development anchor required")
    review = json.loads((a.audit / "complete.json").read_text())
    if (
        review["status"] != "ONLINE_THREE_ROUNDS_TENSOR_AND_OUTCOME_REPRODUCED"
        or review["seed"] != 29
        or review["rounds"][0]["checkpoint_hash"] != a.expected_policy_hash
        or review["rounds"][0]["success_indices"] != [0, 10, 15, 16]
    ):
        raise ValueError("explicit consumed-development4/40 anchor identity differs")
    if (
        not a.indices
        or len(set(a.indices)) != len(a.indices)
        or any(i not in range(40) or bd["rows"][i]["split"] != "train" for i in a.indices)
    ):
        raise ValueError("unique original train contexts only")
    for k, h in review["input_file_hashes"].items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting historical evidence")
        pins[k] = h
    pins[str(a.audit / "complete.json")] = a.expected_audit_hash
    pins[str(a.policy)] = a.expected_policy_hash
    sources = [
        Path(k).parent.parent
        for k in review["input_file_hashes"]
        if k.endswith("/scripts/rsi_online_kinematic_motion.py")
        and Path(k).parent.parent.name == "rosclaw-soccer-online-geometry-source-20261020"
    ]
    if (
        len(sources) != 1
        or subprocess.check_output(
            ["git", "-C", str(sources[0]), "rev-parse", "HEAD"], text=True
        ).strip()
        != "37174508238b36e08983f0002f2ef61c79b82b89"
    ):
        raise ValueError("original frozen shared actor implementation required")
    for folder in ("src", "scripts"):
        for old in (sources[0] / folder).rglob("*.py"):
            if digest(old) != digest(Path.cwd() / old.relative_to(sources[0])):
                raise ValueError("grouping cannot change original actor/physics/update helpers")
    check_pins(pins)
    a.output.mkdir(parents=True)
    a.seed = 536870912 + 10000 * a.group
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.grouped_actor_collection.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            behavior_checkpoint_hash=a.expected_policy_hash,
            group=a.group,
            episode_noise_seed_base=a.seed,
            indices=a.indices,
            expected_groups=4,
            actor_is_same_in_every_group=True,
            grouping_course_ids_are_learner_metadata_not_actor_input=True,
            anchor_selected_posthoc_on_consumed_development=True,
            initial_anchor_not_promoted=True,
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import torch

    from scripts.rsi_motion_ppo_pilot import episodes
    from scripts.rsi_online_kinematic_motion import load_model

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a.world, a.model_root, a.mode = Path(spec["world"]), Path(spec["model_root"]), "collect"
    policy = load_model(spec, a.policy).eval().requires_grad_(False)
    try:
        result = episodes(a, policy, bd, pins)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="FOUR_GROUP_CURRENT_ACTOR_EPISODES_REPLAY_VERIFIED",
                source_head=a.expected_head,
                group=a.group,
                behavior_checkpoint_hash=a.expected_policy_hash,
                result=result,
                new_native_steps=result["new_native_steps"],
                optimizer_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                partial_physical_charge_requires_child_artifact_audit=True,
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
