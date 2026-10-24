"""Admission/setup for a new-kind learned refinement; never relabel as old BC."""

import json
from pathlib import Path


def admit(args):
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from scripts.rsi_refinement_onpolicy_data import admit as admit_experience

    source, _, pins, scope, bank, pf, root = admit_experience(
        args.experience_root, args.own_root, args.own_hash, args.state_id
    )
    tp = args.training_root / "complete.json"
    if file_hash(tp) != args.training_hash:
        raise ValueError("exact completed learned training receipt required")
    trained = json.loads(tp.read_text())
    if trained["config"]["state_id"] != args.state_id:
        raise ValueError("same predeclared state/model required")
    for k, v in trained["input_file_hashes"].items():
        if k in pins and pins[k] != v:
            raise ValueError("conflicting learned ancestry")
        pins[k] = v
    for path in (tp, args.training_root / "declaration.json", Path(trained["checkpoint_path"])):
        pins[str(path)] = file_hash(path)
    return source, pins, scope, bank, pf, root


def open_models(args, admitted):
    from scripts.rsi_joint_delivery_pilot_common import local_preflight
    from scripts.rsi_self_teacher_search_common import open_setup

    source, pins, scope, bank, pf, root = admitted
    setup, old = open_setup(scope, pf, pins, bank, root / "scene.xml")
    learned, _ = open_learned_model(old, args, pins, scope)
    baseline = source["records"][0]
    variants, preflight = local_preflight(
        setup,
        baseline,
        dict(unsearched_TRAIN_offsets_forward_m=[-0.005, 0.005]),
        args.experience_root,
    )
    return setup, old, learned, [baseline["course"], *variants], preflight


def open_learned_model(old, args, pins, scope, *, _depth=0):
    from types import SimpleNamespace

    import torch

    from rosclaw_soccer.rsi import continuing_refinement_contract as continuing
    from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
    from rosclaw_soccer.rsi.refinement_continual_checkpoint import KIND, load_continual_refinement
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import (
        file_hash,
        load_onpolicy_refinement,
    )
    from scripts.rsi_native_fresh_common import receipt

    if _depth > 16 or file_hash(args.training_root / "complete.json") != args.training_hash:
        raise ValueError("bounded exact model ancestry required")
    trained = json.loads((args.training_root / "complete.json").read_text())
    if trained["checkpoint_kind"] not in (KIND, continuing.KIND):
        return load_onpolicy_refinement(
            old,
            behavior_training_hash=scope["student_training_hash"],
            root=args.training_root,
            expected_receipt_hash=args.training_hash,
        )
    prior_path, prior = receipt(pins, trained["prior_training_hash"])
    if trained["config"]["parent_generation_index"] != prior["config"]["generation_index"]:
        raise ValueError("model generation chain cannot skip or reset a parent")
    behavior, _ = open_learned_model(
        old,
        SimpleNamespace(
            training_root=prior_path.parent, training_hash=trained["prior_training_hash"]
        ),
        pins,
        scope,
        _depth=_depth + 1,
    )
    state = torch.load(Path(prior["checkpoint_path"]), map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(prior["config"]["seed"])
        critic = MotorCritic()
    critic.load_state_dict(state["critic"], strict=True)
    critic.eval().requires_grad_(False)
    return load_continual_refinement(
        behavior,
        critic,
        state["optimizer"],
        prior_training_hash=trained["prior_training_hash"],
        root=args.training_root,
        expected_receipt_hash=args.training_hash,
    )


def base_behavior_hash(pins, trained):
    from scripts.rsi_native_fresh_common import receipt

    current = trained
    for _ in range(17):
        if "behavior_training_hash" in current:
            return current["behavior_training_hash"]
        _, current = receipt(pins, current["prior_training_hash"])
    raise ValueError("bounded original behavior ancestry required")
