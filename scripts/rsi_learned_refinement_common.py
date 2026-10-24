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
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import load_onpolicy_refinement
    from scripts.rsi_joint_delivery_pilot_common import local_preflight
    from scripts.rsi_self_teacher_search_common import open_setup

    source, pins, scope, bank, pf, root = admitted
    setup, old = open_setup(scope, pf, pins, bank, root / "scene.xml")
    learned, _ = load_onpolicy_refinement(
        old,
        behavior_training_hash=scope["student_training_hash"],
        root=args.training_root,
        expected_receipt_hash=args.training_hash,
    )
    baseline = source["records"][0]
    variants, preflight = local_preflight(
        setup,
        baseline,
        dict(unsearched_TRAIN_offsets_forward_m=[-0.005, 0.005]),
        args.experience_root,
    )
    return setup, old, learned, [baseline["course"], *variants], preflight
