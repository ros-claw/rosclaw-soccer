"""Read back one consumed PAiD router history, without physics or training."""

import argparse
import json
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import clean, source_pins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.source.is_symlink() or digest(a.source) != a.source_hash:
        raise ValueError("unused output and exact consumed source receipt required")
    source = json.loads(a.source.read_text())
    if source["arm"] != "progressive" or source["promotion_authorized"] is not False:
        raise ValueError("unpromoted learned router source required")
    declaration_path = a.source.parent / "declaration.json"
    if digest(declaration_path) != source["declaration_hash"]:
        raise ValueError("source declaration commitment mismatch")
    declaration = json.loads(declaration_path.read_text())
    fresh_root = Path(declaration["fresh_root"])
    fresh_receipt = fresh_root / "complete.json"
    if digest(fresh_receipt) != declaration["fresh_hash"]:
        raise ValueError("exact consumed source context declaration required")
    fresh_declaration = fresh_root / "declaration.json"
    fresh_metadata = json.loads(fresh_receipt.read_text())
    if digest(fresh_declaration) != fresh_metadata["declaration_hash"]:
        raise ValueError("exact fresh source declaration required")
    fresh = json.loads(fresh_declaration.read_text())
    first = source["worlds"][0]
    rows = [x for x in first["artifacts"] if x["raw_path"] == "case0-repeat0/kick2.npz"]
    if len(rows) != 1:
        raise ValueError("fixed third-kick history required")
    raw = a.source.parent / rows[0]["raw_path"]
    if raw.is_symlink() or digest(raw) != rows[0]["raw_hash"]:
        raise ValueError("exact source history required")
    pins = dict(source["input_file_hashes"])
    for path, h in (
        (a.source, a.source_hash),
        (declaration_path, source["declaration_hash"]),
        (fresh_receipt, declaration["fresh_hash"]),
        (fresh_declaration, fresh_metadata["declaration_hash"]),
        (raw, rows[0]["raw_hash"]),
    ):
        pins[str(path.resolve())] = h
    source_pins(pins)
    check_pins(pins)
    policy = [
        Path(x) for x in pins if x.endswith("/paid-reference-source-20261021/ckp/policy_30000.onnx")
    ]
    if len(policy) != 1:
        raise ValueError("one exact source-bound PAiD model required")
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.paid_ancestor_router_student import FrozenAncestralRouterActor
    from rosclaw_soccer.rsi.paid_fresh_evidence import load_fresh_actor

    actor, variant = load_fresh_actor(policy[0], fresh, "progressive")
    if type(actor) is not FrozenAncestralRouterActor or variant != 2:
        raise ValueError("exact frozen router actor required")
    with np.load(raw, allow_pickle=False) as f:
        obs, actions = f["NN_obs"].copy(), f["NN_action"].copy()
        before_h, before_c = f["NN_h_before"].copy(), f["NN_c_before"].copy()
        after_h, after_c = f["NN_h_after"].copy(), f["NN_c_after"].copy()
    count = len(obs)
    if not 1 <= count <= 300 or obs.shape != (count, 160) or actions.shape != (count, 29):
        raise ValueError("bounded complete actual history required")
    state = (torch.from_numpy(before_h[0].copy()), torch.from_numpy(before_c[0].copy()))
    matched = 0
    failure = None
    try:
        with torch.no_grad():
            for i in range(count):
                if not np.array_equal(state[0].numpy(), before_h[i]) or not np.array_equal(
                    state[1].numpy(), before_c[i]
                ):
                    raise ValueError(f"own recurrent state differs at {i}")
                action, state = actor(torch.from_numpy(obs[i]).reshape(1, 1, 160), state)
                if (
                    not np.array_equal(action[0, 0].numpy(), actions[i])
                    or not np.array_equal(state[0].numpy(), after_h[i])
                    or not np.array_equal(state[1].numpy(), after_c[i])
                ):
                    raise ValueError(f"actual actor output or memory differs at {i}")
                matched += 1
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="CONSUMED_ROUTER_HISTORY_EXACT"
            if failure is None and matched == count
            else "HISTORY_NOT_REPRODUCED",
            source_hash=a.source_hash,
            source_head=a.expected_head,
            raw_hash=rows[0]["raw_hash"],
            controls=count,
            matched=matched,
            runtime_failure=failure,
            native_steps=0,
            new_training_updates=0,
            fresh_scenarios=0,
            team_game_verified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
