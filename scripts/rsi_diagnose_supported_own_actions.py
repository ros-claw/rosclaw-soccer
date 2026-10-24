"""Recompute own parent counterfactual actions on audited candidate observations.

Never execute saved actions or borrow saved h/c. Requires all80 independent
whole-native replay first. This diagnostic adds no physics or training updates.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("own-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("producer-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    op = a.own_root / "complete.json"
    if a.output.exists() or op.is_symlink() or (a.own_root / "failed.json").exists():
        raise ValueError("closed whole own80 proof and unused diagnostic output required")
    own = json.loads(op.read_text())
    pins = dict(own["input_file_hashes"])
    pins[str(op)] = digest(op)
    sp, source = receipt(pins, own["producer_hash"])
    _, training = receipt(pins, own["training_hash"])
    if (
        own["status"] != "TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH"
        or source["source_head"] != a.producer_head
        or own["source_head"] != a.producer_head
        or source["checkpoint_kind"] != "support-masked-refinement-v1"
        or own["training_hash"] != source["training_hash"]
        or len(own["records"]) != 80
        or len(source["records"]) != 80
        or own["records"]
        != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
        or any(
            d[k] is not False
            for d in (own, source, training)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("same explicitly audited whole supported actor required")
    _, bank = receipt(pins, training["bank_hash"])
    for row in source["records"]:
        art = row["artifacts"]["A"]
        path = sp.parent / art["path"]
        if (
            path.is_symlink()
            or path.resolve().parent != (sp.parent / f"trial{row['index']}").resolve()
        ):
            raise ValueError("confined exact actual candidate neural trace required")
        pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    calls = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label
        from rosclaw_soccer.rsi.supported_action_effects import supported_action_effects
        from scripts.rsi_stroke_search_common import open_motor

        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        _, actor = open_motor(
            pins,
            own["training_hash"],
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="support-masked-refinement-v1",
        )
        scale = actor.refinement.parent.memory.parent.get_buffer("action_scale").numpy().copy()
        records = []
        for row in source["records"]:
            with np.load(sp.parent / row["artifacts"]["A"]["path"], allow_pickle=False) as z:
                raw = {k: z[k].copy() for k in z.files if k.startswith("NN_")}
            state = actor.initial_state()
            parents = []
            for i, o in enumerate(raw["NN_obs"]):
                if any(
                    not np.array_equal(t.numpy(), raw[k][i])
                    for t, k in zip(state, ("NN_h_before", "NN_c_before"), strict=True)
                ):
                    raise ValueError("own diagnostic prehistory differs from actual candidate")
                with torch.no_grad():
                    mean, state, e = actor.forward_with_evidence(
                        torch.from_numpy(o).reshape(1, 1, 160), state
                    )
                calls += 1
                if (
                    not np.array_equal(mean[0, 0].numpy(), raw["NN_action"][i])
                    or not np.array_equal(
                        e["support_activation"][0, 0].numpy(), raw["NN_support_activation"][i]
                    )
                    or any(
                        not np.array_equal(t.numpy(), raw[k][i])
                        for t, k in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
                    )
                ):
                    raise ValueError("own masked mean/support/state differs from physical witness")
                parents.append(e["refinement_parent_mean"][0, 0].numpy().copy())
            counterfactual = np.stack(parents)
            effect = supported_action_effects(
                raw["NN_support_activation"],
                raw["NN_action"],
                counterfactual,
                scale,
                first_foot_substep=row["outcome"]["incoming"]["A_first_foot_strike_substep"],
            )
            out = a.output / f"trial{row['index']}.npz"
            np.savez_compressed(out, parent_actions_on_candidate_states=counterfactual)
            record = dict(
                index=row["index"],
                course=row["course"],
                controlled=controlled_pass_label(row["outcome"]),
                passer_clean_foot=row["outcome"]["passer_clean_foot"],
                effect=effect,
                output_path=out.name,
                output_hash=digest(out),
            )
            records.append(record)
            print(json.dumps(record), flush=True)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="AUDITED_SUPPORTED_OWN_STATE_ACTION_EFFECTS_DIAGNOSED_NOT_NEW_PHYSICS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                own_native_hash=digest(op),
                training_hash=own["training_hash"],
                own_actor_decisions=calls,
                records=records,
                physical_steps_executed=0,
                optimizer_updates=0,
                offpolicy_world_success_claim=False,
                Fresh_verified=False,
                team_game_verified=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                own_actor_decisions=calls,
                physical_steps_executed=0,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
