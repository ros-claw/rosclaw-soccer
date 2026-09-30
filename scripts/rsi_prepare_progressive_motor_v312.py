"""Freeze broader consumed learning bank and independently protected new head."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.progressive_motor_actor import make_model, update_from_physics
from rosclaw_soccer.rsi.progressive_motor_bank import build_bank
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "stable-root",
        "online-root",
        "bank-path",
        "preview-root",
        "validation-root",
        "fresh-root",
        "old-quarantine",
        "pool-ledger",
        "new-quarantine",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.new_quarantine.read_text())
    if (
        protocol["partition"] != "FRESH_QUARANTINED_NOT_OPENED"
        or protocol["data_generated"] is not False
        or protocol["labels_observed"] is not False
        or protocol["automatic_open_authorized"] is not False
    ):
        parser.error("new untouched quarantine must precede training")
    bank = build_bank(
        args.stable_root,
        args.online_root,
        args.bank_path,
        args.preview_root,
        args.validation_root,
        args.fresh_root,
        args.old_quarantine,
        args.pool_ledger,
    )
    parent = json.loads((args.stable_root / "stable_model.json").read_text())
    initial = make_model(
        parent,
        [r["observation"] for r in bank["courses"]],
        bank["protected_features"],
        learning_report_hash=bank["report_hash"],
        fresh_protocol_hash=hash_json(protocol),
    )
    updated = update_from_physics(initial, bank["samples"], bank["report_hash"])
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_once(args.output_root / "learning_bank.json", bank)
    write_once(args.output_root / "initial_model.json", initial)
    write_once(args.output_root / "replay_model.json", updated)
    print(
        json.dumps(
            dict(
                record_count=bank["record_count"],
                contexts=52,
                protected_observations=38,
                initial_model_hash=initial["model_hash"],
                replay_model_hash=updated["model_hash"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
