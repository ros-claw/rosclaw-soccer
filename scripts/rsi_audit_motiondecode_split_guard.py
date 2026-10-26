"""Run the real First Touch portfolio split screen without training or promotion."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.output.exists() or not a.dataset.is_dir():
        raise ValueError("unused external output and existing source dataset required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            dataset=str(a.dataset.resolve()),
            selected_per_category_and_foot=6,
            minimum_total_clips=1000,
            matching_filter_is_incomplete=True,
            optimization_updates=0,
            native_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        from rosclaw_soccer.training.first_touch_teacher_portfolio import (
            audit_motiondecode_football,
        )

        result = audit_motiondecode_football(
            a.dataset, selected_per_category_and_foot=6, minimum_total_clips=1000
        )
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="FIRST_TOUCH_DATA_SPLIT_SCREEN_COMPLETED_NOT_INDEPENDENCE_PROOF",
                source_head=a.expected_head,
                audit=result,
                optimization_updates=0,
                native_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
        print(json.dumps(dict(status="completed", total_clips=result["total_clips"])), flush=True)
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="DATA_SCREEN_FAILED_CLOSED",
                source_head=a.expected_head,
                error_type=type(error).__name__,
                reason=str(error),
                optimization_updates=0,
                native_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
