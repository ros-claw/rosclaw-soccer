"""Run a predeclared three-G1 holdout portfolio in one physical world per case.

All paths are supplied by the operator.  This entry point only creates
SIM_ONLY evidence outside the source checkout; rendering is a separate step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosclaw_soccer.training.three_role_save_portfolio import (
    ThreeRoleSaveLane,
    ThreeRoleSavePortfolioConfig,
    run_three_role_save_portfolio_evidence,
)

HOLDOUT_LANES = (
    ThreeRoleSaveLane("inside-right", "INSIDE RIGHT · HIGH SAVE", 0.10, 0.05),
    ThreeRoleSaveLane("inside-center", "INSIDE CENTER · HIGH SAVE", -0.32, -0.28),
    ThreeRoleSaveLane("outside-left", "OUTSIDE LEFT · HIGH SAVE", -0.82, -0.65),
    ThreeRoleSaveLane("wide-left", "WIDE LEFT · HIGH SAVE", -1.05, -1.00),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "asset-root",
        "striker-actor",
        "goalkeeper-actor",
        "gmt-model",
        "gmt-skill",
        "output-dir",
        "source-checkout",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    report = run_three_role_save_portfolio_evidence(
        asset_root=args.asset_root,
        striker_actor_path=args.striker_actor,
        goalkeeper_actor_path=args.goalkeeper_actor,
        gmt_model_path=args.gmt_model,
        gmt_skill_path=args.gmt_skill,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
        config=ThreeRoleSavePortfolioConfig(lanes=HOLDOUT_LANES),
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "promotion_status": report["promotion_status"],
                "portfolio_gates": report["portfolio_gates"],
                "case_passed": {key: case["passed"] for key, case in report["cases"].items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
