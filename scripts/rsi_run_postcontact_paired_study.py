"""Fixed bounded SIM control studies; not teacher admission or Fresh."""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source, core = Path(config["source_root"]), Path(config["core_root"])
    if Path(__file__).resolve() != source / "scripts/rsi_run_postcontact_paired_study.py":
        raise ValueError("exact frozen paired study source required")
    sys.path[:0] = [str(source / "src"), str(core / "src"), str(source)]
    from rosclaw_soccer.rsi.postcontact_paired_study import run

    run(args.config.resolve(), preflight=args.preflight)


if __name__ == "__main__":
    main()
