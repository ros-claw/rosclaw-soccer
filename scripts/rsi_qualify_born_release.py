"""Run the explicit one-context14-action prospective born-release protocol."""

import argparse
from pathlib import Path

from scripts.rsi_born_release_inputs import prepare
from scripts.rsi_qualify_navigation_recovery import run_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config)
    if Path(__file__).resolve() != p.source / "scripts/rsi_qualify_born_release.py":
        raise ValueError("exact pinned born-release entry required")
    run_bank(p, preflight=args.preflight, born_release=True)


if __name__ == "__main__":
    main()
