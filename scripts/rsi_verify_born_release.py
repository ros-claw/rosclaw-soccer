"""Whole14 born-lease Jacobian/raw audit then42000 independent native steps."""

import argparse
from pathlib import Path

from scripts.rsi_verify_navigation_recovery import run_verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--complete-file-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    run_verify(parser.parse_args(), born_release=True)


if __name__ == "__main__":
    main()
