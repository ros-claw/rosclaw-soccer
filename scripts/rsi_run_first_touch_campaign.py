"""Pinned SIM-only first-touch campaign entry; no shell or hardware commands."""

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
    if Path(__file__).resolve() != source / "scripts/rsi_run_first_touch_campaign.py":
        raise ValueError("exact frozen source entry required")
    sys.path[:0] = [str(source / "src"), str(core / "src"), str(source)]
    from rosclaw_soccer.rsi.first_touch_campaign_worker import run

    try:
        run(args.config.resolve(), preflight=args.preflight)
    except (ValueError, TypeError, KeyError, FileNotFoundError):
        # A contract violation is not a transient crash: systemd must not
        # retry it into apparent success. Full traceback remains in the log.
        import traceback

        traceback.print_exc()
        raise SystemExit(78) from None


if __name__ == "__main__":
    main()
