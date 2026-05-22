#!/usr/bin/env python
"""
HPC entry point: load a JSON config and run MD (+ optional plots/validation).

Usage (from the directory that contains your JSON config):
    python run_md_job.py --config nbh_700K.json

sbatch should call this script; see submit.sbatch in this folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from md_sim import run_simulation
from md_sim.config import add_config_cli


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLIP MD using a JSON config (for HPC batch jobs).",
    )
    add_config_cli(parser)
    parser.add_argument(
        "--relax",
        action="store_true",
        help="Run structure relaxation instead of MD (uses relaxation.json profile if no --config).",
    )
    args = parser.parse_args()

    config_path = args.config
    if config_path is not None and not Path(config_path).is_file():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    if args.relax:
        from md_sim import relaxation

        relaxation.run(config_path=config_path)
    else:
        run_simulation.run(config_path=config_path)


if __name__ == "__main__":
    main()
