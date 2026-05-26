#!/usr/bin/env python
"""
Run MD from a JSON config.

1. Edit CONFIG below.
2. Run from the directory that contains that JSON file:
       python run_md_job.py
"""

import sys
from pathlib import Path

from md_sim.run_simulation import run

# --- only edit this line ---
CONFIG = "nbh_700K.json"


def main() -> None:
    config = Path(CONFIG)
    if not config.is_file():
        sys.exit(f"ERROR: config not found: {config.resolve()}")
    run(config_path=config)


if __name__ == "__main__":
    main()
