#!/usr/bin/env python
"""
Run every *.json in the current directory.

Each JSON needs a top-level "job" field:
  "md"          - molecular dynamics
  "relaxation"  - structure relaxation
  "validation"  - trajectory validation (+ plots if make_plots is true)

Example:
    cd my_job_folder
    python /path/to/run_all.py
"""

import json
import sys
from pathlib import Path

from md_sim import validation, md_plotting
from md_sim.config import load_config
from md_sim.relaxation import run as run_relax
from md_sim.run_simulation import run as run_md


def run_validation(path: Path) -> None:
    cfg = load_config(path)
    validation.run(config=cfg)
    if cfg.get("make_plots", False):
        md_plotting.generate_plots(cfg)


def main() -> None:
    configs = sorted(Path.cwd().glob("*.json"))
    if not configs:
        sys.exit("No *.json files found. cd into your job folder first.")

    for path in configs:
        with path.open(encoding="utf-8") as f:
            job = json.load(f).get("job")

        if job not in ("md", "relaxation", "validation"):
            sys.exit(
                f'{path.name}: set "job" to "md", "relaxation", or "validation"'
            )

        print(f"\n--- {path.name} ({job}) ---")

        if job == "md":
            run_md(config_path=path)
        elif job == "relaxation":
            run_relax(config_path=path)
        else:
            run_validation(path)

    print(f"\nFinished {len(configs)} job(s).")


if __name__ == "__main__":
    main()