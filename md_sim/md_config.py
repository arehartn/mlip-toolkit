"""Backward-compatible defaults. Prefer editing configs/simulation.json or --config."""

from md_sim.config import load_config

PARAMS = load_config(profile="simulation")