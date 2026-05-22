"""Load simulation settings from JSON config files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

PATH_KEYS = frozenset({
    "summary_csv",
    "atoms_csv",
    "validation_output_csv",
    "output_dir",
})

BOOLEAN_KEYS = frozenset({
    "stationary",
    "zero_rotation",
    "fix_com",
    "make_plots",
    "run_validation",
    "run_energy",
    "run_forces",
    "run_rdf",
    "run_adf",
    "run_cv",
    "run_msd",
    "run_vacf",
    "run_vdos",
    "write_trajectory",
    "write_log",
    "plot_velocity_histogram",
    "plot_structural_overlays",
    "plot_vacf_overlay",
    "plot_vdos_overlay",
    "plot_vdos_normalized",
    "plot_parity",
    "plot_thermo",
})

DEFAULT_CONFIG_NAMES = {
    "simulation": "simulation.json",
    "relaxation": "relaxation.json",
}


def default_config_path(name: str = "simulation") -> Path:
    """Return the path to a bundled default config file."""
    if name not in DEFAULT_CONFIG_NAMES:
        raise ValueError(
            f"Unknown config profile {name!r}. "
            f"Choose from: {', '.join(DEFAULT_CONFIG_NAMES)}"
        )
    return _CONFIG_DIR / DEFAULT_CONFIG_NAMES[name]


def _coerce_value(key: str, value: Any) -> Any:
    if key in PATH_KEYS and value is not None:
        return Path(value)
    if key in BOOLEAN_KEYS and value is not None:
        return bool(value)
    return value


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply path/bool coercion after manual dict edits or merges."""
    return {key: _coerce_value(key, val) for key, val in raw.items()}


def load_config(
    config_path: str | Path | None = None,
    *,
    profile: str = "simulation",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load settings from a JSON file.

    If config_path is None, uses the bundled default for ``profile``
    (``"simulation"`` or ``"relaxation"``).
    """
    path = Path(config_path) if config_path is not None else default_config_path(profile)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a JSON object, got {type(raw).__name__}")

    cfg = normalize_config(raw)
    if overrides:
        cfg = normalize_config({**cfg, **overrides})
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Write config to JSON (paths become strings)."""
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


def add_config_cli(parser) -> None:
    """Register --config on an argparse parser."""
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config (default: bundled simulation.json or relaxation.json)",
    )
