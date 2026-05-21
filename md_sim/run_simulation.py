import argparse
import random
import numpy as np
import torch
from md_sim.config import add_config_cli, load_config, normalize_config
from md_sim import md_tools
from md_sim import md_plotting
from md_sim import validation

def lock_random_seeds(seed=42):
    """Locks down all sources of randomness for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_config(config=None, config_path=None, config_overrides=None):
    if config is None:
        cfg = load_config(config_path, profile="simulation")
    else:
        cfg = dict(config)
        if config_path is not None:
            cfg = {**load_config(config_path, profile="simulation"), **cfg}
    if config_overrides:
        cfg.update(config_overrides)
    return normalize_config(cfg)


def run(config=None, config_path=None, config_overrides=None):
    """
    Main execution function.

    Parameters
    ----------
    config : dict, optional
        Full settings dict (e.g. from load_config).
    config_path : str or Path, optional
        Path to a JSON config file. Used when ``config`` is None, or merged
        under ``config`` when both are given.
    config_overrides : dict, optional
        Per-run overrides applied last.
    """
    cfg = _resolve_config(config, config_path, config_overrides)

    print(f"--- Starting MD Simulation ---")
    print(f"Temp: {cfg['temp_kelvin']}K | Friction: {cfg['friction']} | dt: {cfg['dt_fs']}fs")
    print(f"Model type: {cfg['model_type']} | Model variant: {cfg['model_variant']}")

    simulation_seed = cfg.get("seed", 42)
    lock_random_seeds(simulation_seed)
    print(f"Random seed locked to: {simulation_seed}")

    atoms = md_tools.setup_atoms_and_calculator(
        structure_path=cfg["input_structure"],
        model_type=cfg["model_type"],
        model_variant=cfg.get("model_variant", "large"),
        device=cfg["device"],
        head=cfg.get("head", None),
    )

    md_tools.initialize_velocities(
        atoms,
        cfg["temp_kelvin"],
        stationary=cfg.get("stationary", True),
        zero_rotation=cfg.get("zero_rotation", False),
    )

    if cfg.get("fix_com", False):
        md_tools.apply_fix_com(atoms)
        print("Center-of-mass constraint (FixCom) applied.")

    dyn = md_tools.setup_dynamics(
        atoms=atoms,
        temperature_K=cfg["temp_kelvin"],
        dt_fs=cfg["dt_fs"],
        friction=cfg["friction"],
    )

    logger = md_tools.MDLogger(atoms, dyn, cfg)
    dyn.attach(logger, interval=cfg["log_interval"])

    dyn.run(cfg["n_steps"])
    print("Simulation Complete.")

    if cfg.get("make_plots", False):
        md_plotting.generate_plots(cfg)

    if cfg.get("run_validation", False):
        print("\n--- Transitioning to Validation Phase ---")
        validation.run(config=cfg)


def main():
    parser = argparse.ArgumentParser(description="Run an MLIP molecular dynamics simulation.")
    add_config_cli(parser)
    args = parser.parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()
