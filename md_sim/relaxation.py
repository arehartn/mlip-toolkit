import argparse
import os
import numpy as np
from ase.io import write
from ase.filters import FrechetCellFilter
from ase.optimize.lbfgs import LBFGS
from ase.optimize.precon import PreconLBFGS
from md_sim import md_tools
from md_sim.config import add_config_cli, load_config, normalize_config


def _resolve_config(config=None, config_path=None, config_overrides=None):
    cfg = load_config(profile="relaxation")
    if config_path is not None:
        cfg = {**cfg, **load_config(config_path)}
    if config is not None:
        cfg = {**cfg, **config}
    if config_overrides:
        cfg.update(config_overrides)
    cfg = normalize_config(cfg)
    cfg["fmax"] = float(cfg["fmax"])
    if cfg.get("max_steps") is not None:
        cfg["max_steps"] = int(cfg["max_steps"])
    return cfg


def run(config=None, config_path=None, config_overrides=None):
    cfg = _resolve_config(config, config_path, config_overrides)

    poscar_path = cfg["input_structure"]
    out_prefix = cfg["output_prefix"]

    if not os.path.exists(poscar_path):
        raise FileNotFoundError(f"Structure not found: {poscar_path}")

    print(f"--- Starting Relaxation ---")
    print(f"Input: {poscar_path}")
    print(f"Model: {cfg['model_type']} ({cfg['model_variant']}) on {cfg['device'].upper()}")

    atoms = md_tools.setup_atoms_and_calculator(
        structure_path=poscar_path,
        model_type=cfg["model_type"],
        model_variant=cfg["model_variant"],
        device=cfg["device"],
        head=cfg.get("head"),
    )

    relax_cell = cfg.get("relax_cell", False)
    fmax = cfg["fmax"]
    max_steps = cfg.get("max_steps", 500)
    optimizer = cfg.get("optimizer", "lbfgs").lower()

    opt_target = FrechetCellFilter(atoms) if relax_cell else atoms

    opt_kwargs = {}
    if cfg.get("write_trajectory", True):
        opt_kwargs["trajectory"] = f"{out_prefix}.traj"
    if cfg.get("write_log", True):
        opt_kwargs["logfile"] = f"{out_prefix}.log"

    if optimizer == "precon_lbfgs":
        opt = PreconLBFGS(opt_target, **opt_kwargs)
    else:
        opt = LBFGS(opt_target, **opt_kwargs)

    print(f"Optimizer: {optimizer} | fmax: {fmax} eV/Å | max_steps: {max_steps}"
          f" | relax_cell: {relax_cell}")
    opt.run(fmax=fmax, steps=max_steps)

    atomic_fmax = float(np.abs(atoms.get_forces()).max())
    print(f"Final max atomic force: {atomic_fmax:.4f} eV/Å")

    e_per_atom = atoms.get_potential_energy() / len(atoms)
    out_file = f"{out_prefix}_relaxed.vasp"
    write(out_file, atoms, format="vasp")

    print(f"Done. Final energy: {e_per_atom:.5f} eV/atom\n")
    return e_per_atom


def main():
    parser = argparse.ArgumentParser(description="Relax a structure with an MLIP.")
    add_config_cli(parser)
    args = parser.parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()
