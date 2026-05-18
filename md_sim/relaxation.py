import argparse
import os
from ase.io import write
from ase.filters import FrechetCellFilter
from ase.optimize.precon import PreconLBFGS
from md_sim import md_tools
from md_sim.config import add_config_cli, load_config, normalize_config


def _resolve_config(config=None, config_path=None, config_overrides=None):
    if config is None:
        cfg = load_config(config_path, profile="relaxation")
    else:
        cfg = dict(config)
        if config_path is not None:
            cfg = {**load_config(config_path, profile="relaxation"), **cfg}
    if config_overrides:
        cfg.update(config_overrides)
    return normalize_config(cfg)


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
    )

    ucf = FrechetCellFilter(atoms)

    opt_kwargs = {}
    if cfg.get("write_trajectory", True):
        opt_kwargs["trajectory"] = f"{out_prefix}.traj"
    if cfg.get("write_log", True):
        opt_kwargs["logfile"] = f"{out_prefix}.log"

    opt = PreconLBFGS(ucf, **opt_kwargs)
    opt.run(fmax=cfg["fmax"])

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
