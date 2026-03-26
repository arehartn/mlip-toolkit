import os
from ase.io import write
from ase.filters import FrechetCellFilter
from ase.optimize.precon import PreconLBFGS
from md_sim import md_tools

DEFAULT_PARAMS = {
    "input_structure": "POSCAR",
    "model_type": "mace",
    "model_variant": "large",
    "device": "cpu",
    "fmax": 0.01,
    "output_prefix": "relaxed_structure",
}

def run(config_overrides=None):
    cfg = DEFAULT_PARAMS.copy()
    if config_overrides:
        cfg.update(config_overrides)

    poscar_path = cfg["input_structure"]
    out_prefix = cfg["output_prefix"]
    
    if not os.path.exists(poscar_path):
        raise FileNotFoundError(f"Structure not found: {poscar_path}")

    print(f"--- Starting Relaxation ---")
    print(f"Input: {poscar_path}")
    print(f"Model: {cfg['model_type']} ({cfg['model_variant']}) on {cfg['device'].upper()}")

    # Initialize atoms and calculator
    atoms = md_tools.setup_atoms_and_calculator(
        structure_path=poscar_path,
        model_type=cfg["model_type"],
        model_variant=cfg["model_variant"],
        device=cfg["device"]
    )

    # Filter to allow unit cell volume/shape to relax
    ucf = FrechetCellFilter(atoms)
    
    # Setup PreconLBFGS
    opt = PreconLBFGS(
        ucf, 
        trajectory=f"{out_prefix}.traj", 
        logfile=f"{out_prefix}.log"
    )
    
    # Run optimization
    opt.run(fmax=cfg["fmax"])
    
    # Extract results
    e_per_atom = atoms.get_potential_energy() / len(atoms)
    out_file = f"{out_prefix}_relaxed.vasp"
    write(out_file, atoms, format="vasp")
    
    print(f"Done. Final energy: {e_per_atom:.5f} eV/atom\n")
    return e_per_atom