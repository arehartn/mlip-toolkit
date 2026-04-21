from pathlib import Path

# --- Simulation Parameters ---
PARAMS = {
    # System
    "input_structure": "POSCAR",
    "temp_kelvin": 800,
    
    "model_type": "mace",       # "mace" or "chgnet"
    "model_variant": "large",   # "large", "medium-0b", "default", etc.
    "device": "cpu",            # Change to "cuda" for GPU
    
    # Dynamics
    "dt_fs": 0.5,
    "friction": 0.02,
    "n_steps": 10000,
    "log_interval": 1,
    
    # Output Files
    "summary_csv": Path("md_summary.csv"),
    "atoms_csv": Path("md_atoms.csv"),
    "trajectory_file": "large_traj.traj",

        # COM / momentum control
    "stationary": True,      # zero net linear momentum at t=0
    "zero_rotation": False,  # zero net angular momentum at t=0 (clusters/molecules)
    "fix_com": True,        # FixCom constraint: keep COM fixed during dynamics

    "make_plots": False,
    "compare_csvs": [],
    "run_validation": False,
    "reference_traj_file": None,
    "equilibration_frame": 0,
}