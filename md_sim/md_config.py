from pathlib import Path

# --- Simulation Parameters ---
PARAMS = {
    # System
    "input_structure": "POSCAR",
    "temp_kelvin": 800,
    
    # Model
    "model_type": "large",
    "device": "cpu",  # Change to "cuda" for GPU
    
    # Dynamics
    "dt_fs": 0.5,
    "friction": 0.02,
    "n_steps": 10000,
    "log_interval": 1,
    
    # Output Files
    "summary_csv": Path("md_summary.csv"),
    "atoms_csv": Path("md_atoms.csv"),
    "trajectory_file": "large_traj.xyz",

    "make_plots": True,
    "compare_csvs": [],
}