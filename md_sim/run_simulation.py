import random
import numpy as np
import torch
from md_sim import md_config
from md_sim import md_tools
from md_sim import md_plotting
def lock_random_seeds(seed=42):
    """Locks down all sources of randomness for exact reproducibility."""
    # Standard Python and Numpy
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        # Force strict determinism on the GPU 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run(config_overrides=None):
    """
    Main execution function.
    config_overrides: A dictionary containing ANY parameter you want to change.
    """
    # 1. Start with defaults from md_config.py
    cfg = md_config.PARAMS.copy()

    # 2. Update with your custom settings from the runner script
    if config_overrides:
        cfg.update(config_overrides)

    print(f"--- Starting MD Simulation ---")
    print(f"Temp: {cfg['temp_kelvin']}K | Friction: {cfg['friction']} | dt: {cfg['dt_fs']}fs")
    print(f"Model type: {cfg['model_type']} | Model variant: {cfg['model_variant']}")

    # --- 3. LOCK SEEDS FOR REPRODUCIBILITY ---
    simulation_seed = cfg.get("seed", 42)
    lock_random_seeds(simulation_seed)
    print(f"Random seed locked to: {simulation_seed}")
    # -----------------------------------------

    # 4. Setup System
    atoms = md_tools.setup_atoms_and_calculator(
        structure_path=cfg["input_structure"],
        model_type=cfg["model_type"],
        model_variant=cfg.get("model_variant", "large"),
        device=cfg["device"]
    )

    # 5. Initialize Physics
    md_tools.initialize_velocities(atoms, cfg["temp_kelvin"])

    # 6. Setup Dynamics Engine (The Thermostat)
    # Passed the specific arguments that md_tools expects
    dyn = md_tools.setup_dynamics(
        atoms=atoms,
        temperature_K=cfg["temp_kelvin"],
        dt_fs=cfg["dt_fs"],
        friction=cfg["friction"]
    )

    # 7. Attach Logger
    # The logger uses 'cfg' to decide where to save files
    logger = md_tools.MDLogger(atoms, dyn, cfg)
    dyn.attach(logger, interval=cfg["log_interval"])

    # 8. Run
    dyn.run(cfg["n_steps"])
    print("Simulation Complete.")

    if cfg.get("make_plots", False):
        md_plotting.generate_plots(cfg)

if __name__ == "__main__":
    run()
