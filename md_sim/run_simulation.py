from . import md_config
from . import md_tools

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

    # 3. Setup System
    atoms = md_tools.setup_atoms_and_calculator(
        structure_path=cfg["input_structure"],
        model_type=cfg["model_type"],
        device=cfg["device"]
    )

    # 4. Initialize Physics
    md_tools.initialize_velocities(atoms, cfg["temp_kelvin"])

    # 5. Setup Dynamics Engine (The Thermostat)
    # We pass the parameters directly from your config dictionary 'cfg'
    dyn = md_tools.setup_dynamics(
        atoms, 
        temperature_K=cfg["temp_kelvin"], 
        dt_fs=cfg["dt_fs"], 
        friction=cfg["friction"]
    )

    # 6. Attach Logger
    # The logger uses 'cfg' to decide where to save files
    logger = md_tools.MDLogger(atoms, dyn, cfg)
    dyn.attach(logger, interval=cfg["log_interval"])

    # 7. Run
    dyn.run(cfg["n_steps"])
    print("Simulation Complete.")

if __name__ == "__main__":
    run()