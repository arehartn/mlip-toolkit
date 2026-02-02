import md_config
import md_tools

def main():
    # 1. Load Configuration
    cfg = md_config.PARAMS
    print(f"--- Starting MD Simulation: {cfg['temp_kelvin']}K ---")

    # 2. Setup System
    atoms = md_tools.setup_atoms_and_calculator(
        structure_path=cfg["input_structure"],
        model_type=cfg["model_type"],
        device=cfg["device"]
    )

    # 3. Initialize Physics (Velocities)
    md_tools.initialize_velocities(atoms, cfg["temp_kelvin"])

    # 4. Setup Dynamics Engine (Langevin)
    dyn = md_tools.setup_dynamics(
        atoms, 
        temperature_K=cfg["temp_kelvin"], 
        dt_fs=cfg["dt_fs"], 
        friction=cfg["friction"]
    )

    # 5. Attach Logger
    # We instantiate the Logger class and pass the simulation objects
    logger = md_tools.MDLogger(atoms, dyn, cfg)
    dyn.attach(logger, interval=cfg["log_interval"])

    # 6. Run
    print("Running dynamics...")
    dyn.run(cfg["n_steps"])
    print("Simulation Complete.")

if __name__ == "__main__":
    main()