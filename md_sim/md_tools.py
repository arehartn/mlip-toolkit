import csv
import numpy as np
from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.velocityverlet import VelocityVerlet
from ase.md.npt import NPT
from ase import units


def setup_atoms_and_calculator(structure_path, model_type="mace", model_variant="large", device="cpu"):
    """
    Reads the structure and attaches the requested MLIP calculator using lazy imports.
    """
    atoms = read(structure_path)
    model_variant = str(model_variant).strip()
    
    if model_type.lower() == "mace":
        # --- FIX 1: Support for your fine-tuned model ---
        if model_variant.endswith(".model"):
            from mace.calculators import MACECalculator
            print(f"Loading custom fine-tuned MACE model: {model_variant}...")
            # Use the base MACECalculator for local fine-tuned weights
            calc = MACECalculator(model_paths=model_variant, device=device)
        else:
            from mace.calculators import mace_mp 
            print(f"Initializing MACE Foundation ({model_variant}) calculator...")
            calc = mace_mp(model=model_variant, device=device)
        
    elif model_type.lower() == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        from chgnet.model.model import CHGNet
        print(f"Initializing CHGNet ({model_variant}) calculator...")
        
        if model_variant.lower() in ["default", "", "none"]:
            print("Loading standard default CHGNet weights...")
            chgnet_model = CHGNet.load()
        else:
            print(f"Attempting to load specific CHGNet weights: {model_variant}")
            chgnet_model = CHGNet.load(model_name=model_variant)
            
        calc = CHGNetCalculator(model=chgnet_model, use_device=device)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    atoms.calc = calc
    return atoms

def initialize_velocities(atoms, temperature_K, seed=42):
    # --- FIX 2: Apply the deterministic seed to the velocities ---
    rng = np.random.RandomState(seed)
    
    # ASE requires temperature in eV (temperature_K * units.kB) when setting velocities!
    MaxwellBoltzmannDistribution(
        atoms, 
        temperature_K=temperature_K * units.kB, 
        force_temp=True, 
        rng=rng
    )
    Stationary(atoms) # Removes center of mass translation

def setup_dynamics(atoms, cfg):
    """Sets up the ASE dynamics engine based on the chosen ensemble."""
    ensemble = cfg.get("ensemble", "NVT").upper()
    dt = cfg["dt_fs"] * units.fs
    temp = cfg["temp_kelvin"] * units.kB
    
    if ensemble == "NVE":
        print("Setting up NVE ensemble (Velocity Verlet)...")
        dyn = VelocityVerlet(atoms, timestep=dt)
        
    elif ensemble == "NVT":
        print(f"Setting up NVT ensemble (Langevin) at {cfg['temp_kelvin']}K...")
        dyn = Langevin(
            atoms,
            timestep=dt,
            temperature_K=temp,
            friction=cfg.get("friction", 0.01)
        )
        
    elif ensemble == "NPT":
        print(f"Setting up NPT ensemble at {cfg['temp_kelvin']}K and {cfg.get('pressure_bar', 1.0)} bar...")
        pressure = cfg.get("pressure_bar", 1.0) * units.bar
        dyn = NPT(
            atoms,
            timestep=dt,
            temperature_K=temp,
            externalstress=pressure,
            ttime=cfg.get("ttime_fs", 25.0) * units.fs,
            ptime=cfg.get("ptime_fs", 75.0) * units.fs
        )
        
    else:
        raise ValueError(f"Unknown ensemble requested: {ensemble}")
        
    return dyn

class MDLogger:
    def __init__(self, atoms, dynamics, params):
        self.atoms = atoms
        self.dyn = dynamics
        self.params = params
        self.summary_path = params["summary_csv"]
        self.atoms_path = params["atoms_csv"]
        self.traj_path = str(params["trajectory_file"])
        
        self.traj_writer = Trajectory(self.traj_path, 'w', self.atoms)
        self._init_csv_files()

    def _init_csv_files(self):
        with self.summary_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "time_ps", "energy_pot_eV", "energy_kin_eV", "energy_tot_eV", "temperature_K"])

        with self.atoms_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "atom_index", "x", "y", "z",
                "vx", "vy", "vz", "fx", "fy", "fz"
            ])

    def __call__(self):
        step = self.dyn.nsteps
        time_ps = step * self.params["dt_fs"] / 1000.0
        
        forces = self.atoms.get_forces()
        positions = self.atoms.get_positions()
        velocities = self.atoms.get_velocities()
        
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        etot = epot + ekin
        temp = self.atoms.get_temperature()

        print(f"Step: {step:6d}  Time: {time_ps:8.3f} ps  E_tot: {etot:12.6f} eV  Temp: {temp:8.2f} K")

        with self.summary_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, time_ps, epot, ekin, etot, temp])

        with self.atoms_path.open("a", newline="") as f:
            writer = csv.writer(f)
            rows = []
            for i, ((x, y, z), (vx, vy, vz), (fx, fy, fz)) in enumerate(zip(positions, velocities, forces)):
                rows.append([step, i, x, y, z, vx, vy, vz, fx, fy, fz])
            writer.writerows(rows)

        self.atoms.set_array('velocities', velocities)
        self.traj_writer.write()