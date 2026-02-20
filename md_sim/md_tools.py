import csv
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from mace.calculators import mace_mp

def setup_atoms_and_calculator(structure_path, model_type="large", device="cpu"):
    """
    Reads the structure and attaches the MACE calculator.
    """
    atoms = read(structure_path)
    calc = mace_mp(model=model_type, device=device)
    atoms.calc = calc
    return atoms

def initialize_velocities(atoms, temperature_K):
    """
    Sets initial Maxwell-Boltzmann distribution and removes drift.
    """
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    Stationary(atoms)

def setup_dynamics(atoms, temperature_K, dt_fs, friction):
    """
    Initializes the Langevin dynamics engine.
    """
    dyn = Langevin(atoms,
                   timestep=dt_fs * units.fs,
                   temperature_K=temperature_K,
                   friction=friction)
    return dyn

class MDLogger:
    """
    Handles CSV initialization and per-step logging.
    """
    def __init__(self, atoms, dynamics, params):
        self.atoms = atoms
        self.dyn = dynamics
        self.params = params
        self.summary_path = params["summary_csv"]
        self.atoms_path = params["atoms_csv"]
        self.traj_path = params["trajectory_file"]
        
        self._init_csv_files()

    def _init_csv_files(self):
        with self.summary_path.open("w", newline="") as f:
            writer = csv.writer(f)
            # UPDATED: Added new energy headers
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
        
        # UPDATED: Calculate all three energies
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        etot = epot + ekin
        temp = self.atoms.get_temperature()

        print(f"Step: {step:6d}  Time: {time_ps:8.3f} ps  "
              f"E_tot: {etot:12.6f} eV  Temp: {temp:8.2f} K")

        with self.summary_path.open("a", newline="") as f:
            writer = csv.writer(f)
            # UPDATED: Save all three energies
            writer.writerow([step, time_ps, epot, ekin, etot, temp])

        with self.atoms_path.open("a", newline="") as f:
            writer = csv.writer(f)
            rows = []
            for i, ((x, y, z), (vx, vy, vz), (fx, fy, fz)) in enumerate(zip(positions, velocities, forces)):
                rows.append([step, i, x, y, z, vx, vy, vz, fx, fy, fz])
            writer.writerows(rows)

        self.atoms.set_array('velocities', velocities)
        comment = f"Time={time_ps:.3f}ps E_tot={etot:.6f}eV Temp={temp:.2f}K"
        write(self.traj_path, self.atoms, format='extxyz', append=True, comment=comment)