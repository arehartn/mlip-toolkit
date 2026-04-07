import numpy as np
import pandas as pd
from pathlib import Path
from ase.io import read
from ase.geometry import get_distances
from scipy.stats import wasserstein_distance
from scipy.signal import correlate

# --- 1. Structural Utilities ---
def _get_distance_distribution(frames, rmax=6.0, bins=100):
    """Calculates a global pseudo-RDF (pairwise distance distribution)."""
    hists = np.zeros(bins)
    for frame in frames:
        dists = frame.get_all_distances(mic=True)
        # Upper triangle to avoid double counting and self-distances (0.0)
        dists = dists[np.triu_indices_from(dists, k=1)]
        h, _ = np.histogram(dists, bins=bins, range=(0.1, rmax))
        hists += h
    return hists / np.sum(hists) # Normalize to probability

def _get_angle_distribution(frames, rcut=3.0, bins=90):
    """Calculates a global pseudo-ADF (angular distribution function)."""
    hists = np.zeros(bins)
    for frame in frames:
        # Get neighbor list using a simple distance matrix
        dists = frame.get_all_distances(mic=True)
        for i in range(len(frame)):
            neighbors = np.where((dists[i] < rcut) & (dists[i] > 0.1))[0]
            if len(neighbors) < 2: continue
            
            # Calculate angles between all pairs of neighbors around atom i
            vecs = frame.positions[neighbors] - frame.positions[i]
            # Apply MIC to vectors
            vecs = vecs - np.round(vecs / frame.cell.lengths()) * frame.cell.lengths()
            norms = np.linalg.norm(vecs, axis=1)
            vecs = vecs / norms[:, np.newaxis]
            
            # Dot products to angles
            cos_angles = np.clip(np.dot(vecs, vecs.T), -1.0, 1.0)
            angles = np.arccos(cos_angles[np.triu_indices_from(cos_angles, k=1)])
            
            h, _ = np.histogram(np.degrees(angles), bins=bins, range=(0, 180))
            hists += h
    # Normalize safely
    total = np.sum(hists)
    return hists / total if total > 0 else hists

# --- 2. Dynamical Utilities ---
def _calculate_msd(frames):
    """Calculates final Mean Squared Displacement from initial frame."""
    pos_0 = frames[0].get_positions()
    msds = []
    for frame in frames:
        # Assumes unwrapped coordinates or short runs where wrapping doesn't dominate
        diff = frame.get_positions() - pos_0
        msds.append(np.mean(np.sum(diff**2, axis=1)))
    return np.mean(msds)

def _calculate_vdos_spectrum(frames, dt_fs):
    """Calculates Vibrational Density of States via Velocity Autocorrelation."""
    vels = np.array([f.get_velocities() for f in frames]) # Shape: (steps, atoms, 3)
    steps, atoms, dims = vels.shape
    
    # Calculate Velocity Autocorrelation Function (VACF)
    vacf = np.zeros(steps)
    for i in range(atoms):
        for j in range(dims):
            v = vels[:, i, j]
            # Correlate velocity with itself
            corr = correlate(v, v, mode='full')
            vacf += corr[steps-1:] / np.arange(steps, 0, -1)
            
    vacf = vacf / vacf[0] # Normalize
    
    # Fourier Transform to get VDOS
    vdos = np.abs(np.fft.rfft(vacf))**2
    return vdos / np.sum(vdos) # Normalize to probability distribution

# --- 3. Main Validation Engine ---
def validate_trajectories(ref_path, pred_path, temp_k, dt_fs, burn_in=0):
    print(f"Loading {ref_path} and {pred_path}...")
    try:
        ref_frames = read(ref_path, index=f"{burn_in}:")
        pred_frames = read(pred_path, index=f"{burn_in}:")
    except Exception as e:
        print(f"Error loading trajectories: {e}")
        return None

    if not ref_frames or not pred_frames:
        print("Trajectories are empty or invalid.")
        return None

    kb = 8.617333262145e-5 # eV/K
    n_atoms = len(ref_frames[0])

    print(f"Validating {len(ref_frames)} AIMD frames vs {len(pred_frames)} MACE frames...")
    res = {}

    # 1-2. Energy Distributions (Thermodynamics)
    ref_epot = np.array([f.get_potential_energy() for f in ref_frames]) / n_atoms
    pred_epot = np.array([f.get_potential_energy() for f in pred_frames]) / n_atoms
    res["EMD_Epot_eV_atom"] = wasserstein_distance(ref_epot, pred_epot)

    ref_ekin = np.array([f.get_kinetic_energy() for f in ref_frames]) / n_atoms
    pred_ekin = np.array([f.get_kinetic_energy() for f in pred_frames]) / n_atoms
    res["EMD_Ekin_eV_atom"] = wasserstein_distance(ref_ekin, pred_ekin)

    # 3. Force Magnitude Distributions
    ref_f_mag = np.linalg.norm(np.vstack([f.get_forces() for f in ref_frames]), axis=1)
    pred_f_mag = np.linalg.norm(np.vstack([f.get_forces() for f in pred_frames]), axis=1)
    res["EMD_Forces_eV_A"] = wasserstein_distance(ref_f_mag, pred_f_mag)

    # 4-5. Structural Overlaps (RDF and ADF)
    ref_rdf = _get_distance_distribution(ref_frames)
    pred_rdf = _get_distance_distribution(pred_frames)
    res["EMD_RDF_Structure"] = wasserstein_distance(ref_rdf, pred_rdf)

    ref_adf = _get_angle_distribution(ref_frames)
    pred_adf = _get_angle_distribution(pred_frames)
    res["EMD_ADF_Structure"] = wasserstein_distance(ref_adf, pred_adf)

    # 6. Heat Capacity (Cv) Error via Energy Variance 
    # Cv = Variance(E) / (kb * T^2)
    ref_var_E = np.var(ref_epot * n_atoms) 
    pred_var_E = np.var(pred_epot * n_atoms)
    ref_cv = ref_var_E / (kb * temp_k**2 * n_atoms)
    pred_cv = pred_var_E / (kb * temp_k**2 * n_atoms)
    res["Cv_Error_eV_K_atom"] = abs(ref_cv - pred_cv)

    # 7. Mean Squared Displacement (MSD) Error
    ref_msd = _calculate_msd(ref_frames)
    pred_msd = _calculate_msd(pred_frames)
    res["MSD_Error_A2"] = abs(ref_msd - pred_msd)

    # 8. Vibrational Density of States (VDOS) Overlap
    try:
        ref_vdos = _calculate_vdos_spectrum(ref_frames, dt_fs)
        pred_vdos = _calculate_vdos_spectrum(pred_frames, dt_fs)
        res["EMD_VDOS_Spectrum"] = wasserstein_distance(ref_vdos, pred_vdos)
    except Exception as e:
        print(f"Warning: Could not compute VDOS (Check if velocities exist): {e}")

    # Output Results
    print("\n--- Validation Results ---")
    for k, v in res.items():
        print(f"{k:25s}: {v:.6f}")
        
    return res

def run(config_overrides=None):
    """Main entry point compatible with your runner script."""
    from md_sim.md_config import PARAMS
    cfg = PARAMS.copy()
    if config_overrides:
        cfg.update(config_overrides)

    ref_traj = cfg.get("reference_traj_file")
    pred_traj = cfg.get("trajectory_file")
    out_csv = Path(cfg.get("validation_output_csv", "validation_results.csv"))
    temp = cfg.get("temp_kelvin", 300)
    dt = cfg.get("dt_fs", 1.0)
    burn_in = cfg.get("validation_burn_in_frames", 0)

    if not ref_traj or not pred_traj:
        raise ValueError("Both 'reference_traj_file' and 'trajectory_file' must be provided in config.")

    results = validate_trajectories(ref_traj, pred_traj, temp, dt, burn_in)

    if results:
        df = pd.DataFrame([results])
        df.to_csv(out_csv, index=False)
        print(f"Saved results to {out_csv}")
    
    return results

if __name__ == "__main__":
    run()