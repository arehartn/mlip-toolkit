import numpy as np
import pandas as pd
from pathlib import Path
from ase.io import read
from scipy.stats import wasserstein_distance
from scipy.signal import correlate

# --- 1. Structural Utilities ---
def _get_distance_distribution(frames, rmax=6.0, bins=100):
    """Calculates pseudo-RDF. Returns (normalized, raw_hist, x_axis)."""
    hists = np.zeros(bins)
    for frame in frames:
        dists = frame.get_all_distances(mic=True)
        dists = dists[np.triu_indices_from(dists, k=1)]
        h, _ = np.histogram(dists, bins=bins, range=(0.1, rmax))
        hists += h
    
    x_axis = np.linspace(0.1, rmax, bins)
    norm_hists = hists / np.sum(hists) if np.sum(hists) > 0 else hists
    return norm_hists, hists, x_axis

def _get_angle_distribution(frames, rcut=3.0, bins=90):
    """Calculates pseudo-ADF. Returns (normalized, raw_hist, x_axis)."""
    hists = np.zeros(bins)
    for frame in frames:
        dists = frame.get_all_distances(mic=True)
        for i in range(len(frame)):
            neighbors = np.where((dists[i] < rcut) & (dists[i] > 0.1))[0]
            if len(neighbors) < 2: continue
            
            vecs = frame.positions[neighbors] - frame.positions[i]
            vecs = vecs - np.round(vecs / frame.cell.lengths()) * frame.cell.lengths()
            norms = np.linalg.norm(vecs, axis=1)
            vecs = vecs / norms[:, np.newaxis]
            
            cos_angles = np.clip(np.dot(vecs, vecs.T), -1.0, 1.0)
            angles = np.arccos(cos_angles[np.triu_indices_from(cos_angles, k=1)])
            
            h, _ = np.histogram(np.degrees(angles), bins=bins, range=(0, 180))
            hists += h
            
    x_axis = np.linspace(0, 180, bins)
    norm_hists = hists / np.sum(hists) if np.sum(hists) > 0 else hists
    return norm_hists, hists, x_axis

# --- 2. Dynamical Utilities ---
def _calculate_msd(frames):
    pos_0 = frames[0].get_positions()
    msds = []
    for frame in frames:
        diff = frame.get_positions() - pos_0
        msds.append(np.mean(np.sum(diff**2, axis=1)))
    return np.mean(msds)

def _calculate_vdos_spectrum(frames, dt_fs):
    """Calculates VDOS. Returns (normalized, raw_vdos, frequencies_THz)."""
    vels = np.array([f.get_velocities() for f in frames])
    
    if vels[0] is None:
        raise ValueError("Velocities not found in trajectory. VDOS cannot be calculated.")
        
    steps, atoms, dims = vels.shape
    
    vacf = np.zeros(steps)
    for i in range(atoms):
        for j in range(dims):
            v = vels[:, i, j]
            corr = correlate(v, v, mode='full')
            vacf += corr[steps-1:] / np.arange(steps, 0, -1)
            
    vacf = vacf / vacf[0]
    vdos = np.abs(np.fft.rfft(vacf))**2
    
    # Calculate Frequency X-Axis in THz
    # dt_fs is timestep in femtoseconds. Time between frames in seconds = dt_fs * 1e-15
    freqs_Hz = np.fft.rfftfreq(steps, d=dt_fs * 1e-15)
    freqs_THz = freqs_Hz * 1e-12 
    
    norm_vdos = vdos / np.sum(vdos) if np.sum(vdos) > 0 else vdos
    return norm_vdos, vdos, freqs_THz

# --- 3. Main Validation Engine ---
def validate_trajectories(ref_path, pred_path, temp_k, dt_fs, burn_in=0, cfg=None):
    if cfg is None: cfg = {}
    
    print(f"Loading {ref_path} and {pred_path}...")
    try:
        ref_frames = read(ref_path, index=f"{burn_in}:")
        pred_frames = read(pred_path, index=f"{burn_in}:")
    except Exception as e:
        print(f"Error loading trajectories: {e}")
        return None

    kb = 8.617333262145e-5 # eV/K
    n_atoms = len(ref_frames[0])

    print(f"Validating {len(ref_frames)} AIMD frames vs {len(pred_frames)} MACE frames...")
    
    res = {}
    raw_data = {} # Dictionary to hold raw data arrays for export

    # 1-2. Energy Distributions
    if cfg.get("run_energy", True):
        ref_epot = np.array([f.get_potential_energy() for f in ref_frames]) / n_atoms
        pred_epot = np.array([f.get_potential_energy() for f in pred_frames]) / n_atoms
        res["EMD_Epot_eV_atom"] = wasserstein_distance(ref_epot, pred_epot)

        ref_ekin = np.array([f.get_kinetic_energy() for f in ref_frames]) / n_atoms
        pred_ekin = np.array([f.get_kinetic_energy() for f in pred_frames]) / n_atoms
        res["EMD_Ekin_eV_atom"] = wasserstein_distance(ref_ekin, pred_ekin)
        
        raw_data["Ref_Epot_Raw"] = ref_epot
        raw_data["Pred_Epot_Raw"] = pred_epot

    # 3. Force Magnitude Distributions
    if cfg.get("run_forces", True):
        ref_f_mag = np.linalg.norm(np.vstack([f.get_forces() for f in ref_frames]), axis=1)
        pred_f_mag = np.linalg.norm(np.vstack([f.get_forces() for f in pred_frames]), axis=1)
        res["EMD_Forces_eV_A"] = wasserstein_distance(ref_f_mag, pred_f_mag)

    # 4. RDF (Radial Distribution Function)
    if cfg.get("run_rdf", True):
        ref_rdf_norm, ref_rdf_raw, x_rdf = _get_distance_distribution(ref_frames)
        pred_rdf_norm, pred_rdf_raw, _ = _get_distance_distribution(pred_frames)
        res["EMD_RDF_Structure"] = wasserstein_distance(ref_rdf_norm, pred_rdf_norm)
        
        raw_data["RDF_Distance_Angstrom"] = x_rdf
        raw_data["RDF_Ref_Raw"] = ref_rdf_raw
        raw_data["RDF_Pred_Raw"] = pred_rdf_raw

    # 5. ADF (Angular Distribution Function)
    if cfg.get("run_adf", True):
        ref_adf_norm, ref_adf_raw, x_adf = _get_angle_distribution(ref_frames)
        pred_adf_norm, pred_adf_raw, _ = _get_angle_distribution(pred_frames)
        res["EMD_ADF_Structure"] = wasserstein_distance(ref_adf_norm, pred_adf_norm)
        
        raw_data["ADF_Angle_Degrees"] = x_adf
        raw_data["ADF_Ref_Raw"] = ref_adf_raw
        raw_data["ADF_Pred_Raw"] = pred_adf_raw

    # 6. Heat Capacity (Cv) Error
    if cfg.get("run_cv", True):
        ref_epot_tot = np.array([f.get_potential_energy() for f in ref_frames])
        pred_epot_tot = np.array([f.get_potential_energy() for f in pred_frames])
        res["Cv_Error_eV_K_atom"] = abs(np.var(ref_epot_tot) - np.var(pred_epot_tot)) / (kb * temp_k**2 * n_atoms)

    # 7. Mean Squared Displacement (MSD) Error
    if cfg.get("run_msd", True):
        res["MSD_Error_A2"] = abs(_calculate_msd(ref_frames) - _calculate_msd(pred_frames))

    # 8. Vibrational Density of States (VDOS)
    if cfg.get("run_vdos", False):
        try:
            ref_vdos_norm, ref_vdos_raw, x_vdos = _calculate_vdos_spectrum(ref_frames, dt_fs)
            pred_vdos_norm, pred_vdos_raw, _ = _calculate_vdos_spectrum(pred_frames, dt_fs)
            res["EMD_VDOS_Spectrum"] = wasserstein_distance(ref_vdos_norm, pred_vdos_norm)
            
            raw_data["VDOS_Freq_THz"] = x_vdos
            raw_data["VDOS_Ref_Raw"] = ref_vdos_raw
            raw_data["VDOS_Pred_Raw"] = pred_vdos_raw
        except Exception as e:
            print(f"Warning: Could not compute VDOS: {e}")

    # Print Scalar Results
    print("\n--- Validation Results ---")
    for k, v in res.items():
        print(f"{k:25s}: {v:.6f}")
        
    # --- EXPORT RAW DATA TO CSV ---
    if raw_data:
        # We use pd.Series so pandas doesn't crash from columns having different row lengths
        raw_df = pd.DataFrame({k: pd.Series(v) for k, v in raw_data.items()})
        raw_csv_path = str(cfg.get("validation_output_csv", "validation_results.csv")).replace(".csv", "_RAW_DATA.csv")
        raw_df.to_csv(raw_csv_path, index=False)
        print(f"Saved raw distributional data to {raw_csv_path}")

    return res

def run(config_overrides=None):
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

    # Note: We now pass `cfg` into validate_trajectories!
    results = validate_trajectories(ref_traj, pred_traj, temp, dt, burn_in, cfg)

    if results:
        df = pd.DataFrame([results])
        df.to_csv(out_csv, index=False)
        print(f"Saved scalar EMD results to {out_csv}")
    
    return results

if __name__ == "__main__":
    run()