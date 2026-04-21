import numpy as np
import pandas as pd
from pathlib import Path
from ase.io import read
from scipy.stats import wasserstein_distance
from scipy.signal import correlate
from ase.geometry import find_mic
from scipy.signal.windows import hann

# --- 1. Structural Utilities ---
def _get_distance_distribution(frames, rmax=6.0, bins=100):
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
    hists = np.zeros(bins)
    for frame in frames:
        dists = frame.get_all_distances(mic=True)
        for i in range(len(frame)):
            neighbors = np.where((dists[i] < rcut) & (dists[i] > 0.1))[0]
            if len(neighbors) < 2: continue
            
            vecs = frame.positions[neighbors] - frame.positions[i]
            
            # Fix: Use ASE's robust MIC instead of manual array division
            if any(frame.pbc):
                vecs, _ = find_mic(vecs, frame.cell, frame.pbc)
                
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
    """Calculates MSD using unwrapped coordinates and multiple time origins (literature standard)."""
    # 1. Extract and unwrap positions so boundary jumps don't ruin the distance math
    positions = np.array([f.get_positions() for f in frames])
    cell = frames[0].get_cell()
    
    # Simple unwrapping algorithm (assumes continuous frames)
    for i in range(1, len(positions)):
        # Find the jump from the previous frame
        delta = positions[i] - positions[i-1]
        # If the jump is larger than half the box size, it crossed a boundary! Shift it back.
        positions[i] -= np.round(delta / cell.lengths()) * cell.lengths()

    n_frames, n_atoms, _ = positions.shape
    msd_arr = np.zeros(n_frames)
    
    # 2. Multiple Time Origins (Rolling Window)
    # We average the displacement over ALL possible starting times to get a smooth curve
    for lag in range(1, n_frames):
        # Difference between positions separated by 'lag' frames
        diff = positions[lag:] - positions[:-lag]
        # Square the distances, average over all atoms and all starting frames for this lag
        sq_dist = np.sum(diff**2, axis=-1)
        msd_arr[lag] = np.mean(sq_dist)
        
    return np.mean(msd_arr), msd_arr

def _calculate_vdos_spectrum(frames, dt_fs, log_interval=1):
    vels = np.array([f.get_velocities() for f in frames])
    
    # Fallback just in case velocities are missing
    if vels[0] is None or np.all(vels[0] == 0.0):
        print("    Warning: Valid velocities not found! Approximating from positions...")
        steps, atoms = len(frames), len(frames[0])
        vels = np.zeros((steps, atoms, 3))
        cell, pbc = frames[0].get_cell(), frames[0].get_pbc()
        frame_dt = dt_fs * log_interval
        
        for i in range(steps):
            if i == 0:
                delta_r = frames[1].get_positions() - frames[0].get_positions()
                dt_step = frame_dt
            elif i == steps - 1:
                delta_r = frames[-1].get_positions() - frames[-2].get_positions()
                dt_step = frame_dt
            else:
                delta_r = frames[i+1].get_positions() - frames[i-1].get_positions()
                dt_step = 2 * frame_dt
            delta_r_mic, _ = find_mic(delta_r, cell, pbc)
            vels[i] = delta_r_mic / dt_step

    steps, atoms, dims = vels.shape
    vacf = np.zeros(steps)
    for i in range(atoms):
        for j in range(dims):
            v = vels[:, i, j]
            corr = correlate(v, v, mode='full')
            vacf += corr[steps-1:] / np.arange(steps, 0, -1)
            
    # Normalize VACF so it starts at 1.0
    vacf = vacf / vacf[0]
    
    # --- LITERATURE STANDARD WINDOWING ---
    # We apply a Hann half-window to smoothly taper the tail end of the VACF to zero
    window = hann(steps * 2)[steps:] 
    vacf_windowed = vacf * window
    
    # Fourier Transform the *windowed* VACF (using magnitude, standard practice)
    vdos = np.abs(np.fft.rfft(vacf_windowed))
    
    freqs_Hz = np.fft.rfftfreq(steps, d=dt_fs * log_interval * 1e-15)
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
    raw_data = {}

    # 1-2. Energy Distributions
    if cfg.get("run_energy", True):
        ref_epot = np.array([f.get_potential_energy() for f in ref_frames]) / n_atoms
        pred_epot = np.array([f.get_potential_energy() for f in pred_frames]) / n_atoms
        res["EMD_Epot_eV_atom"] = wasserstein_distance(ref_epot, pred_epot)

        ref_ekin = np.array([f.get_kinetic_energy() for f in ref_frames]) / n_atoms
        pred_ekin = np.array([f.get_kinetic_energy() for f in pred_frames]) / n_atoms
        res["EMD_Ekin_eV_atom"] = wasserstein_distance(ref_ekin, pred_ekin)
        
        raw_data["Ref_Epot_eV_atom"] = ref_epot
        raw_data["Pred_Epot_eV_atom"] = pred_epot
        raw_data["Ref_Ekin_eV_atom"] = ref_ekin
        raw_data["Pred_Ekin_eV_atom"] = pred_ekin

    # 3. Force Magnitude Distributions
    if cfg.get("run_forces", True):
        ref_f_mag = np.linalg.norm(np.vstack([f.get_forces() for f in ref_frames]), axis=1)
        pred_f_mag = np.linalg.norm(np.vstack([f.get_forces() for f in pred_frames]), axis=1)
        res["EMD_Forces_eV_A"] = wasserstein_distance(ref_f_mag, pred_f_mag)
        
        raw_data["Ref_Force_Mag_eV_A"] = ref_f_mag
        raw_data["Pred_Force_Mag_eV_A"] = pred_f_mag

    # 4. RDF (Radial Distribution Function)
    if cfg.get("run_rdf", True):
        ref_rdf_norm, ref_rdf_raw, x_rdf = _get_distance_distribution(ref_frames)
        pred_rdf_norm, pred_rdf_raw, _ = _get_distance_distribution(pred_frames)
        
        # Calculate BOTH Normalized and Raw EMD
        res["EMD_RDF_Structure_Norm"] = wasserstein_distance(ref_rdf_norm, pred_rdf_norm)
        res["EMD_RDF_Structure_Raw"] = wasserstein_distance(ref_rdf_raw, pred_rdf_raw)
        
        raw_data["RDF_Distance_Angstrom"] = x_rdf
        raw_data["RDF_Ref_Raw_Counts"] = ref_rdf_raw
        raw_data["RDF_Pred_Raw_Counts"] = pred_rdf_raw
        raw_data["RDF_Ref_Norm_Prob"] = ref_rdf_norm
        raw_data["RDF_Pred_Norm_Prob"] = pred_rdf_norm

    # 5. ADF (Angular Distribution Function)
    if cfg.get("run_adf", True):
        ref_adf_norm, ref_adf_raw, x_adf = _get_angle_distribution(ref_frames)
        pred_adf_norm, pred_adf_raw, _ = _get_angle_distribution(pred_frames)
        
        # Calculate BOTH Normalized and Raw EMD
        res["EMD_ADF_Structure_Norm"] = wasserstein_distance(ref_adf_norm, pred_adf_norm)
        res["EMD_ADF_Structure_Raw"] = wasserstein_distance(ref_adf_raw, pred_adf_raw)
        
        raw_data["ADF_Angle_Degrees"] = x_adf
        raw_data["ADF_Ref_Raw_Counts"] = ref_adf_raw
        raw_data["ADF_Pred_Raw_Counts"] = pred_adf_raw
        raw_data["ADF_Ref_Norm_Prob"] = ref_adf_norm
        raw_data["ADF_Pred_Norm_Prob"] = pred_adf_norm

    # 6. Heat Capacity (Cv) Error
    if cfg.get("run_cv", True):
        ref_epot_tot = np.array([f.get_potential_energy() for f in ref_frames])
        pred_epot_tot = np.array([f.get_potential_energy() for f in pred_frames])
        res["Cv_Error_eV_K_atom"] = abs(np.var(ref_epot_tot) - np.var(pred_epot_tot)) / (kb * temp_k**2 * n_atoms)
        
        raw_data["Ref_Total_Epot_eV"] = ref_epot_tot
        raw_data["Pred_Total_Epot_eV"] = pred_epot_tot

    # 7. Mean Squared Displacement (MSD) Error
    if cfg.get("run_msd", True):
        ref_msd_mean, ref_msd_arr = _calculate_msd(ref_frames)
        pred_msd_mean, pred_msd_arr = _calculate_msd(pred_frames)
        res["MSD_Error_A2"] = abs(ref_msd_mean - pred_msd_mean)
        
        raw_data["Ref_MSD_vs_Time_A2"] = ref_msd_arr
        raw_data["Pred_MSD_vs_Time_A2"] = pred_msd_arr

    # 8. Vibrational Density of States (VDOS)
    if cfg.get("run_vdos", False):
        try:
            log_int = cfg.get("log_interval", 1) 
            ref_vdos_norm, ref_vdos_raw, x_vdos = _calculate_vdos_spectrum(ref_frames, dt_fs, log_interval=log_int)
            pred_vdos_norm, pred_vdos_raw, _ = _calculate_vdos_spectrum(pred_frames, dt_fs, log_interval=log_int)
            
            # Calculate BOTH Normalized and Raw EMD
            res["EMD_VDOS_Spectrum_Norm"] = wasserstein_distance(ref_vdos_norm, pred_vdos_norm)
            res["EMD_VDOS_Spectrum_Raw"] = wasserstein_distance(ref_vdos_raw, pred_vdos_raw)
            
            raw_data["VDOS_Freq_THz"] = x_vdos
            raw_data["VDOS_Ref_Raw_Intensity"] = ref_vdos_raw
            raw_data["VDOS_Pred_Raw_Intensity"] = pred_vdos_raw
            raw_data["VDOS_Ref_Norm_Prob"] = ref_vdos_norm
            raw_data["VDOS_Pred_Norm_Prob"] = pred_vdos_norm
        except Exception as e:
            print(f"Warning: Could not compute VDOS: {e}")

    # Print Scalar Results
    print("\n--- Validation Results ---")
    for k, v in res.items():
        print(f"{k:30s}: {v:.6f}")
        
    # --- EXPORT RAW DATA TO CSV ---
    if raw_data:
        raw_df = pd.DataFrame({k: pd.Series(v) for k, v in raw_data.items()})
        raw_csv_path = str(cfg.get("validation_output_csv", "validation_results.csv")).replace(".csv", "_RAW_DATA.csv")
        raw_df.to_csv(raw_csv_path, index=False)
        print(f"Saved ALL raw distributional data to {raw_csv_path}")

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

    results = validate_trajectories(ref_traj, pred_traj, temp, dt, burn_in, cfg)

    if results:
        df = pd.DataFrame([results])
        df.to_csv(out_csv, index=False)
        print(f"Saved scalar validation metrics to {out_csv}")
    
    return results

if __name__ == "__main__":
    run()