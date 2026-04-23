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
    """Compute the radial distribution function g(r).

    Returns g(r) normalized so that g(r) → 1 at large r for a homogeneous
    liquid/solid (standard literature definition).  The second return value is
    the raw histogram counts for archiving.
    """
    n_atoms = len(frames[0])
    n_frames = len(frames)
    volume = frames[0].get_volume()  # assumed constant (NVT)

    bin_edges = np.linspace(0.1, rmax, bins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dr = bin_edges[1] - bin_edges[0]

    hists = np.zeros(bins)
    for frame in frames:
        dists = frame.get_all_distances(mic=True)
        dists = dists[np.triu_indices_from(dists, k=1)]
        h, _ = np.histogram(dists, bins=bin_edges)
        hists += h

    # g(r) = n(r) / [N_frames * N_pairs * rho * 4*pi*r^2 * dr]
    # where N_pairs = N*(N-1)/2 (unique pairs, upper-triangle convention)
    # and rho = N/V is the number density.
    # For a perfectly uniform density this gives g(r) = 1 at every r.
    n_pairs = n_atoms * (n_atoms - 1) / 2
    rho = n_atoms / volume
    ideal_counts = n_frames * n_pairs * rho * 4.0 * np.pi * r_centers**2 * dr
    with np.errstate(invalid='ignore', divide='ignore'):
        g_r = np.where(ideal_counts > 0, hists / ideal_counts, 0.0)

    return g_r, hists, r_centers

def _get_angle_distribution(frames, rcut=3.0, bins=90):
    """Compute the bond-angle distribution function (ADF).

    Returns a probability mass function (PMF) over angle bins, normalised so
    the values sum to 1.  Bin centres are computed correctly as mid-points of
    the histogram edges.
    """
    bin_edges = np.linspace(0, 180, bins + 1)
    angle_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hists = np.zeros(bins)
    for frame in frames:
        dists = frame.get_all_distances(mic=True)
        for i in range(len(frame)):
            neighbors = np.where((dists[i] < rcut) & (dists[i] > 0.1))[0]
            if len(neighbors) < 2:
                continue

            vecs = frame.positions[neighbors] - frame.positions[i]

            if any(frame.pbc):
                vecs, _ = find_mic(vecs, frame.cell, frame.pbc)

            norms = np.linalg.norm(vecs, axis=1)
            vecs = vecs / norms[:, np.newaxis]

            cos_angles = np.clip(np.dot(vecs, vecs.T), -1.0, 1.0)
            angles = np.arccos(cos_angles[np.triu_indices_from(cos_angles, k=1)])

            h, _ = np.histogram(np.degrees(angles), bins=bin_edges)
            hists += h

    norm_hists = hists / np.sum(hists) if np.sum(hists) > 0 else hists
    return norm_hists, hists, angle_centers

# --- 2. Dynamical Utilities ---
def _calculate_msd(frames):
    """Calculates MSD using unwrapped coordinates and multiple time origins (literature standard)."""
    positions = np.array([f.get_positions() for f in frames])
    cell_matrix = frames[0].get_cell().array  # 3×3, rows are lattice vectors

    # Unwrap in fractional (scaled) coordinates so triclinic cells are handled
    # correctly.  Cartesian unwrapping with cell.lengths() only works for
    # orthogonal boxes; for any non-orthogonal cell it produces wrong jumps.
    inv_cell = np.linalg.inv(cell_matrix)
    scaled = positions @ inv_cell.T  # (n_frames, n_atoms, 3)
    for i in range(1, len(scaled)):
        delta_frac = scaled[i] - scaled[i - 1]
        scaled[i] -= np.round(delta_frac)
    positions = scaled @ cell_matrix.T  # back to Cartesian

    n_frames, n_atoms, _ = positions.shape
    # lag=0 is trivially 0 by definition; allocate length n_frames but only
    # fill lags 1..n_frames-1 (lag 0 stays 0 as a placeholder for the curve).
    msd_arr = np.zeros(n_frames)
    for lag in range(1, n_frames):
        diff = positions[lag:] - positions[:-lag]
        sq_dist = np.sum(diff**2, axis=-1)
        msd_arr[lag] = np.mean(sq_dist)

    return msd_arr

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
    if vacf[0] == 0.0:
        raise ValueError("VACF[0] == 0: all velocities appear to be zero.")
    vacf = vacf / vacf[0]

    # Half-Hann window: take the right half of an odd-length full window.
    # hann(2N-1)[N-1:] starts at exactly 1.0 (the peak) and ends at exactly
    # 0.0, giving a smooth taper with no amplitude distortion at lag=0.
    # Using hann(2N)[N:] is a common mistake — its first sample is ≈0.9998,
    # not 1.0, and gets worse for short trajectories.
    window = hann(2 * steps - 1)[steps - 1:]
    vacf_windowed = vacf * window

    # The VACF is real and symmetric so by the Wiener–Khinchin theorem its
    # Fourier transform is real and non-negative.  Use np.real + clip(0)
    # rather than np.abs, which would inflate tiny negative numerical artefacts
    # at high frequencies into spurious positive spectral density.
    vdos = np.fft.rfft(vacf_windowed).real.clip(0)
    
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
        ref_epot_total = np.array([f.get_potential_energy() for f in ref_frames])
        pred_epot_total = np.array([f.get_potential_energy() for f in pred_frames])
        ref_epot = ref_epot_total / n_atoms
        pred_epot = pred_epot_total / n_atoms
        # Per-atom EMD (dimensionless scale-invariant comparison)
        res["EMD_Epot_eV_atom"] = wasserstein_distance(ref_epot, pred_epot)
        # Raw total-energy EMD: result has units of eV (same as the quantity being compared)
        res["EMD_Epot_eV_raw"] = wasserstein_distance(ref_epot_total, pred_epot_total)

        ref_ekin_total = np.array([f.get_kinetic_energy() for f in ref_frames])
        pred_ekin_total = np.array([f.get_kinetic_energy() for f in pred_frames])
        ref_ekin = ref_ekin_total / n_atoms
        pred_ekin = pred_ekin_total / n_atoms
        res["EMD_Ekin_eV_atom"] = wasserstein_distance(ref_ekin, pred_ekin)
        res["EMD_Ekin_eV_raw"] = wasserstein_distance(ref_ekin_total, pred_ekin_total)

        raw_data["Ref_Epot_eV_atom"] = ref_epot
        raw_data["Pred_Epot_eV_atom"] = pred_epot
        raw_data["Ref_Epot_eV_total"] = ref_epot_total
        raw_data["Pred_Epot_eV_total"] = pred_epot_total
        raw_data["Ref_Ekin_eV_atom"] = ref_ekin
        raw_data["Pred_Ekin_eV_atom"] = pred_ekin
        raw_data["Ref_Ekin_eV_total"] = ref_ekin_total
        raw_data["Pred_Ekin_eV_total"] = pred_ekin_total

    # 3. Force Magnitude Distributions
    if cfg.get("run_forces", True):
        ref_f_mag = np.linalg.norm(np.vstack([f.get_forces() for f in ref_frames]), axis=1)
        pred_f_mag = np.linalg.norm(np.vstack([f.get_forces() for f in pred_frames]), axis=1)
        res["EMD_Forces_eV_A"] = wasserstein_distance(ref_f_mag, pred_f_mag)
        
        raw_data["Ref_Force_Mag_eV_A"] = ref_f_mag
        raw_data["Pred_Force_Mag_eV_A"] = pred_f_mag

    # 4. RDF (Radial Distribution Function)
    if cfg.get("run_rdf", True):
        ref_rdf_g, ref_rdf_raw, x_rdf = _get_distance_distribution(ref_frames)
        pred_rdf_g, pred_rdf_raw, _ = _get_distance_distribution(pred_frames)

        # Normalise g(r) to a PMF to use as weights for the Wasserstein distance.
        # The result has units of Angstroms: the average distance probability mass
        # must be transported to transform the reference g(r) into the predicted one.
        ref_rdf_pmf = ref_rdf_g / np.sum(ref_rdf_g) if np.sum(ref_rdf_g) > 0 else ref_rdf_g
        pred_rdf_pmf = pred_rdf_g / np.sum(pred_rdf_g) if np.sum(pred_rdf_g) > 0 else pred_rdf_g
        res["EMD_RDF_Angstrom"] = wasserstein_distance(
            x_rdf, x_rdf, u_weights=ref_rdf_pmf, v_weights=pred_rdf_pmf
        )

        raw_data["RDF_Distance_Angstrom"] = x_rdf
        raw_data["RDF_Ref_Raw_Counts"] = ref_rdf_raw
        raw_data["RDF_Pred_Raw_Counts"] = pred_rdf_raw
        raw_data["RDF_Ref_g_r"] = ref_rdf_g
        raw_data["RDF_Pred_g_r"] = pred_rdf_g

    # 5. ADF (Angular Distribution Function)
    if cfg.get("run_adf", True):
        ref_adf_norm, ref_adf_raw, x_adf = _get_angle_distribution(ref_frames)
        pred_adf_norm, pred_adf_raw, _ = _get_angle_distribution(pred_frames)

        # EMD with bin-centre positions as the 1D support.
        # Result has units of degrees.
        res["EMD_ADF_Degrees"] = wasserstein_distance(
            x_adf, x_adf, u_weights=ref_adf_norm, v_weights=pred_adf_norm
        )

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

    # 7. Diffusion Coefficient from MSD (Einstein relation: MSD = 6*D*t)
    if cfg.get("run_msd", True):
        log_int = cfg.get("log_interval", 1)
        # Physical time between saved frames in seconds
        dt_frame_s = dt_fs * log_int * 1e-15

        ref_msd_arr  = _calculate_msd(ref_frames)
        pred_msd_arr = _calculate_msd(pred_frames)

        n_lags = len(ref_msd_arr)
        time_s = np.arange(n_lags) * dt_frame_s  # seconds

        def _diffusion_coeff(msd_arr, time_s):
            """Fit MSD[25%:75%] vs t to extract D = slope/6 in m²/s.

            The first quarter is dominated by ballistic/vibrational motion;
            the last quarter has high statistical noise (few time-origin pairs).
            Fitting the middle 25–75% of the trajectory is the standard
            literature approach for extracting D from MD.
            """
            n = len(msd_arr)
            lo, hi = n // 4, 3 * n // 4
            if hi - lo < 2:
                # Trajectory too short for a reliable fit; fall back to endpoint
                return msd_arr[-1] / (6.0 * time_s[-1]) * 1e-20 if time_s[-1] > 0 else 0.0
            slope = np.polyfit(time_s[lo:hi], msd_arr[lo:hi], 1)[0]  # Å²/s
            return slope / 6.0 * 1e-20  # convert Å²/s → m²/s

        D_ref  = _diffusion_coeff(ref_msd_arr,  time_s)
        D_pred = _diffusion_coeff(pred_msd_arr, time_s)

        res["D_ref_m2_s"]   = D_ref
        res["D_pred_m2_s"]  = D_pred
        res["D_Error_m2_s"] = abs(D_ref - D_pred)

        # Keep MSD time-axis in ps for human-readable CSV
        raw_data["MSD_Time_ps"]         = time_s * 1e12
        raw_data["Ref_MSD_vs_Time_A2"]  = ref_msd_arr
        raw_data["Pred_MSD_vs_Time_A2"] = pred_msd_arr

    # 8. Vibrational Density of States (VDOS)
    if cfg.get("run_vdos", False):
        try:
            log_int = cfg.get("log_interval", 1) 
            ref_vdos_norm, ref_vdos_raw, x_vdos = _calculate_vdos_spectrum(ref_frames, dt_fs, log_interval=log_int)
            pred_vdos_norm, pred_vdos_raw, _ = _calculate_vdos_spectrum(pred_frames, dt_fs, log_interval=log_int)
            
            # EMD with frequency bin positions as the 1D support.
            # Result has units of THz.
            res["EMD_VDOS_THz"] = wasserstein_distance(
                x_vdos, x_vdos, u_weights=ref_vdos_norm, v_weights=pred_vdos_norm
            )
            
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