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

    # g(r) = hists / ideal_counts, where ideal_counts is the expected number of
    # upper-triangle pair counts in [r, r+dr] per frame for a uniform ideal gas:
    #
    #   ideal = N*(N-1)/2 * 4*pi*r^2*dr / V
    #
    # For a perfectly uniform density this gives g(r) = 1 at every r.
    # Note: do NOT multiply by rho = N/V here — that introduces an extra N factor
    # and suppresses all g(r) values by N (e.g. 768×), flattening the plot.
    n_pairs = n_atoms * (n_atoms - 1) / 2
    ideal_counts = n_frames * n_pairs * 4.0 * np.pi * r_centers**2 * dr / volume
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
def _unwrap_positions(frames):
    """Unwrap Cartesian positions for triclinic cells (same logic as MSD)."""
    positions = np.array([f.get_positions() for f in frames])
    cell_matrix = frames[0].get_cell().array
    inv_cell = np.linalg.inv(cell_matrix)
    scaled = positions @ inv_cell.T
    for i in range(1, len(scaled)):
        delta_frac = scaled[i] - scaled[i - 1]
        scaled[i] -= np.round(delta_frac)
    return scaled @ cell_matrix.T


def _calculate_msd(frames):
    """Calculates MSD using unwrapped coordinates and multiple time origins (literature standard)."""
    positions = _unwrap_positions(frames)

    n_frames, n_atoms, _ = positions.shape
    msd_arr = np.zeros(n_frames)
    for lag in range(1, n_frames):
        diff = positions[lag:] - positions[:-lag]
        sq_dist = np.sum(diff**2, axis=-1)
        msd_arr[lag] = np.mean(sq_dist)

    return msd_arr


def _velocities_from_positions(frames, dt_fs, log_interval=1):
    """Central-difference velocities (Å/fs) from unwrapped Cartesian positions."""
    pos = _unwrap_positions(frames)
    steps = len(frames)
    frame_dt_fs = dt_fs * log_interval
    vels = np.zeros_like(pos)
    for i in range(steps):
        if i == 0:
            vels[i] = (pos[1] - pos[0]) / frame_dt_fs
        elif i == steps - 1:
            vels[i] = (pos[-1] - pos[-2]) / frame_dt_fs
        else:
            vels[i] = (pos[i + 1] - pos[i - 1]) / (2.0 * frame_dt_fs)
    return vels


def _subtract_com_velocity(vels, masses):
    """Remove center-of-mass drift (standard before VACF / VDOS)."""
    vels = np.array(vels, dtype=float, copy=True)
    for t in range(len(vels)):
        vels[t] -= np.average(vels[t], weights=masses, axis=0)
    return vels


def _stored_velocities_usable(stored, pos_derived, min_frame_frac=0.9, max_rel_err=0.35):
    """Reject AIMD stored velocities that are partial zeros or disagree with positions."""
    if np.any(np.isnan(stored)):
        return False
    frame_speed = np.mean(np.linalg.norm(stored, axis=2), axis=1)
    if np.mean(frame_speed > 1e-4) < min_frame_frac:
        return False
    pos_norm = np.linalg.norm(pos_derived)
    if pos_norm < 1e-12:
        return False
    rel_err = np.linalg.norm(stored - pos_derived) / pos_norm
    return rel_err <= max_rel_err


def _get_velocities(frames, dt_fs, log_interval=1):
    """Velocities (Å/fs) for VACF/VDOS; position finite differences when stored data are bad."""
    steps = len(frames)
    if steps < 2:
        raise ValueError("Need at least 2 frames to estimate velocities.")

    pos_derived = _velocities_from_positions(frames, dt_fs, log_interval)

    try:
        stored = np.array([f.get_velocities() for f in frames], dtype=float)
        if _stored_velocities_usable(stored, pos_derived):
            vels = stored
        else:
            print(
                "    Warning: Stored velocities missing, partial zeros, or inconsistent "
                "with positions — using unwrapped position finite differences."
            )
            vels = pos_derived
    except (TypeError, ValueError):
        print("    Warning: Velocities missing — computing from unwrapped positions...")
        vels = pos_derived

    return _subtract_com_velocity(vels, frames[0].get_masses())


def _vacf_unnormalized(vels):
    """Σ_{i,α} ⟨v_{iα}(0) v_{iα}(t)⟩ via scipy.signal.correlate (user workflow)."""
    steps, atoms, dims = vels.shape
    vacf = np.zeros(steps)
    for i in range(atoms):
        for j in range(dims):
            v = vels[:, i, j]
            corr = correlate(v, v, mode='full')
            vacf += corr[steps - 1:] / np.arange(steps, 0, -1)
    return vacf


def _calculate_vacf(frames, dt_fs, log_interval=1):
    """Normalized VACF C(t)/C(0); lag axis in fs."""
    vels = _get_velocities(frames, dt_fs, log_interval)
    vacf = _vacf_unnormalized(vels)
    if vacf[0] == 0.0:
        raise ValueError("VACF[0] == 0: all velocities appear to be zero.")
    lag_fs = np.arange(len(vacf)) * dt_fs * log_interval
    return vacf / vacf[0], lag_fs


def _vacf_integration_cutoff(vacf_norm):
    """Integrate VACF until the first zero crossing (standard GK practice)."""
    for k in range(1, len(vacf_norm)):
        if vacf_norm[k] <= 0.0:
            return k
    return max(2, 3 * len(vacf_norm) // 4)


def _green_kubo_diffusion(frames, dt_fs, log_interval=1):
    """Self-diffusion from Green–Kubo: D = (1/(3N)) ∫₀^t_c ⟨Σ_i v_i(0)·v_i(t)⟩ dt."""
    vels = _get_velocities(frames, dt_fs, log_interval)
    n_atoms = len(frames[0])
    vacf = _vacf_unnormalized(vels)
    if vacf[0] == 0.0:
        raise ValueError("VACF[0] == 0: all velocities appear to be zero.")

    dt_frame_s = dt_fs * log_interval * 1e-15
    lag_s = np.arange(len(vacf)) * dt_frame_s
    vacf_norm = vacf / vacf[0]
    cutoff = _vacf_integration_cutoff(vacf_norm)

    integral = np.trapz(vacf[:cutoff], lag_s[:cutoff])
    return integral / (3.0 * n_atoms) * 1e-5


def _aimd_dt_params(cfg, dt_fs, log_int):
    """AIMD frame spacing; None / missing keys fall back to MLIP dt_fs and log_interval."""
    aimd_dt = cfg.get("aimd_dt_fs")
    aimd_log = cfg.get("aimd_log_interval")
    return (
        dt_fs if aimd_dt is None else aimd_dt,
        log_int if aimd_log is None else aimd_log,
    )


def traj_dt_params(traj_path, cfg):
    """Return (dt_fs, log_interval) for a trajectory path (AIMD vs MLIP)."""
    log_int = cfg.get("log_interval", 1)
    dt_fs = cfg.get("dt_fs", 1.0)
    ref = cfg.get("reference_traj_file")
    if ref and Path(traj_path).resolve() == Path(ref).resolve():
        return _aimd_dt_params(cfg, dt_fs, log_int)
    return dt_fs, log_int


def _calculate_vdos_spectrum(frames, dt_fs, log_interval=1):
    """Phonon spectral function g(ω) = |∫ C(t) e^{-iωt} dt|² from windowed normalized VACF."""
    vels = _get_velocities(frames, dt_fs, log_interval)
    vacf = _vacf_unnormalized(vels)
    if vacf[0] == 0.0:
        raise ValueError("VACF[0] == 0: all velocities appear to be zero.")
    vacf /= vacf[0]

    steps = len(vacf)
    dt_frame_s = dt_fs * log_interval * 1e-15
    window = hann(2 * steps - 1)[steps - 1:]
    ft = np.fft.rfft(vacf * window)
    vdos_raw = (np.abs(ft) ** 2) * dt_frame_s

    freqs_thz = np.fft.rfftfreq(steps, d=dt_frame_s) * 1e-12

    vdos_pmf = vdos_raw / np.sum(vdos_raw) if np.sum(vdos_raw) > 0 else vdos_raw
    return vdos_raw, vdos_pmf, freqs_thz
    
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

    log_int = cfg.get("log_interval", 1)

    # 7. Diffusion Coefficient from MSD (Einstein relation: MSD = 6*D*t)
    if cfg.get("run_msd", True):
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

    # 7b. Diffusion coefficient — Green–Kubo (VACF integral)
    if cfg.get("run_green_kubo", True):
        try:
            aimd_dt_fs, aimd_log_interval = _aimd_dt_params(cfg, dt_fs, log_int)
            D_ref_gk = _green_kubo_diffusion(
                ref_frames, aimd_dt_fs, log_interval=aimd_log_interval)
            D_pred_gk = _green_kubo_diffusion(
                pred_frames, dt_fs, log_interval=log_int)

            res["D_ref_GK_m2_s"] = D_ref_gk
            res["D_pred_GK_m2_s"] = D_pred_gk
            res["D_GK_Error_m2_s"] = abs(D_ref_gk - D_pred_gk)
        except Exception as e:
            print(f"Warning: Could not compute Green–Kubo diffusion: {e}")

    # 8. Velocity Autocorrelation Function (VACF)
    if cfg.get("run_vacf", False):
        try:
            aimd_dt_fs, aimd_log_interval = _aimd_dt_params(cfg, dt_fs, log_int)

            ref_vacf, ref_vacf_lag = _calculate_vacf(
                ref_frames, aimd_dt_fs, log_interval=aimd_log_interval)
            pred_vacf, pred_vacf_lag = _calculate_vacf(
                pred_frames, dt_fs, log_interval=log_int)

            if len(ref_vacf) != len(pred_vacf):
                pred_vacf_interp = np.interp(ref_vacf_lag, pred_vacf_lag, pred_vacf)
            else:
                pred_vacf_interp = pred_vacf

            ref_w = ref_vacf / ref_vacf.sum() if ref_vacf.sum() > 0 else ref_vacf
            pred_w = pred_vacf_interp / pred_vacf_interp.sum() if pred_vacf_interp.sum() > 0 else pred_vacf_interp
            res["EMD_VACF_fs"] = wasserstein_distance(
                ref_vacf_lag, ref_vacf_lag, u_weights=ref_w, v_weights=pred_w
            )

            raw_data["Ref_VACF_Lag_fs"] = ref_vacf_lag
            raw_data["Ref_VACF"] = ref_vacf
            raw_data["Pred_VACF_Lag_fs"] = pred_vacf_lag
            raw_data["Pred_VACF"] = pred_vacf
        except Exception as e:
            print(f"Warning: Could not compute VACF: {e}")

    # 9. Vibrational Density of States (VDOS)
    if cfg.get("run_vdos", False):
        try:
            # AIMD and MACE trajectories can have different timesteps and output
            # intervals.  Use aimd_dt_fs / aimd_log_interval for the reference if
            # provided; fall back to the MACE values if not.
            aimd_dt_fs, aimd_log_interval = _aimd_dt_params(cfg, dt_fs, log_int)

            ref_vdos, ref_vdos_pmf, ref_vdos_freq = _calculate_vdos_spectrum(
                ref_frames, aimd_dt_fs, log_interval=aimd_log_interval)
            pred_vdos, pred_vdos_pmf, pred_vdos_freq = _calculate_vdos_spectrum(
                pred_frames, dt_fs, log_interval=log_int)

            # EMD on spectral shape (PMF), not on raw g(ω) amplitudes.
            if len(ref_vdos_pmf) != len(pred_vdos_pmf):
                pred_pmf_interp = np.interp(ref_vdos_freq, pred_vdos_freq, pred_vdos_pmf)
                pred_pmf_interp /= pred_pmf_interp.sum()
            else:
                pred_pmf_interp = pred_vdos_pmf

            res["EMD_VDOS_THz"] = wasserstein_distance(
                ref_vdos_freq, ref_vdos_freq,
                u_weights=ref_vdos_pmf, v_weights=pred_pmf_interp
            )

            raw_data["Ref_VDOS_Freq_THz"] = ref_vdos_freq
            raw_data["Ref_VDOS_g_omega"] = ref_vdos
            raw_data["Ref_VDOS_Shape_PMF"] = ref_vdos_pmf
            raw_data["Pred_VDOS_Freq_THz"] = pred_vdos_freq
            raw_data["Pred_VDOS_g_omega"] = pred_vdos
            raw_data["Pred_VDOS_Shape_PMF"] = pred_vdos_pmf
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

def run(config=None, config_path=None, config_overrides=None):
    from md_sim.config import load_config, normalize_config

    if config is None:
        cfg = load_config(config_path, profile="simulation")
    else:
        cfg = dict(config)
        if config_path is not None:
            cfg = {**load_config(config_path, profile="simulation"), **cfg}
    if config_overrides:
        cfg.update(config_overrides)
    cfg = normalize_config(cfg)

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

def main():
    import argparse
    from md_sim.config import add_config_cli

    parser = argparse.ArgumentParser(description="Validate MD trajectories against a reference.")
    add_config_cli(parser)
    args = parser.parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()