import numpy as np
import pandas as pd
from pathlib import Path
from ase import units
from ase.io import read
from scipy.stats import wasserstein_distance
from itertools import combinations_with_replacement
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter1d
from ase.geometry import find_mic
from scipy.signal.windows import hann
from scipy.integrate import cumulative_trapezoid
from ase.neighborlist import neighbor_list

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


def _com_free_positions(frames):
    """Unwrapped positions with the system centre-of-mass motion removed.

    The MLIP run uses FixCom, but AIMD references usually do not, so any COM
    drift in the reference would otherwise inflate its MSD.
    """
    positions = _unwrap_positions(frames)
    com = np.average(positions, axis=1, weights=frames[0].get_masses())
    return positions - com[:, None, :]


def _msd_from_positions(positions):
    """MSD over multiple time origins from (n_frames, n_atoms, 3) positions."""
    n_frames = positions.shape[0]
    msd_arr = np.zeros(n_frames)
    for lag in range(1, n_frames):
        diff = positions[lag:] - positions[:-lag]
        sq_dist = np.sum(diff**2, axis=-1)
        msd_arr[lag] = np.mean(sq_dist)

    return msd_arr


def _calculate_msd(frames, indices=None):
    """Calculates MSD using unwrapped, COM-free coordinates and multiple time origins.

    indices : optional atom indices (e.g. one element) to average over.
    """
    positions = _com_free_positions(frames)
    if indices is not None:
        positions = positions[:, indices]
    return _msd_from_positions(positions)


def _msd_fit_window(n):
    """Middle 25–75% of lag times: skips the ballistic regime and the noisy tail."""
    return n // 4, 3 * n // 4


def _msd_diffusion_coeff(msd_arr, time_s):
    """Fit MSD[25%:75%] vs t to extract D = slope/6 in m²/s."""
    n = len(msd_arr)
    lo, hi = _msd_fit_window(n)
    if hi - lo < 2:
        # Trajectory too short for a reliable fit; fall back to endpoint
        return msd_arr[-1] / (6.0 * time_s[-1]) * 1e-20 if time_s[-1] > 0 else 0.0
    slope = np.polyfit(time_s[lo:hi], msd_arr[lo:hi], 1)[0]  # Å²/s
    return slope / 6.0 * 1e-20  # convert Å²/s → m²/s


def _msd_exponent(msd_arr, time_s):
    """Log-log slope α of MSD ∝ t^α over the fit window.

    α ≈ 1 is diffusive (D is meaningful), α < 1 is caged/sub-diffusive,
    α ≈ 2 is ballistic. If α is far from 1, the Einstein D is not converged.
    """
    lo, hi = _msd_fit_window(len(msd_arr))
    lo = max(lo, 1)
    t, m = time_s[lo:hi], msd_arr[lo:hi]
    ok = (t > 0) & (m > 0)
    if ok.sum() < 2:
        return np.nan
    return np.polyfit(np.log(t[ok]), np.log(m[ok]), 1)[0]


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


def _stored_velocities_in_A_fs(frames, pos_derived):
    """Stored velocities converted to Å/fs, or None if they match positions in neither unit.

    ASE-written trajectories store Å per ASE time unit (1 fs = units.fs ≈ 0.098),
    but AIMD converters often write Å/fs directly, so both are tried against
    the position-derived velocities.
    """
    raw = np.array([f.get_velocities() for f in frames], dtype=float)
    for label, vels in (("ASE units", raw * units.fs), ("Å/fs", raw)):
        if _stored_velocities_usable(vels, pos_derived):
            print(f"    Using stored velocities ({label}).")
            return vels
    return None


def _get_velocities(frames, dt_fs, log_interval=1):
    """Velocities (Å/fs) for VACF/VDOS; position finite differences when stored data are bad."""
    steps = len(frames)
    if steps < 2:
        raise ValueError("Need at least 2 frames to estimate velocities.")

    pos_derived = _velocities_from_positions(frames, dt_fs, log_interval)

    try:
        vels = _stored_velocities_in_A_fs(frames, pos_derived)
        if vels is None:
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


def _select_velocities(frames, dt_fs, log_interval, vels, indices):
    """Use precomputed velocities if given, then restrict to atom indices."""
    if vels is None:
        vels = _get_velocities(frames, dt_fs, log_interval)
    if indices is not None:
        vels = vels[:, indices]
    return vels


def _calculate_vacf(frames, dt_fs, log_interval=1, vels=None, indices=None):
    """Normalized VACF C(t)/C(0); lag axis in fs."""
    vels = _select_velocities(frames, dt_fs, log_interval, vels, indices)
    vacf = _vacf_unnormalized(vels)
    if vacf[0] == 0.0:
        raise ValueError("VACF[0] == 0: all velocities appear to be zero.")
    lag_fs = np.arange(len(vacf)) * dt_fs * log_interval
    return vacf / vacf[0], lag_fs


def _gk_plateau(running, max_lag_frac=0.5, plateau_frac=0.5):
    """D from the plateau of the running Green–Kubo integral.

    The integral runs over the full VACF (through its negative backscattering
    lobes) up to max_lag_frac of the trajectory; longer lags have too few time
    origins. D is the mean of the running integral over the last plateau_frac
    of that window, and its std there (m²/s) is returned as a convergence
    flag: compare it with D to see whether the running integral has levelled off.
    """
    n_max = min(len(running), max(3, int(len(running) * max_lag_frac)))
    lo = min(int(n_max * (1.0 - plateau_frac)), n_max - 2)
    window = running[lo:n_max]
    return float(window.mean()), float(window.std())


def _green_kubo_diffusion(frames, dt_fs, log_interval=1, vels=None, indices=None,
                          max_lag_frac=0.5, plateau_frac=0.5):
    """Self-diffusion from Green–Kubo: D = (1/(3N)) ∫₀^∞ ⟨Σ_i v_i(0)·v_i(t)⟩ dt.

    Returns (D, plateau std, running integral D(t) in m²/s, lag in fs).
    """
    vels = _select_velocities(frames, dt_fs, log_interval, vels, indices)
    n_atoms = vels.shape[1]
    vacf = _vacf_unnormalized(vels)
    if vacf[0] == 0.0:
        raise ValueError("VACF[0] == 0: all velocities appear to be zero.")

    dt_frame_s = dt_fs * log_interval * 1e-15
    lag_s = np.arange(len(vacf)) * dt_frame_s
    # VACF is in (Å/fs)², lag_s in s → integral in Å²·s/fs² → m²/s via ×1e10
    # (1 Å/fs)²·s = 10⁻²⁰ m² / (10⁻³⁰ s²) · s = 10¹⁰ m²/s
    running = cumulative_trapezoid(vacf, lag_s, initial=0.0) / (3.0 * n_atoms) * 1e10
    d, std = _gk_plateau(running, max_lag_frac, plateau_frac)
    return d, std, running, lag_s * 1e15


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


def _calculate_vdos_spectrum(frames, dt_fs, log_interval=1, vels=None, indices=None):
    """VDOS g(ω) = ∫ C(t) cos(ωt) dt (Fourier transform of the even VACF), Hann-windowed."""
    vels = _select_velocities(frames, dt_fs, log_interval, vels, indices)
    vacf = _vacf_unnormalized(vels)
    if vacf[0] == 0.0:
        raise ValueError("VACF[0] == 0: all velocities appear to be zero.")
    vacf /= vacf[0]

    steps = len(vacf)
    dt_frame_s = dt_fs * log_interval * 1e-15
    window = hann(2 * steps - 1)[steps - 1:]
    windowed = vacf * window
    # Full-line transform of the even function C(|t|): 2·Re Σ_{n≥0} − C(0)
    vdos_raw = (2.0 * np.fft.rfft(windowed).real - windowed[0]) * dt_frame_s

    freqs_thz = np.fft.rfftfreq(steps, d=dt_frame_s) * 1e-12

    # Windowing ripples can dip slightly below zero; EMD weights must be non-negative
    w = np.clip(vdos_raw, 0.0, None)
    vdos_pmf = w / w.sum() if w.sum() > 0 else w
    return vdos_raw, vdos_pmf, freqs_thz


# --- 2b. Element-resolved / peak utilities ---
def _element_indices(frames):
    """{element: atom indices} in order of first appearance."""
    symbols = np.array(frames[0].get_chemical_symbols())
    return {el: np.where(symbols == el)[0] for el in dict.fromkeys(symbols)}


def _partial_rdf_pairs(cfg, elements):
    """Pairs from cfg['partial_rdf_pairs'], else every element pair (incl. A–A)."""
    custom = cfg.get("partial_rdf_pairs")
    if custom:
        pairs = [tuple(p) for p in custom]
        missing = [p for p in pairs if p[0] not in elements or p[1] not in elements]
        for p in missing:
            print(f"  Skipping partial RDF {p[0]}-{p[1]}: element not in both trajectories.")
        return [p for p in pairs if p not in missing]
    return list(combinations_with_replacement(sorted(elements), 2))


def _get_partial_rdfs(frames, pairs, rmax=6.0, bins=100):
    """Partial g_ab(r) for each (a, b) pair, normalised so g_ab → 1 at large r.

    Same ideal-gas normalisation as _get_distance_distribution:
      a == b : N_a(N_a-1)/2 pairs     a != b : N_a·N_b pairs
    Returns {pair: (g_r, raw_counts)}, r_centers.
    """
    idx = _element_indices(frames)
    n_frames = len(frames)
    volume = frames[0].get_volume()  # assumed constant (NVT)

    bin_edges = np.linspace(0.1, rmax, bins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dr = bin_edges[1] - bin_edges[0]

    hists = {pair: np.zeros(bins) for pair in pairs}
    for frame in frames:
        dists = frame.get_all_distances(mic=True)
        for a, b in pairs:
            ia, ib = idx[a], idx[b]
            sub = dists[np.ix_(ia, ib)]
            sub = sub[np.triu_indices(len(ia), k=1)] if a == b else sub.ravel()
            hists[(a, b)] += np.histogram(sub, bins=bin_edges)[0]

    out = {}
    for a, b in pairs:
        na, nb = len(idx[a]), len(idx[b])
        n_pairs = na * (na - 1) / 2 if a == b else na * nb
        ideal = n_frames * n_pairs * 4.0 * np.pi * r_centers**2 * dr / volume
        with np.errstate(invalid='ignore', divide='ignore'):
            g = np.where(ideal > 0, hists[(a, b)] / ideal, 0.0)
        out[(a, b)] = (g, hists[(a, b)])
    return out, r_centers


def _get_triplet_angle_distribution(frames, triplet, rcut, bins=90):
    """Angle distribution for one a–center–b triplet (e.g. H–B–H), as a PMF.

    Both a and b must lie within rcut of the center atom. Returns
    (pmf, raw_counts, angle_centers).
    """
    a, center, b = triplet
    bin_edges = np.linspace(0, 180, bins + 1)
    angle_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hists = np.zeros(bins)

    for frame in frames:
        sym = np.array(frame.get_chemical_symbols())
        i, j, D = neighbor_list("ijD", frame, rcut)
        keep = (sym[i] == center) & np.isin(sym[j], (a, b))
        i, j, D = i[keep], j[keep], D[keep]
        D = D / np.linalg.norm(D, axis=1)[:, None]

        for c in np.unique(i):
            m = i == c
            vecs, nsym = D[m], sym[j[m]]
            cos = np.clip(vecs @ vecs.T, -1.0, 1.0)
            if a == b:
                iu = np.triu_indices(len(vecs), k=1)
                cos = cos[iu][(nsym[iu[0]] == a) & (nsym[iu[1]] == a)]
            else:
                cos = cos[np.ix_(nsym == a, nsym == b)].ravel()
            hists += np.histogram(np.degrees(np.arccos(cos)), bins=bin_edges)[0]

    pmf = hists / hists.sum() if hists.sum() > 0 else hists
    return pmf, hists, angle_centers


def _partial_adf_triplets(cfg, elements):
    """[(a, center, b, rcut)] from cfg['partial_adf_triplets'].

    Entries are ["H","B","H"] (uses partial_adf_rcut) or ["H","B","H", 1.4].
    """
    default_rcut = cfg.get("partial_adf_rcut", 1.5)
    out = []
    for t in cfg.get("partial_adf_triplets") or []:
        a, c, b = t[:3]
        if not {a, c, b} <= set(elements):
            print(f"  Skipping partial ADF {a}-{c}-{b}: element not in both trajectories.")
            continue
        out.append((a, c, b, float(t[3]) if len(t) > 3 else default_rcut))
    return out


def _first_peak(x, y, rel_prominence=0.1):
    """(position, height) of the first prominent peak; falls back to the global max."""
    if not np.any(y > 0):
        return np.nan, np.nan
    peaks, _ = find_peaks(y, prominence=rel_prominence * y.max())
    k = peaks[0] if len(peaks) else int(np.argmax(y))
    return x[k], y[k]


def _rdf_emd(x, ref_g, pred_g):
    """EMD (in units of x) between two g(r) curves treated as distributions."""
    ref_w = ref_g / ref_g.sum() if ref_g.sum() > 0 else ref_g
    pred_w = pred_g / pred_g.sum() if pred_g.sum() > 0 else pred_g
    return wasserstein_distance(x, x, u_weights=ref_w, v_weights=pred_w)


def _vacf_curve_errors(ref_vacf, ref_lag, pred_vacf, pred_lag, max_fs=None):
    """(RMSE, MAE) between normalized VACFs C(t)/C(0) on the reference lag grid.

    Compared pointwise rather than with EMD so the negative (backscattering)
    lobes count with their sign. Lags are limited to max_fs, by default half the
    shorter trajectory, beyond which few time origins contribute.
    """
    t_max = min(ref_lag[-1], pred_lag[-1])
    t_max = t_max / 2 if max_fs is None else min(max_fs, t_max)
    mask = ref_lag <= t_max
    diff = np.interp(ref_lag[mask], pred_lag, pred_vacf) - ref_vacf[mask]
    return float(np.sqrt(np.mean(diff**2))), float(np.mean(np.abs(diff)))


def _vdos_emd(ref_freq, ref_pmf, pred_freq, pred_pmf):
    """EMD (THz) on spectral shape (PMF), pred interpolated onto the ref frequency axis."""
    if len(ref_freq) != len(pred_freq) or not np.allclose(ref_freq, pred_freq):
        pred_pmf = np.interp(ref_freq, pred_freq, pred_pmf)
        pred_pmf = pred_pmf / pred_pmf.sum()
    return wasserstein_distance(ref_freq, ref_freq, u_weights=ref_pmf, v_weights=pred_pmf)


def _vdos_peaks(freqs, spectrum, cfg):
    """Peak frequencies (THz) of a VDOS, strongest `vdos_max_peaks`, sorted by frequency.

    The spectrum is Gaussian-smoothed by `vdos_peak_smooth_thz` first, since
    short trajectories give noisy spectra. Peaks must exceed
    `vdos_peak_prominence` × the maximum and lie below `vdos_peak_fmax_thz`.
    """
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    sigma = cfg.get("vdos_peak_smooth_thz", 0.5) / df
    y = gaussian_filter1d(spectrum, sigma) if sigma > 0 else spectrum
    mask = freqs <= cfg.get("vdos_peak_fmax_thz", 100.0)
    y = np.where(mask, y, 0.0)
    if y.max() <= 0:
        return np.array([])
    peaks, props = find_peaks(y, prominence=cfg.get("vdos_peak_prominence", 0.05) * y.max())
    top = peaks[np.argsort(props["prominences"])[::-1][:cfg.get("vdos_max_peaks", 10)]]
    return np.sort(freqs[top])


def _match_peaks(ref_peaks, pred_peaks, tol_thz):
    """For each reference peak, the shift to the nearest predicted peak."""
    if len(ref_peaks) == 0:
        return {"shift_mean_THz": np.nan, "shift_max_THz": np.nan, "frac_matched": np.nan}
    if len(pred_peaks) == 0:
        return {"shift_mean_THz": np.nan, "shift_max_THz": np.nan, "frac_matched": 0.0}
    shifts = np.array([np.min(np.abs(pred_peaks - p)) for p in ref_peaks])
    return {
        "shift_mean_THz": float(shifts.mean()),
        "shift_max_THz": float(shifts.max()),
        "frac_matched": float(np.mean(shifts <= tol_thz)),
    }


KB_EV_K = 8.617333262145e-5


def _expected_kinetic_energy(n_atoms, temp_k):
    """Equipartition kinetic energy (eV) for 3N degrees of freedom at temp_k."""
    return 1.5 * n_atoms * KB_EV_K * temp_k


def _plausible_kinetic_energy(ke, n_atoms, temp_k, tol=0.25):
    """True when ke is within tol of the equipartition value at temp_k."""
    if ke is None or not np.isfinite(ke) or ke <= 0:
        return False
    if temp_k is None or temp_k <= 0:
        return True
    expected = _expected_kinetic_energy(n_atoms, temp_k)
    if expected <= 0:
        return True
    ratio = ke / expected
    return (1.0 - tol) <= ratio <= (1.0 + tol) * 4.0


def _get_potential_energy(frame):
    """Extract potential energy (eV); mirrors md_plotting._get_energy for AIMD metadata."""
    if frame.calc is not None:
        if hasattr(frame.calc, "results"):
            if "energy" in frame.calc.results:
                return float(frame.calc.results["energy"])
            if "free_energy" in frame.calc.results:
                return float(frame.calc.results["free_energy"])
        try:
            return float(frame.get_potential_energy())
        except Exception:
            pass

    for key in ("energy", "REF_energy", "dft_energy", "Energy", "E", "free_energy", "E_pot_eV"):
        if key in frame.info:
            return float(frame.info[key])

    return float(frame.get_potential_energy())


def _get_forces(frame):
    """Extract forces (eV/Å); mirrors md_plotting._get_forces for AIMD metadata."""
    if frame.calc is not None:
        try:
            return np.asarray(frame.get_forces(), dtype=float)
        except Exception:
            pass

    for key in ("forces", "REF_forces", "dft_forces", "Forces", "force"):
        if key in frame.arrays:
            return np.asarray(frame.arrays[key], dtype=float)

    return np.asarray(frame.get_forces(), dtype=float)


def _kinetic_energy_from_velocities(frame):
    """Kinetic energy (eV) from the velocities array, or None if unusable."""
    try:
        v = frame.get_velocities()
    except (TypeError, ValueError):
        return None
    if v is None:
        return None
    v = np.asarray(v, dtype=float)
    if v.shape != (len(frame), 3) or np.any(np.isnan(v)):
        return None
    if np.mean(np.linalg.norm(v, axis=1)) < 1e-8:
        return None
    return float(frame.get_kinetic_energy())


def _kinetic_energy_from_metadata(frame, n_atoms):
    """Candidate total kinetic energies (eV) stored on AIMD frames."""
    candidates = []
    for key in (
        "kinetic_energy", "E_kin", "E_kin_eV", "KE", "kin_energy",
        "energy_kin_eV", "energy_kin",
    ):
        if key not in frame.info:
            continue
        val = float(frame.info[key])
        candidates.append(val)
        # Some VASP/ASE converters store per-atom KE under a total-energy key name.
        if val < 1.0:
            candidates.append(val * n_atoms)
    return candidates


def _kinetic_energy_from_temperature(frame, n_atoms):
    """Infer total KE (eV) from an instantaneous temperature in frame.info."""
    for key in ("temperature", "Temperature", "temp", "temp_inst_K", "temperature_K"):
        if key in frame.info:
            temp = float(frame.info[key])
            if temp > 0:
                return _expected_kinetic_energy(n_atoms, temp)
    return None


def _get_kinetic_energy(frame, temp_k=None):
    """Robust kinetic energy (eV) for validation.

    AIMD trajectories often carry correct KE in frame.info (from OUTCAR) while
    the velocities array is missing or in non-ASE units.  Thermo CSV plots use
    the parsed OUTCAR values, so we must not trust get_kinetic_energy() alone.
    """
    n_atoms = len(frame)
    vel_ke = _kinetic_energy_from_velocities(frame)
    meta_candidates = _kinetic_energy_from_metadata(frame, n_atoms)
    temp_ke = _kinetic_energy_from_temperature(frame, n_atoms)

    if vel_ke is not None and _plausible_kinetic_energy(vel_ke, n_atoms, temp_k):
        return vel_ke

    for cand in meta_candidates:
        if _plausible_kinetic_energy(cand, n_atoms, temp_k):
            return cand

    if temp_ke is not None and _plausible_kinetic_energy(temp_ke, n_atoms, temp_k):
        return temp_ke

    if meta_candidates:
        best = meta_candidates[0]
        if not _plausible_kinetic_energy(best, n_atoms, temp_k):
            print(
                "    Warning: Using metadata kinetic energy without equipartition "
                f"confirmation ({best:.3f} eV total)."
            )
        return best

    if vel_ke is not None:
        if not _plausible_kinetic_energy(vel_ke, n_atoms, temp_k):
            print(
                f"    Warning: Velocity-derived KE ({vel_ke:.3f} eV) is inconsistent "
                f"with 3/2 N k_B T at {temp_k} K — metric may be unreliable."
            )
        return vel_ke
    if temp_ke is not None:
        return temp_ke
    return 0.0


def _get_stress(frame):
    """Voigt stress (eV/Å³) stored on a frame, or None if the reference has none."""
    try:
        return np.asarray(frame.get_stress(voigt=True), dtype=float)
    except Exception:
        pass
    if "stress" in frame.info:
        s = np.asarray(frame.info["stress"], dtype=float)
        if s.shape == (3, 3):
            s = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        return s
    return None


def snapshot_traj_paths(cfg):
    """Reference / MLIP single-point trajectories written by run_snapshot_errors."""
    base = str(cfg.get("validation_output_csv", "validation_results.csv")).replace(".csv", "")
    return Path(f"{base}_SNAPSHOT_REF.traj"), Path(f"{base}_SNAPSHOT_MLIP.traj")


def _snapshot_errors(frames, cfg):
    """MLIP single points on reference (AIMD) frames vs. the stored DFT labels.

    Unlike the distribution metrics, this compares energies/forces on the
    *same* configurations, so it isolates potential error from sampling.
    Writes paired single-point trajectories for the parity plots.
    """
    from ase import units
    from ase.io import write
    from ase.calculators.singlepoint import SinglePointCalculator
    from md_sim.md_tools import make_calculator

    calc = make_calculator(
        cfg.get("model_type", "mace"), cfg.get("model_variant", "large"),
        cfg.get("device", "cpu"), cfg.get("head"))

    e_ref, e_pred, f_ref, f_pred, s_ref, s_pred = [], [], [], [], [], []
    ref_out, mlip_out = [], []
    for k, frame in enumerate(frames):
        er = _get_potential_energy(frame)
        fr = _get_forces(frame)
        sr = _get_stress(frame)

        atoms = frame.copy()
        atoms.calc = calc
        ep = atoms.get_potential_energy()
        fp = atoms.get_forces()
        sp = None
        if sr is not None:
            try:
                sp = atoms.get_stress(voigt=True)
            except Exception:
                sp = None

        e_ref.append(er)
        e_pred.append(ep)
        f_ref.append(fr)
        f_pred.append(fp)
        if sr is not None and sp is not None:
            s_ref.append(sr)
            s_pred.append(sp)

        for out, e, f, s in ((ref_out, er, fr, sr), (mlip_out, ep, fp, sp)):
            a = frame.copy()
            results = {"energy": e, "forces": f}
            if s is not None:
                results["stress"] = s
            a.calc = SinglePointCalculator(a, **results)
            out.append(a)

        if (k + 1) % 10 == 0 or k + 1 == len(frames):
            print(f"  Snapshot single points: {k + 1}/{len(frames)}")

    ref_path, mlip_path = snapshot_traj_paths(cfg)
    write(ref_path, ref_out)
    write(mlip_path, mlip_out)
    print(f"  Saved paired snapshots to {ref_path} and {mlip_path}")

    n_atoms = len(frames[0])
    symbols = np.array(frames[0].get_chemical_symbols())
    e_ref = np.array(e_ref) / n_atoms
    e_pred = np.array(e_pred) / n_atoms
    f_ref = np.array(f_ref)
    f_pred = np.array(f_pred)

    res = {"Snap_n_frames": len(frames)}

    # Energies: a constant offset (different DFT settings / references) is
    # reported separately from the configuration-dependent error.
    de = e_pred - e_ref
    offset = de.mean()
    res["Snap_E_MAE_meV_atom"] = np.mean(np.abs(de)) * 1000
    res["Snap_E_RMSE_meV_atom"] = np.sqrt(np.mean(de**2)) * 1000
    res["Snap_E_offset_meV_atom"] = offset * 1000
    res["Snap_E_MAE_offset_removed_meV_atom"] = np.mean(np.abs(de - offset)) * 1000

    # Forces (per Cartesian component)
    df = f_pred - f_ref
    rmse = np.sqrt(np.mean(df**2))
    res["Snap_F_MAE_eV_A"] = np.mean(np.abs(df))
    res["Snap_F_RMSE_eV_A"] = rmse
    res["Snap_F_relative_RMSE"] = rmse / np.sqrt(np.mean(f_ref**2))
    # Slope of pred vs ref forces through the origin; < 1 means the MLIP PES is
    # too soft (systematic softening), which speeds up diffusion
    res["Snap_F_parity_slope"] = np.sum(f_ref * f_pred) / np.sum(f_ref**2)

    # Force direction agreement, ignoring near-zero reference forces
    nr = np.linalg.norm(f_ref, axis=2)
    npred = np.linalg.norm(f_pred, axis=2)
    ok = (nr > 0.1) & (npred > 0)
    if ok.any():
        cos = np.sum(f_ref * f_pred, axis=2)[ok] / (nr[ok] * npred[ok])
        res["Snap_F_mean_cosine"] = float(np.mean(cos))

    for el in dict.fromkeys(symbols):
        d = df[:, symbols == el]
        res[f"Snap_F_MAE_{el}_eV_A"] = np.mean(np.abs(d))
        res[f"Snap_F_RMSE_{el}_eV_A"] = np.sqrt(np.mean(d**2))
        fr_el, fp_el = f_ref[:, symbols == el], f_pred[:, symbols == el]
        res[f"Snap_F_parity_slope_{el}"] = np.sum(fr_el * fp_el) / np.sum(fr_el**2)

    # Stress, only if the reference frames carry it
    if s_ref:
        ds = (np.array(s_pred) - np.array(s_ref)) / units.GPa
        res["Snap_Stress_MAE_GPa"] = np.mean(np.abs(ds))
        res["Snap_Pressure_MAE_GPa"] = np.mean(np.abs(ds[:, :3].mean(axis=1)))

    raw = {
        "Snap_E_ref_eV_atom": e_ref,
        "Snap_E_pred_eV_atom": e_pred,
    }
    return res, raw


_KIN_CSV_DEFAULTS = (
    "energy_kin_eV", "energy_kin", "E_kin_eV", "E_kin",
    "Kinetic_Energy", "KE", "ekin",
)
_POT_CSV_DEFAULTS = (
    "energy_pot_eV", "energy_pot", "E_pot_eV", "E_pot",
    "Potential_Energy", "PE", "epot",
)


def _find_csv_column(df, cfg, custom_key, defaults):
    """Resolve a thermo CSV column using config overrides then common names."""
    for col in list(cfg.get(custom_key, []) or []) + list(defaults):
        if col in df.columns:
            return col
    return None


def _resolve_reference_summary_csv(cfg):
    """AIMD thermo CSV: explicit path, else first entry in compare_csvs."""
    explicit = cfg.get("reference_summary_csv")
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        print(f"Warning: reference_summary_csv not found: {path}")

    compare = cfg.get("compare_csvs") or {}
    if isinstance(compare, dict) and compare:
        path = Path(next(iter(compare.values())))
        if path.is_file():
            return path
        print(f"Warning: compare_csvs reference not found: {path}")
    elif compare:
        path = Path(compare[0])
        if path.is_file():
            return path
    return None


def _resolve_predicted_summary_csv(cfg):
    pred_csv = cfg.get("summary_csv")
    if pred_csv and Path(pred_csv).is_file():
        return Path(pred_csv)
    return None


def _require_reference_summary_csv(cfg):
    path = _resolve_reference_summary_csv(cfg)
    if path is None:
        raise ValueError(
            "Reference thermo CSV required: set reference_summary_csv or compare_csvs "
            "to the AIMD summary CSV used for thermo plots (E_pot_eV, E_kin_eV)."
        )
    return path


def _load_thermo_energies_from_csv(csv_path, cfg, burn_in=0):
    """Total potential and kinetic energy time series (eV) from one thermo CSV."""
    df = pd.read_csv(csv_path)
    epot_col = _find_csv_column(df, cfg, "custom_pot_cols", _POT_CSV_DEFAULTS)
    ekin_col = _find_csv_column(df, cfg, "custom_kin_cols", _KIN_CSV_DEFAULTS)
    if epot_col is None:
        raise ValueError(
            f"No potential energy column in {csv_path}. "
            f"Available: {list(df.columns)}. Set custom_pot_cols in config."
        )
    if ekin_col is None:
        raise ValueError(
            f"No kinetic energy column in {csv_path}. "
            f"Available: {list(df.columns)}. Set custom_kin_cols in config."
        )
    epot = df[epot_col].astype(float).to_numpy()
    ekin = df[ekin_col].astype(float).to_numpy()
    if burn_in > 0:
        epot = epot[burn_in:]
        ekin = ekin[burn_in:]
    return epot, ekin, epot_col, ekin_col


def _load_side_thermo_energies(cfg, burn_in, side, frames, temp_k):
    """Epot and Ekin (eV) from thermo CSV; predicted side falls back to trajectory."""
    label = "Reference" if side == "reference" else "Predicted"
    if side == "reference":
        csv_path = _require_reference_summary_csv(cfg)
    else:
        csv_path = _resolve_predicted_summary_csv(cfg)

    if csv_path is not None:
        epot, ekin, epot_col, ekin_col = _load_thermo_energies_from_csv(
            csv_path, cfg, burn_in)
        print(f"  {label}: CSV {csv_path}")
        print(f"    Epot [{epot_col}] ({len(epot)} pts), Ekin [{ekin_col}] ({len(ekin)} pts)")
        sources = {
            "epot": f"csv:{csv_path}:{epot_col}",
            "ekin": f"csv:{csv_path}:{ekin_col}",
        }
        return epot, ekin, sources

    print(f"  Warning: {label} summary_csv not found — falling back to trajectory.")
    epot = np.array([_get_potential_energy(f) for f in frames], dtype=float)
    ekin = np.array([_get_kinetic_energy(f, temp_k) for f in frames], dtype=float)
    return epot, ekin, {"epot": "trajectory", "ekin": "trajectory"}


def _align_energy_series(ref_epot, pred_epot, ref_ekin, pred_ekin):
    """Trim energy series to a common length."""
    n = min(len(ref_epot), len(pred_epot), len(ref_ekin), len(pred_ekin))
    lengths = {
        "ref_epot": len(ref_epot),
        "pred_epot": len(pred_epot),
        "ref_ekin": len(ref_ekin),
        "pred_ekin": len(pred_ekin),
    }
    if n < max(lengths.values()):
        print(
            f"  Warning: thermo length mismatch {lengths}; using first {n} points."
        )
    return (
        ref_epot[:n], pred_epot[:n],
        ref_ekin[:n], pred_ekin[:n],
    )


def _print_kinetic_energy_diagnostics(label, ke_total, n_atoms, temp_k, source):
    """Print mean KE and flag values inconsistent with equipartition at temp_k."""
    expected = _expected_kinetic_energy(n_atoms, temp_k)
    print(
        f"    {label} Ekin: mean={ke_total.mean():.3f} eV "
        f"({ke_total.mean() / n_atoms:.5f} eV/atom), "
        f"expected≈{expected:.3f} eV at {temp_k} K [{source}]"
    )
    if expected > 0 and not _plausible_kinetic_energy(ke_total.mean(), n_atoms, temp_k):
        print(
            f"    Warning: {label} mean Ekin differs strongly from 3/2 N k_B T — "
            "check compare_csvs / summary_csv paths and custom_kin_cols."
        )


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

    kb = KB_EV_K
    n_ref = len(ref_frames[0])
    n_pred = len(pred_frames[0])
    if n_ref != n_pred:
        print(
            f"Warning: atom count mismatch — reference={n_ref}, predicted={n_pred}. "
            "Per-atom metrics use each trajectory's own atom count."
        )

    print(f"Validating {len(ref_frames)} AIMD frames vs {len(pred_frames)} MACE frames...")
    
    res = {}
    raw_data = {}

    ref_epot_total = pred_epot_total = ref_ekin_total = pred_ekin_total = None

    if cfg.get("run_energy", True) or cfg.get("run_cv", True):
        print("Thermo energies from summary CSV (same source as thermo plots):")
        try:
            ref_epot_total, ref_ekin_total, ref_src = _load_side_thermo_energies(
                cfg, burn_in, "reference", ref_frames, temp_k)
            pred_epot_total, pred_ekin_total, pred_src = _load_side_thermo_energies(
                cfg, burn_in, "predicted", pred_frames, temp_k)
        except ValueError as e:
            print(f"Error: {e}")
            return None

        ref_epot_total, pred_epot_total, ref_ekin_total, pred_ekin_total = (
            _align_energy_series(
                ref_epot_total, pred_epot_total, ref_ekin_total, pred_ekin_total)
        )

    # 1-2. Energy Distributions
    if cfg.get("run_energy", True):
        ref_epot = ref_epot_total / n_ref
        pred_epot = pred_epot_total / n_pred
        # DFT and MLIP absolute energies differ by a constant reference offset, so
        # the distributions are compared as fluctuations about each run's mean.
        res["Epot_mean_offset_eV_atom"] = pred_epot.mean() - ref_epot.mean()
        res["EMD_Epot_fluct_eV_atom"] = wasserstein_distance(
            ref_epot - ref_epot.mean(), pred_epot - pred_epot.mean())
        res["EMD_Epot_fluct_eV_raw"] = wasserstein_distance(
            ref_epot_total - ref_epot_total.mean(), pred_epot_total - pred_epot_total.mean())
        res["Epot_std_ratio"] = (
            pred_epot_total.std() / ref_epot_total.std() if ref_epot_total.std() > 0 else np.nan)

        _print_kinetic_energy_diagnostics(
            "Reference", ref_ekin_total, n_ref, temp_k, ref_src["ekin"])
        _print_kinetic_energy_diagnostics(
            "Predicted", pred_ekin_total, n_pred, temp_k, pred_src["ekin"])

        ref_ekin = ref_ekin_total / n_ref
        pred_ekin = pred_ekin_total / n_pred
        res["EMD_Ekin_eV_atom"] = wasserstein_distance(ref_ekin, pred_ekin)
        res["EMD_Ekin_eV_raw"] = wasserstein_distance(ref_ekin_total, pred_ekin_total)

        raw_data["Ref_Epot_eV_atom"] = ref_epot
        raw_data["Pred_Epot_eV_atom"] = pred_epot
        raw_data["Ref_Epot_fluct_eV_atom"] = ref_epot - ref_epot.mean()
        raw_data["Pred_Epot_fluct_eV_atom"] = pred_epot - pred_epot.mean()
        raw_data["Ref_Epot_eV_total"] = ref_epot_total
        raw_data["Pred_Epot_eV_total"] = pred_epot_total
        raw_data["Ref_Ekin_eV_atom"] = ref_ekin
        raw_data["Pred_Ekin_eV_atom"] = pred_ekin
        raw_data["Ref_Ekin_eV_total"] = ref_ekin_total
        raw_data["Pred_Ekin_eV_total"] = pred_ekin_total

    # 3. Force Magnitude Distributions
    if cfg.get("run_forces", True):
        ref_f_mag = np.linalg.norm(np.vstack([_get_forces(f) for f in ref_frames]), axis=1)
        pred_f_mag = np.linalg.norm(np.vstack([_get_forces(f) for f in pred_frames]), axis=1)
        res["EMD_Forces_eV_A"] = wasserstein_distance(ref_f_mag, pred_f_mag)
        
        raw_data["Ref_Force_Mag_eV_A"] = ref_f_mag
        raw_data["Pred_Force_Mag_eV_A"] = pred_f_mag

    # 3b. Single-point errors on reference frames (same configurations, not distributions)
    if cfg.get("run_snapshot_errors", False):
        print("Snapshot errors: MLIP single points on reference frames...")
        try:
            snap_frames = ref_frames[::max(1, int(cfg.get("snapshot_stride", 10)))]
            max_frames = cfg.get("snapshot_max_frames")
            if max_frames:
                snap_frames = snap_frames[:int(max_frames)]
            snap_res, snap_raw = _snapshot_errors(snap_frames, cfg)
            res.update(snap_res)
            raw_data.update(snap_raw)
        except Exception as e:
            print(f"Warning: Could not compute snapshot errors: {e}")

    # Structural metrics can use every Nth frame (distance matrices are O(N²) per frame)
    stride = max(1, int(cfg.get("structure_frame_stride", 1)))
    ref_struct, pred_struct = ref_frames[::stride], pred_frames[::stride]

    # Elements present in both trajectories (for element-resolved metrics)
    ref_el = _element_indices(ref_frames)
    pred_el = _element_indices(pred_frames)
    elements = [el for el in ref_el if el in pred_el]

    # 4. RDF (Radial Distribution Function)
    if cfg.get("run_rdf", True):
        ref_rdf_g, ref_rdf_raw, x_rdf = _get_distance_distribution(ref_struct)
        pred_rdf_g, pred_rdf_raw, _ = _get_distance_distribution(pred_struct)

        # Normalise g(r) to a PMF to use as weights for the Wasserstein distance.
        # The result has units of Angstroms: the average distance probability mass
        # must be transported to transform the reference g(r) into the predicted one.
        res["EMD_RDF_Angstrom"] = _rdf_emd(x_rdf, ref_rdf_g, pred_rdf_g)

        raw_data["RDF_Distance_Angstrom"] = x_rdf
        raw_data["RDF_Ref_Raw_Counts"] = ref_rdf_raw
        raw_data["RDF_Pred_Raw_Counts"] = pred_rdf_raw
        raw_data["RDF_Ref_g_r"] = ref_rdf_g
        raw_data["RDF_Pred_g_r"] = pred_rdf_g

    # 4b. Partial RDFs g_ab(r) for every element pair (or cfg['partial_rdf_pairs'])
    if cfg.get("run_partial_rdf", False):
        pairs = _partial_rdf_pairs(cfg, elements)
        rmax = cfg.get("partial_rdf_rmax", 6.0)
        bins = cfg.get("partial_rdf_bins", 100)
        ref_prdf, x_prdf = _get_partial_rdfs(ref_struct, pairs, rmax=rmax, bins=bins)
        pred_prdf, _ = _get_partial_rdfs(pred_struct, pairs, rmax=rmax, bins=bins)

        raw_data["pRDF_Distance_Angstrom"] = x_prdf
        for a, b in pairs:
            label = f"{a}-{b}"
            ref_g, pred_g = ref_prdf[(a, b)][0], pred_prdf[(a, b)][0]
            res[f"EMD_pRDF_{label}_Angstrom"] = _rdf_emd(x_prdf, ref_g, pred_g)

            # First-shell peak: the most physically interpretable number (e.g. B–H bond)
            r_ref, h_ref = _first_peak(x_prdf, ref_g)
            r_pred, h_pred = _first_peak(x_prdf, pred_g)
            res[f"pRDF_{label}_peak1_ref_A"] = r_ref
            res[f"pRDF_{label}_peak1_pred_A"] = r_pred
            res[f"pRDF_{label}_peak1_shift_A"] = abs(r_pred - r_ref)
            res[f"pRDF_{label}_peak1_height_ref"] = h_ref
            res[f"pRDF_{label}_peak1_height_pred"] = h_pred

            raw_data[f"pRDF_{label}_Ref_g_r"] = ref_g
            raw_data[f"pRDF_{label}_Pred_g_r"] = pred_g

    # 4c. Partial ADFs for specific a–center–b triplets (e.g. H–B–H in the cages)
    if cfg.get("run_partial_adf", False):
        triplets = _partial_adf_triplets(cfg, elements)
        if not triplets:
            print("  run_partial_adf is on but no partial_adf_triplets are set; skipping.")
        bins = cfg.get("partial_adf_bins", 90)
        for a, c, b, rcut in triplets:
            label = f"{a}-{c}-{b}"
            ref_p, ref_raw, x_padf = _get_triplet_angle_distribution(
                ref_struct, (a, c, b), rcut, bins)
            pred_p, pred_raw, _ = _get_triplet_angle_distribution(
                pred_struct, (a, c, b), rcut, bins)
            if ref_raw.sum() == 0 or pred_raw.sum() == 0:
                print(f"  Partial ADF {label}: no angles within {rcut} Å; increase rcut.")
                continue

            res[f"EMD_pADF_{label}_Degrees"] = wasserstein_distance(
                x_padf, x_padf, u_weights=ref_p, v_weights=pred_p)
            res[f"pADF_{label}_mean_ref_deg"] = float(np.sum(x_padf * ref_p))
            res[f"pADF_{label}_mean_pred_deg"] = float(np.sum(x_padf * pred_p))
            res[f"pADF_{label}_peak_ref_deg"] = _first_peak(x_padf, ref_p)[0]
            res[f"pADF_{label}_peak_pred_deg"] = _first_peak(x_padf, pred_p)[0]

            raw_data["pADF_Angle_Degrees"] = x_padf
            raw_data[f"pADF_{label}_Ref_Prob"] = ref_p
            raw_data[f"pADF_{label}_Pred_Prob"] = pred_p

    # 5. ADF (Angular Distribution Function)
    if cfg.get("run_adf", True):
        ref_adf_norm, ref_adf_raw, x_adf = _get_angle_distribution(ref_struct)
        pred_adf_norm, pred_adf_raw, _ = _get_angle_distribution(pred_struct)

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

    # 6. Heat Capacity (Cv) Error — same CSV Epot series as run_energy
    if cfg.get("run_cv", True):
        res["Cv_Error_eV_K_atom"] = abs(
            np.var(ref_epot_total) - np.var(pred_epot_total)
        ) / (kb * temp_k**2 * n_ref)

        raw_data["Ref_Total_Epot_eV"] = ref_epot_total
        raw_data["Pred_Total_Epot_eV"] = pred_epot_total

    # Frame spacing: AIMD may be logged at a different dt / interval than the MLIP run
    log_int = cfg.get("log_interval", 1)
    aimd_dt_fs, aimd_log_interval = _aimd_dt_params(cfg, dt_fs, log_int)
    ref_dt_frame_s = aimd_dt_fs * aimd_log_interval * 1e-15
    pred_dt_frame_s = dt_fs * log_int * 1e-15

    # Velocities are computed once per trajectory and reused by GK / VACF / VDOS
    _vel_cache = {}

    def ref_vels():
        if "ref" not in _vel_cache:
            _vel_cache["ref"] = _get_velocities(ref_frames, aimd_dt_fs, aimd_log_interval)
        return _vel_cache["ref"]

    def pred_vels():
        if "pred" not in _vel_cache:
            _vel_cache["pred"] = _get_velocities(pred_frames, dt_fs, log_int)
        return _vel_cache["pred"]

    # Species-resolved diffusion: every element with run_element_msd, otherwise only
    # cfg['diffusion_elements'] (e.g. ["Na"] so immobile B/H don't dilute the mobile ion)
    if cfg.get("run_element_msd", False):
        diff_elements = elements
    else:
        diff_elements = [el for el in (cfg.get("diffusion_elements") or []) if el in elements]
        for el in set(cfg.get("diffusion_elements") or []) - set(diff_elements):
            print(f"  Skipping diffusion for {el}: not present in both trajectories.")

    # 7. Diffusion Coefficient from MSD (Einstein relation: MSD = 6*D*t)
    if cfg.get("run_msd", True) or diff_elements:
        ref_pos = _com_free_positions(ref_frames)
        pred_pos = _com_free_positions(pred_frames)
        ref_time_s = np.arange(len(ref_frames)) * ref_dt_frame_s
        pred_time_s = np.arange(len(pred_frames)) * pred_dt_frame_s
        raw_data["MSD_Time_ps"] = ref_time_s * 1e12
        raw_data["Pred_MSD_Time_ps"] = pred_time_s * 1e12

    if cfg.get("run_msd", True):
        ref_msd_arr = _msd_from_positions(ref_pos)
        pred_msd_arr = _msd_from_positions(pred_pos)

        D_ref = _msd_diffusion_coeff(ref_msd_arr, ref_time_s)
        D_pred = _msd_diffusion_coeff(pred_msd_arr, pred_time_s)

        res["D_ref_m2_s"] = D_ref
        res["D_pred_m2_s"] = D_pred
        res["D_Error_m2_s"] = abs(D_ref - D_pred)
        # α ≈ 1 means the fit window is diffusive; otherwise D is not converged
        res["MSD_alpha_ref"] = _msd_exponent(ref_msd_arr, ref_time_s)
        res["MSD_alpha_pred"] = _msd_exponent(pred_msd_arr, pred_time_s)

        raw_data["Ref_MSD_vs_Time_A2"] = ref_msd_arr
        raw_data["Pred_MSD_vs_Time_A2"] = pred_msd_arr

    # 7a. Element-resolved MSD diffusion (+ Green–Kubo if run_green_kubo)
    if diff_elements:
        for el in diff_elements:
            ref_msd_el = _msd_from_positions(ref_pos[:, ref_el[el]])
            pred_msd_el = _msd_from_positions(pred_pos[:, pred_el[el]])
            D_ref_el = _msd_diffusion_coeff(ref_msd_el, ref_time_s)
            D_pred_el = _msd_diffusion_coeff(pred_msd_el, pred_time_s)

            res[f"D_ref_{el}_m2_s"] = D_ref_el
            res[f"D_pred_{el}_m2_s"] = D_pred_el
            res[f"D_Error_{el}_m2_s"] = abs(D_ref_el - D_pred_el)
            res[f"MSD_alpha_ref_{el}"] = _msd_exponent(ref_msd_el, ref_time_s)
            res[f"MSD_alpha_pred_{el}"] = _msd_exponent(pred_msd_el, pred_time_s)

            raw_data[f"Ref_MSD_{el}_A2"] = ref_msd_el
            raw_data[f"Pred_MSD_{el}_A2"] = pred_msd_el

    # 7b. Diffusion coefficient — Green–Kubo (VACF integral)
    if cfg.get("run_green_kubo", True):
        gk_kw = {
            "max_lag_frac": cfg.get("gk_max_lag_frac", 0.5),
            "plateau_frac": cfg.get("gk_plateau_frac", 0.5),
        }
        gk_targets = [("", None, None)] + [
            (f"_{el}", ref_el[el], pred_el[el]) for el in diff_elements]
        for suffix, ref_idx, pred_idx in gk_targets:
            try:
                D_ref_gk, std_ref, run_ref, lag_ref = _green_kubo_diffusion(
                    ref_frames, aimd_dt_fs, aimd_log_interval,
                    vels=ref_vels(), indices=ref_idx, **gk_kw)
                D_pred_gk, std_pred, run_pred, lag_pred = _green_kubo_diffusion(
                    pred_frames, dt_fs, log_int,
                    vels=pred_vels(), indices=pred_idx, **gk_kw)

                res[f"D_ref_GK{suffix}_m2_s"] = D_ref_gk
                res[f"D_pred_GK{suffix}_m2_s"] = D_pred_gk
                res[f"D_GK_Error{suffix}_m2_s"] = abs(D_ref_gk - D_pred_gk)
                # Std of the running integral over the plateau window; comparable
                # to D means it has not levelled off (D not converged)
                res[f"GK_plateau_std_ref{suffix}_m2_s"] = std_ref
                res[f"GK_plateau_std_pred{suffix}_m2_s"] = std_pred

                raw_data[f"Ref_GK{suffix}_Lag_fs"] = lag_ref
                raw_data[f"Ref_GK{suffix}_Running_D_m2_s"] = run_ref
                raw_data[f"Pred_GK{suffix}_Lag_fs"] = lag_pred
                raw_data[f"Pred_GK{suffix}_Running_D_m2_s"] = run_pred
            except Exception as e:
                print(f"Warning: Could not compute Green–Kubo diffusion{suffix}: {e}")

    # 8. Velocity Autocorrelation Function (VACF), total and per element
    vacf_targets = []
    if cfg.get("run_vacf", False):
        vacf_targets.append(("", None, None))
    if cfg.get("run_element_vacf", False):
        vacf_targets += [(f"_{el}", ref_el[el], pred_el[el]) for el in elements]

    for suffix, ref_idx, pred_idx in vacf_targets:
        try:
            ref_vacf, ref_vacf_lag = _calculate_vacf(
                ref_frames, aimd_dt_fs, aimd_log_interval, vels=ref_vels(), indices=ref_idx)
            pred_vacf, pred_vacf_lag = _calculate_vacf(
                pred_frames, dt_fs, log_int, vels=pred_vels(), indices=pred_idx)

            rmse, mae = _vacf_curve_errors(
                ref_vacf, ref_vacf_lag, pred_vacf, pred_vacf_lag,
                cfg.get("vacf_compare_max_fs"))
            res[f"VACF{suffix}_RMSE"] = rmse
            res[f"VACF{suffix}_MAE"] = mae

            raw_data[f"Ref_VACF{suffix}_Lag_fs"] = ref_vacf_lag
            raw_data[f"Ref_VACF{suffix}"] = ref_vacf
            raw_data[f"Pred_VACF{suffix}_Lag_fs"] = pred_vacf_lag
            raw_data[f"Pred_VACF{suffix}"] = pred_vacf
        except Exception as e:
            print(f"Warning: Could not compute VACF{suffix}: {e}")

    # 9. Vibrational Density of States (VDOS), total and per element, + peak analysis
    run_peaks = cfg.get("run_vdos_peaks", False)
    vdos_targets = []
    if cfg.get("run_vdos", False) or run_peaks:
        vdos_targets.append(("", None, None))
    if cfg.get("run_element_vdos", False):
        vdos_targets += [(f"_{el}", ref_el[el], pred_el[el]) for el in elements]

    if vdos_targets:
        # Frequency resolution is 1 / (trajectory length): short runs cannot resolve close peaks
        res["VDOS_resolution_ref_THz"] = 1e-12 / (len(ref_frames) * ref_dt_frame_s)
        res["VDOS_resolution_pred_THz"] = 1e-12 / (len(pred_frames) * pred_dt_frame_s)
        peak_tol = cfg.get("vdos_peak_tol_thz") or 2.0 * max(
            res["VDOS_resolution_ref_THz"], res["VDOS_resolution_pred_THz"])

    for suffix, ref_idx, pred_idx in vdos_targets:
        try:
            # AIMD and MACE trajectories can have different timesteps and output
            # intervals; aimd_dt_fs / aimd_log_interval handle the reference.
            ref_vdos, ref_vdos_pmf, ref_vdos_freq = _calculate_vdos_spectrum(
                ref_frames, aimd_dt_fs, aimd_log_interval, vels=ref_vels(), indices=ref_idx)
            pred_vdos, pred_vdos_pmf, pred_vdos_freq = _calculate_vdos_spectrum(
                pred_frames, dt_fs, log_int, vels=pred_vels(), indices=pred_idx)

            # EMD on spectral shape (PMF), not on raw g(ω) amplitudes.
            if cfg.get("run_vdos", False) or suffix:
                res[f"EMD_VDOS{suffix}_THz"] = _vdos_emd(
                    ref_vdos_freq, ref_vdos_pmf, pred_vdos_freq, pred_vdos_pmf)

            if run_peaks:
                ref_peaks = _vdos_peaks(ref_vdos_freq, ref_vdos_pmf, cfg)
                pred_peaks = _vdos_peaks(pred_vdos_freq, pred_vdos_pmf, cfg)
                match = _match_peaks(ref_peaks, pred_peaks, peak_tol)
                res[f"VDOS{suffix}_n_peaks_ref"] = len(ref_peaks)
                res[f"VDOS{suffix}_n_peaks_pred"] = len(pred_peaks)
                res[f"VDOS{suffix}_peak_shift_mean_THz"] = match["shift_mean_THz"]
                res[f"VDOS{suffix}_peak_shift_max_THz"] = match["shift_max_THz"]
                res[f"VDOS{suffix}_frac_peaks_matched"] = match["frac_matched"]
                raw_data[f"Ref_VDOS{suffix}_Peaks_THz"] = ref_peaks
                raw_data[f"Pred_VDOS{suffix}_Peaks_THz"] = pred_peaks

            raw_data[f"Ref_VDOS{suffix}_Freq_THz"] = ref_vdos_freq
            raw_data[f"Ref_VDOS{suffix}_g_omega"] = ref_vdos
            raw_data[f"Ref_VDOS{suffix}_Shape_PMF"] = ref_vdos_pmf
            raw_data[f"Pred_VDOS{suffix}_Freq_THz"] = pred_vdos_freq
            raw_data[f"Pred_VDOS{suffix}_g_omega"] = pred_vdos
            raw_data[f"Pred_VDOS{suffix}_Shape_PMF"] = pred_vdos_pmf
        except Exception as e:
            print(f"Warning: Could not compute VDOS{suffix}: {e}")

    # Print Scalar Results
    print("\n--- Validation Results ---")
    for k, v in res.items():
        print(f"{k:30s}: {v:.6g}")
        
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