"""Static (0 K) validation: relax a structure with the MLIP and compare it to a
reference (usually DFT-relaxed) structure.

Metrics
-------
- MLIP forces / pressure evaluated at the reference geometry
- Lattice parameters (a, b, c, alpha, beta, gamma, volume) after variable-cell relaxation
- Position RMSD after fixed-cell and variable-cell relaxation (total and per element)
- Bond lengths, bond by bond, grouped by element pair
- dE(fixed - variable): energy the MLIP gains by relaxing the cell
- Space group of reference / fixed / variable structures (spglib)
- Harmonic phonon DOS (finite displacements), imaginary modes, optional
  comparison to a reference phonon DOS file
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase import units
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, neighbor_list
from scipy.stats import wasserstein_distance

from md_sim import md_tools
from md_sim.config import add_config_cli, load_config, normalize_config
from md_sim.relaxation import relax_atoms

EV_TO_THZ = units._e / units._hplanck * 1e-12  # 1 eV = 241.8 THz


def _resolve_config(config=None, config_path=None, config_overrides=None):
    cfg = load_config(profile="static")
    if config_path is not None:
        cfg = {**cfg, **load_config(config_path)}
    if config is not None:
        cfg = {**cfg, **config}
    if config_overrides:
        cfg.update(config_overrides)
    cfg = normalize_config(cfg)
    cfg["fmax"] = float(cfg["fmax"])
    cfg["max_steps"] = int(cfg["max_steps"])
    return cfg


# --- Structure comparison utilities ---
def _frac_displacements(ref, test):
    """Per-atom fractional displacement test - ref, wrapped to [-0.5, 0.5)."""
    d = test.get_scaled_positions(wrap=False) - ref.get_scaled_positions(wrap=False)
    return d - np.round(d)


def _position_rmsd(ref, test):
    """RMSD (Å) of atomic positions, measured in the reference cell.

    Positions are compared in fractional coordinates mapped through the
    reference cell, so a uniform cell change is not counted as atomic
    displacement. Rigid translation is removed. Assumes identical atom order.
    """
    disp = _frac_displacements(ref, test) @ ref.cell.array
    disp -= disp.mean(axis=0)
    sq = np.sum(disp**2, axis=1)
    symbols = np.array(ref.get_chemical_symbols())
    per_el = {el: float(np.sqrt(sq[symbols == el].mean())) for el in dict.fromkeys(symbols)}
    return float(np.sqrt(sq.mean())), float(np.sqrt(sq.max())), per_el


def _bond_lengths(ref, test, mult=1.2):
    """Bond-by-bond lengths in ref and test.

    Bonds are defined on the reference structure (ASE natural cutoffs × mult)
    and the same atom pairs are measured in the test structure.
    """
    cutoffs = natural_cutoffs(ref, mult=mult)
    i, j, S = neighbor_list("ijS", ref, cutoffs)
    keep = i < j
    i, j, S = i[keep], j[keep], S[keep]

    frac_ref = ref.get_scaled_positions(wrap=False)
    dfrac = _frac_displacements(ref, test)
    vec_ref_frac = frac_ref[j] - frac_ref[i] + S
    vec_test_frac = vec_ref_frac + dfrac[j] - dfrac[i]

    d_ref = np.linalg.norm(vec_ref_frac @ ref.cell.array, axis=1)
    d_test = np.linalg.norm(vec_test_frac @ test.cell.array, axis=1)

    symbols = np.array(ref.get_chemical_symbols())
    pairs = ["-".join(sorted((a, b))) for a, b in zip(symbols[i], symbols[j])]
    return pd.DataFrame({
        "pair": pairs, "i": i, "j": j,
        "d_ref_A": d_ref, "d_mlip_A": d_test, "error_A": d_test - d_ref,
    })


def _space_group(atoms, symprec):
    try:
        import spglib
    except ImportError:
        return None
    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    return spglib.get_spacegroup(cell, symprec=symprec)


# --- Phonons ---
def _phonon_dos(atoms, calc, cfg, prefix):
    """Harmonic phonon DOS (THz) from finite displacements with ASE Phonons."""
    from ase.phonons import Phonons

    supercell = tuple(cfg.get("phonon_supercell", [1, 1, 1]))
    kpts = tuple(cfg.get("phonon_kpts") or supercell)
    ph = Phonons(atoms, calc, supercell=supercell,
                 delta=cfg.get("phonon_delta", 0.01),
                 name=f"{prefix}_phonon_cache")
    ph.run()
    ph.read(acoustic=True)

    gamma_thz = ph.band_structure([[0.0, 0.0, 0.0]], verbose=False)[0] * EV_TO_THZ

    width_ev = cfg.get("phonon_smearing_thz", 0.5) / EV_TO_THZ
    grid = ph.get_dos(kpts=kpts, verbose=False).sample_grid(
        npts=cfg.get("phonon_npts", 1000), width=width_ev)
    freqs_thz = grid.get_energies() * EV_TO_THZ
    dos = grid.get_weights() / EV_TO_THZ  # states per THz
    return gamma_thz, freqs_thz, dos


def _load_reference_dos(path):
    """Two-column reference DOS (frequency in THz, DOS); e.g. phonopy total_dos.dat."""
    data = np.loadtxt(path, comments="#", delimiter="," if str(path).endswith(".csv") else None)
    return data[:, 0], data[:, 1]


def _positive_pmf(freqs, dos):
    mask = freqs >= 0
    w = np.clip(dos[mask], 0, None)
    return freqs[mask], (w / w.sum() if w.sum() > 0 else w)


# --- Main ---
def run(config=None, config_path=None, config_overrides=None):
    cfg = _resolve_config(config, config_path, config_overrides)

    start_path = cfg["input_structure"]
    ref_path = cfg.get("reference_structure") or start_path
    prefix = cfg.get("output_prefix", "static")
    fmax, max_steps = cfg["fmax"], cfg["max_steps"]
    optimizer = cfg.get("optimizer", "lbfgs")

    print("--- Starting Static Validation ---")
    print(f"Start: {start_path} | Reference: {ref_path}")

    start = md_tools.setup_atoms_and_calculator(
        structure_path=start_path,
        model_type=cfg["model_type"],
        model_variant=cfg["model_variant"],
        device=cfg["device"],
        head=cfg.get("head"),
    )
    calc = start.calc
    ref = read(ref_path)
    n_atoms = len(ref)

    same_atoms = ref.get_chemical_symbols() == start.get_chemical_symbols()
    if not same_atoms:
        print("Warning: start and reference structures have different atoms/order — "
              "RMSD and bond comparisons will be skipped.")

    res = {}
    raw = {}

    # MLIP evaluated at the reference geometry: a perfect MLIP gives zero force/stress here
    ref_mlip = ref.copy()
    ref_mlip.calc = calc
    e_ref = ref_mlip.get_potential_energy() / n_atoms
    res["E_at_reference_eV_atom"] = e_ref
    res["Fmax_at_reference_eV_A"] = float(np.linalg.norm(ref_mlip.get_forces(), axis=1).max())
    try:
        stress = ref_mlip.get_stress(voigt=True)
        res["Pressure_at_reference_GPa"] = float(-stress[:3].mean() / units.GPa)
    except Exception as e:
        print(f"Warning: stress unavailable ({e})")

    write_traj = cfg.get("write_trajectory", True)

    # Fixed-cell relaxation
    print("\nFixed-cell relaxation...")
    fixed = start.copy()
    fixed.calc = calc
    res["Converged_fixed"] = relax_atoms(
        fixed, fmax=fmax, max_steps=max_steps, relax_cell=False, optimizer=optimizer,
        trajectory=f"{prefix}_fixed.traj" if write_traj else None)
    e_fixed = fixed.get_potential_energy() / n_atoms
    write(f"{prefix}_fixed_relaxed.vasp", fixed, format="vasp")

    # Variable-cell relaxation
    print("\nVariable-cell relaxation...")
    variable = start.copy()
    variable.calc = calc
    res["Converged_variable"] = relax_atoms(
        variable, fmax=fmax, max_steps=max_steps, relax_cell=True, optimizer=optimizer,
        trajectory=f"{prefix}_variable.traj" if write_traj else None)
    e_var = variable.get_potential_energy() / n_atoms
    write(f"{prefix}_variable_relaxed.vasp", variable, format="vasp")

    # Energies
    if cfg.get("run_relax_energy", True):
        res["E_fixed_eV_atom"] = e_fixed
        res["E_variable_eV_atom"] = e_var
        res["dE_fixed_minus_variable_meV_atom"] = (e_fixed - e_var) * 1000
        res["dE_reference_minus_fixed_meV_atom"] = (e_ref - e_fixed) * 1000

    # Lattice parameters
    if cfg.get("run_lattice", True):
        names = ["a_A", "b_A", "c_A", "alpha_deg", "beta_deg", "gamma_deg"]
        for name, r, p in zip(names, ref.cell.cellpar(), variable.cell.cellpar()):
            res[f"{name}_ref"] = r
            res[f"{name}_mlip"] = p
            res[f"{name}_err_pct"] = 100.0 * (p - r) / r
        v_ref, v_mlip = ref.get_volume() / n_atoms, variable.get_volume() / n_atoms
        res["Volume_A3_atom_ref"] = v_ref
        res["Volume_A3_atom_mlip"] = v_mlip
        res["Volume_err_pct"] = 100.0 * (v_mlip - v_ref) / v_ref

    # Position RMSD
    if cfg.get("run_position_rmsd", True) and same_atoms:
        for label, test in (("fixed", fixed), ("variable", variable)):
            rmsd, max_disp, per_el = _position_rmsd(ref, test)
            res[f"RMSD_{label}_A"] = rmsd
            res[f"Max_displacement_{label}_A"] = max_disp
            for el, val in per_el.items():
                res[f"RMSD_{label}_{el}_A"] = val

    # Bond lengths (reference vs variable-cell MLIP structure)
    if cfg.get("run_bond_lengths", True) and same_atoms:
        bonds = _bond_lengths(ref, variable, mult=cfg.get("bond_cutoff_mult", 1.2))
        if len(bonds):
            res["Bond_MAE_A"] = float(bonds["error_A"].abs().mean())
            for pair, grp in bonds.groupby("pair"):
                res[f"Bond_{pair}_n"] = len(grp)
                res[f"Bond_{pair}_ref_A"] = grp["d_ref_A"].mean()
                res[f"Bond_{pair}_mlip_A"] = grp["d_mlip_A"].mean()
                res[f"Bond_{pair}_MAE_A"] = grp["error_A"].abs().mean()
            bonds.to_csv(f"{prefix}_bonds.csv", index=False)
            print(f"Saved bond-by-bond data to {prefix}_bonds.csv")
        else:
            print("Warning: no bonds found; increase bond_cutoff_mult.")

    # Space group
    if cfg.get("run_space_group", True):
        for symprec in cfg.get("symprec_list", [0.001, 0.01, 0.1]):
            sg = {label: _space_group(a, symprec)
                  for label, a in (("ref", ref), ("fixed", fixed), ("variable", variable))}
            if sg["ref"] is None and sg["variable"] is None:
                print("Warning: spglib not installed — skipping space group.")
                break
            for label, val in sg.items():
                res[f"SpaceGroup_{label}_symprec{symprec}"] = val
            res[f"SpaceGroup_match_symprec{symprec}"] = sg["ref"] == sg["variable"]

    # Phonon DOS
    if cfg.get("run_phonon_dos", False):
        which = cfg.get("phonon_structure", "variable")
        ph_atoms = {"variable": variable, "fixed": fixed, "reference": ref}[which].copy()
        print(f"\nPhonon DOS on {which} structure ({6 * len(ph_atoms)} force calls)...")
        gamma_thz, freqs, dos = _phonon_dos(ph_atoms, calc, cfg, prefix)

        tol = cfg.get("phonon_imag_tol_thz", 0.1)
        res["Phonon_n_imaginary_Gamma"] = int(np.sum(gamma_thz < -tol))
        res["Phonon_min_freq_Gamma_THz"] = float(gamma_thz.min())
        res["Phonon_imaginary_DOS_fraction"] = float(
            np.clip(dos[freqs < -tol], 0, None).sum() / np.clip(dos, 0, None).sum())

        raw["Phonon_Freq_THz"] = freqs
        raw["Phonon_DOS_MLIP"] = dos
        raw["Phonon_Gamma_Freqs_THz"] = gamma_thz

        ref_dos_file = cfg.get("reference_phonon_dos_file")
        if ref_dos_file:
            ref_f, ref_d = _load_reference_dos(ref_dos_file)
            ref_f, ref_w = _positive_pmf(ref_f, ref_d)
            mlip_f, mlip_w = _positive_pmf(freqs, dos)
            res["EMD_PhononDOS_THz"] = wasserstein_distance(
                ref_f, mlip_f, u_weights=ref_w, v_weights=mlip_w)
            raw["Phonon_Ref_Freq_THz"] = ref_f
            raw["Phonon_Ref_DOS_PMF"] = ref_w

    print("\n--- Static Validation Results ---")
    for k, v in res.items():
        print(f"{k:40s}: {v:.6f}" if isinstance(v, (float, np.floating)) else f"{k:40s}: {v}")

    out_csv = Path(cfg.get("static_output_csv") or f"{prefix}_validation.csv")
    pd.DataFrame([res]).to_csv(out_csv, index=False)
    print(f"Saved static metrics to {out_csv}")

    if raw:
        raw_path = str(out_csv).replace(".csv", "_RAW_DATA.csv")
        pd.DataFrame({k: pd.Series(v) for k, v in raw.items()}).to_csv(raw_path, index=False)
        print(f"Saved raw static data to {raw_path}")

    return res


def main():
    parser = argparse.ArgumentParser(description="Static (0 K) validation of an MLIP against a reference structure.")
    add_config_cli(parser)
    args = parser.parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()
