import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read

try:
    from sklearn.metrics import mean_absolute_error
except ImportError:
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

# --- Robust Column Searching ---
def get_col(df, custom_list, default_list):
    """Searches for columns in order of preference."""
    search_list = (custom_list if custom_list else []) + default_list
    for col in search_list:
        if col in df.columns: return col
    return None

def generate_plots(cfg):
    """The central entry point. Orchestrates all requested plots."""
    out_dir = Path(cfg.get("output_dir", "./plots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Thermodynamics Plots (Temp, Kin/Pot, Tot) with comparisons
    _plot_thermo(cfg, out_dir)
    
    # 2. RDF Plot (Radial Distribution Function)
    _plot_rdf(cfg, out_dir)

    # 3. Parity Plots (Energy/Force Accuracy)
    ref_traj = cfg.get("reference_traj_file")
    curr_traj = cfg.get("trajectory_file")
    
    if ref_traj and Path(ref_traj).exists() and curr_traj and Path(curr_traj).exists():
        plot_parity(ref_traj, curr_traj, str(out_dir / "accuracy_parity"))
    elif ref_traj:
        print(f"Skipping parity plots. Ensure both trajectories exist.\n"
              f"  Ref:  {ref_traj}\n  Pred: {curr_traj}")
    # 2.5 Velocity Histograms (v_x, v_y, v_z at last step)
    atoms_csv = cfg.get("atoms_csv")
    if atoms_csv and Path(atoms_csv).exists():
        _plot_velocity_histogram(atoms_csv, out_dir / "velocity_histogram.png")

def _load_data_sources(cfg):
    """Loads the main run and any comparison CSVs, resolving columns."""
    sources = []
    main_csv = cfg.get("summary_csv")
    if main_csv and Path(main_csv).exists():
        sources.append({
            "label": cfg.get("current_run_label", "Current Run"),
            "df": pd.read_csv(main_csv),
            "linestyle": "-",
            "alpha": 0.9,
            "linewidth": 2
        })
    
    compare_csvs = cfg.get("compare_csvs", {})
    if isinstance(compare_csvs, dict):
        for label, path in compare_csvs.items():
            if Path(path).exists():
                sources.append({
                    "label": label,
                    "df": pd.read_csv(path),
                    "linestyle": "--",
                    "alpha": 0.7,
                    "linewidth": 1.5
                })
    return sources

def _plot_thermo(cfg, out_dir):
    """Plots separated thermodynamic properties vs step, handling different column names."""
    sources = _load_data_sources(cfg)
    if not sources:
        print("Warning: No valid CSVs found for thermo plots.")
        return

    # Prepare the figures
    fig_temp, ax_temp = plt.subplots(figsize=(10, 6))
    fig_tot, ax_tot = plt.subplots(figsize=(10, 6))
    fig_kinpot, ax_kinpot = plt.subplots(figsize=(10, 6))

    plotted_temp = plotted_tot = plotted_kinpot = False

    # Get column preferences from config
    x_custom = cfg.get("custom_x_cols", [])
    temp_custom = cfg.get("custom_temp_cols", [])
    tot_custom = cfg.get("custom_tot_cols", [])
    kin_custom = cfg.get("custom_kin_cols", [])
    pot_custom = cfg.get("custom_pot_cols", [])

    for s in sources:
        df = s["df"]
        lbl = s["label"]
        ls = s["linestyle"]
        alpha = s["alpha"]
        lw = s["linewidth"]

        x_col = get_col(df, x_custom, ['step', 'time_ps'])
        if not x_col: continue

        # 1. Temp vs Step
        t_col = get_col(df, temp_custom, ['temperature_K', 'temp_inst_K', 'T'])
        if t_col:
            ax_temp.plot(df[x_col], df[t_col], label=lbl, linestyle=ls, alpha=alpha, linewidth=lw)
            plotted_temp = True

        # 2. Total Energy vs Step
        tot_col = get_col(df, tot_custom, ['energy_tot_eV', 'E_tot_eV', 'energy_eV', 'etot'])
        if tot_col:
            ax_tot.plot(df[x_col], df[tot_col], label=lbl, linestyle=ls, alpha=alpha, linewidth=lw)
            plotted_tot = True

        # 3. Kinetic & Potential Energy vs Step
        k_col = get_col(df, kin_custom, ['energy_kin_eV', 'E_kin_eV', 'ekin'])
        p_col = get_col(df, pot_custom, ['energy_pot_eV', 'E_pot_eV', 'epot_eV', 'epot'])
        
        if k_col and p_col:
            ax_kinpot.plot(df[x_col], df[p_col], label=f"{lbl} (Pot)", linestyle=ls, alpha=alpha, linewidth=lw)
            ax_kinpot.plot(df[x_col], df[k_col], label=f"{lbl} (Kin)", linestyle=':', alpha=alpha, linewidth=lw)
            plotted_kinpot = True

    # Save Temp Plot
    if plotted_temp:
        ax_temp.set_xlabel("Simulation Step")
        ax_temp.set_ylabel("Temperature (K)")
        ax_temp.set_title("Temperature vs. Simulation Step")
        ax_temp.legend()
        ax_temp.grid(True, alpha=0.3)
        fig_temp.tight_layout()
        fig_temp.savefig(out_dir / "temp_vs_step.png", dpi=300)
    plt.close(fig_temp)

    # Save Tot Plot
    if plotted_tot:
        ax_tot.set_xlabel("Simulation Step")
        ax_tot.set_ylabel("Total Energy (eV)")
        ax_tot.set_title("Total Energy vs. Simulation Step")
        ax_tot.legend()
        ax_tot.grid(True, alpha=0.3)
        fig_tot.tight_layout()
        fig_tot.savefig(out_dir / "tot_energy_vs_step.png", dpi=300)
    plt.close(fig_tot)

    # Save Kin/Pot Plot
    if plotted_kinpot:
        ax_kinpot.set_xlabel("Simulation Step")
        ax_kinpot.set_ylabel("Energy (eV)")
        ax_kinpot.set_title("Kinetic and Potential Energy vs. Simulation Step")
        ax_kinpot.legend()
        ax_kinpot.grid(True, alpha=0.3)
        fig_kinpot.tight_layout()
        fig_kinpot.savefig(out_dir / "energies_vs_step.png", dpi=300)
    plt.close(fig_kinpot)


def _plot_rdf(cfg, out_dir, rmax=6.0, bins=100):
    """Calculates and plots a pseudo-RDF from the trajectory."""
    traj_path = cfg.get("trajectory_file")
    if not traj_path or not Path(traj_path).exists():
        return
        
    print(f"Calculating RDF from {traj_path}...")
    try:
        frames = read(str(traj_path), index="::10") # read every 10th to speed up
    except Exception as e:
        print(f"Error reading trajectory for RDF: {e}")
        return

    if not frames:
        return

    hists = np.zeros(bins)
    for frame in frames:
        dists = frame.get_all_distances(mic=True)
        dists = dists[np.triu_indices_from(dists, k=1)]
        h, edges = np.histogram(dists, bins=bins, range=(0.1, rmax))
        hists += h
        
    if np.sum(hists) == 0:
        return
        
    hists = hists / np.sum(hists)
    r_centers = (edges[1:] + edges[:-1]) / 2.0
    
    plt.figure(figsize=(8, 5))
    plt.plot(r_centers, hists, color='purple', linewidth=2)
    plt.xlabel("Distance (Å)")
    plt.ylabel("Probability")
    plt.title("Radial Distribution Function (Pseudo-RDF)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "rdf_plot.png", dpi=300)
    plt.close()


def _get_energy(frame):
    """Robustly extract energy from a frame, checking calculators, results, and info."""
    if frame.calc is not None:
        if hasattr(frame.calc, 'results'):
            if 'energy' in frame.calc.results:
                return float(frame.calc.results['energy'])
            if 'free_energy' in frame.calc.results:
                return float(frame.calc.results['free_energy'])
        try:
            return frame.get_potential_energy()
        except Exception:
            pass

    for key in ['energy', 'REF_energy', 'dft_energy', 'Energy', 'E', 'free_energy']:
        if key in frame.info:
            return float(frame.info[key])

    raise ValueError(f"Could not find energy. Available info: {list(frame.info.keys())}")


def _get_forces(frame):
    """Robustly extract forces from a frame."""
    if frame.calc is not None:
        if hasattr(frame.calc, 'results') and 'forces' in frame.calc.results:
            return np.array(frame.calc.results['forces'])
        try:
            return frame.get_forces()
        except Exception:
            pass

    for key in ['forces', 'REF_forces', 'dft_forces', 'Forces', 'force']:
        if key in frame.arrays:
            return np.array(frame.arrays[key])

    raise ValueError(f"Could not find forces. Available arrays: {list(frame.arrays.keys())}")


def plot_parity(true_traj_path, pred_traj_path, output_prefix):
    """Generates energy and force parity plots comparing reference to prediction."""
    print(f"Reading trajectories for parity:\n  Ref:  {true_traj_path}\n  Pred: {pred_traj_path}")
    
    try:
        true_frames = read(str(true_traj_path), index=":")
        pred_frames = read(str(pred_traj_path), index=":")
    except Exception as e:
        print(f"ERROR reading trajectory files: {e}")
        return

    if len(true_frames) != len(pred_frames):
        n = min(len(true_frames), len(pred_frames))
        print(f"Warning: Frame mismatch ({len(true_frames)} vs {len(pred_frames)}). Using first {n} frames.")
        true_frames, pred_frames = true_frames[:n], pred_frames[:n]

    if not true_frames:
        return

    n_atoms = len(true_frames[0])

    # --- 1. Energy Parity ---
    true_e, pred_e = [], []
    for i, (tf, pf) in enumerate(zip(true_frames, pred_frames)):
        try:
            true_e.append(_get_energy(tf) / n_atoms)
            pred_e.append(_get_energy(pf) / n_atoms)
        except ValueError as e:
            pass 

    if len(true_e) > 0:
        mae = mean_absolute_error(true_e, pred_e) * 1000 # to meV
        
        plt.figure(figsize=(6, 6))
        plt.scatter(true_e, pred_e, alpha=0.6, color='royalblue', edgecolors='k', s=20)
        
        lims = [min(min(true_e), min(pred_e)), max(max(true_e), max(pred_e))]
        plt.plot(lims, lims, 'r--', alpha=0.7, label=f'MAE: {mae:.2f} meV/atom')
        
        plt.xlabel("Reference Energy (eV/atom)")
        plt.ylabel("Predicted Energy (eV/atom)")
        plt.title("Energy Parity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_energy.png", dpi=300)
        plt.close()
        print(f"Saved: {output_prefix}_energy.png")

    # --- 2. Force Parity ---
    true_f_all, pred_f_all = [], []
    for i, (tf, pf) in enumerate(zip(true_frames, pred_frames)):
        try:
            true_f_all.append(_get_forces(tf).flatten())
            pred_f_all.append(_get_forces(pf).flatten())
        except ValueError as e:
            pass 

    if len(true_f_all) > 0:
        true_f = np.concatenate(true_f_all)
        pred_f = np.concatenate(pred_f_all)
        mae_f = mean_absolute_error(true_f, pred_f)

        plt.figure(figsize=(6, 6))
        plt.scatter(true_f, pred_f, alpha=0.1, s=1, color='darkorange')
        
        lims = [min(min(true_f), min(pred_f)), max(max(true_f), max(pred_f))]
        plt.plot(lims, lims, 'k--', alpha=0.5, label=f'MAE: {mae_f:.3f} eV/Å')
        
        plt.xlabel("Reference Force (eV/Å)")
        plt.ylabel("Predicted Force (eV/Å)")
        plt.title("Force Component Parity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_forces.png", dpi=300)
        plt.close()
        print(f"Saved: {output_prefix}_forces.png")
        
def _plot_velocity_histogram(atoms_csv, out_path):
    """Plots a histogram of velocities to check Maxwell-Boltzmann distribution."""
    if not Path(atoms_csv).exists():
        return
        
    df = pd.read_csv(atoms_csv)
    if not all(col in df.columns for col in ['vx', 'vy', 'vz']):
        return
        
    # Get velocities of the last step to check equilibration
    last_step = df['step'].max()
    df_last = df[df['step'] == last_step]
    
    plt.figure(figsize=(8, 5))
    plt.hist(df_last['vx'], bins=50, alpha=0.5, label='vx', density=True)
    plt.hist(df_last['vy'], bins=50, alpha=0.5, label='vy', density=True)
    plt.hist(df_last['vz'], bins=50, alpha=0.5, label='vz', density=True)
    
    plt.xlabel("Velocity (Å/fs)")
    plt.ylabel("Density")
    plt.title(f"Velocity Distribution at Step {last_step}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()