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

    # 1. Standard MD Summary (Energy/Temp vs Step)
    _plot_summary(cfg, out_dir)
    
    # 2. Velocity Histograms (v_x, v_y, v_z at last step)
    atoms_csv = cfg.get("atoms_csv")
    if atoms_csv and Path(atoms_csv).exists():
        _plot_velocity_histogram(atoms_csv, out_dir / "velocity_histogram.png")

    # 3. Parity Plots (Energy/Force Accuracy)
    # Triggered only if a reference trajectory is provided in config
    ref_traj = cfg.get("reference_traj_file")
    curr_traj = cfg.get("trajectory_file")
    if ref_traj and Path(ref_traj).exists() and Path(curr_traj).exists():
        plot_parity(ref_traj, curr_traj, out_dir / "accuracy_parity")

def _plot_summary(cfg, out_dir):
    current_csv = cfg.get("summary_csv")
    if not Path(current_csv).exists(): return
    
    df = pd.read_csv(current_csv)
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Identify columns using user-defined lists from master script
    x_col = get_col(df, cfg.get("custom_x_cols"), ['step', 'time_ps'])
    pot_col = get_col(df, cfg.get("custom_pot_cols"), ['energy_pot_eV', 'E_pot'])
    temp_col = get_col(df, cfg.get("custom_temp_cols"), ['temperature_K', 'T'])

    # Plot Primary Data
    if pot_col:
        axs[0].plot(df[x_col], df[pot_col], label=cfg.get("current_run_label", "Current"), color='black', linewidth=2)
    
    # Overlay Comparison CSVs (AIMD, etc.)
    compare_csvs = cfg.get("compare_csvs", {})
    for label, path in compare_csvs.items():
        if Path(path).exists():
            cdf = pd.read_csv(path)
            cx = get_col(cdf, cfg.get("custom_x_cols"), ['step'])
            cp = get_col(cdf, cfg.get("custom_pot_cols"), ['energy_pot_eV'])
            if cp: axs[0].plot(cdf[cx], cdf[cp], label=label, linestyle='--', alpha=0.7)

    axs[0].set_ylabel("Potential Energy (eV)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    if temp_col:
        axs[1].plot(df[x_col], df[temp_col], color='red')
        axs[1].set_ylabel("Temperature (K)")
        axs[1].set_xlabel("Simulation Step")
    
    plt.tight_layout()
    plt.savefig(out_dir / "md_summary.png", dpi=300)
    plt.close()

def _plot_velocity_histogram(csv_path, output_png):
    df = pd.read_csv(csv_path)
    last_step = df['step'].max()
    df_last = df[df['step'] == last_step]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Velocity Distributions at Step {last_step}', fontsize=14)
    
    for i, (col, color) in enumerate(zip(['vx', 'vy', 'vz'], ['red', 'green', 'blue'])):
        axs[i].hist(df_last[col], bins=30, alpha=0.7, color=color, edgecolor='black')
        axs[i].set_title(f'{col.upper()} Distribution')
        axs[i].set_xlabel('Velocity')
        axs[i].set_ylabel('Atom Count')
        axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

def plot_parity(true_traj_path, pred_traj_path, output_prefix):
    """Compares reference energies/forces vs simulated predictions."""
    try:
        true_frames = read(true_traj_path, index=':')
        pred_frames = read(pred_traj_path, index=':')
    except Exception as e:
        print(f"Skipping Parity: {e}")
        return

    # Energy Parity
    n = len(true_frames[0])
    true_e = [f.get_potential_energy()/n for f in true_frames]
    pred_e = [f.get_potential_energy()/n for f in pred_frames]
    
    plt.figure(figsize=(6, 6))
    plt.scatter(true_e, pred_e, alpha=0.6, color='royalblue')
    lims = [min(true_e), max(true_e)]
    plt.plot(lims, lims, 'r--', label="Perfect Agreement")
    plt.title(f"Energy Parity (MAE: {mean_absolute_error(true_e, pred_e)*1000:.2f} meV/atom)")
    plt.xlabel("Reference Energy (eV/atom)")
    plt.ylabel("Predicted Energy (eV/atom)")
    plt.savefig(f"{output_prefix}_energy.png", dpi=300); plt.close()

    # Force Parity
    true_f = np.concatenate([f.get_forces().flatten() for f in true_frames])
    pred_f = np.concatenate([f.get_forces().flatten() for f in pred_frames])
    
    plt.figure(figsize=(6, 6))
    plt.scatter(true_f, pred_f, alpha=0.1, s=1, color='darkorange')
    lims = [min(true_f), max(true_f)]
    plt.plot(lims, lims, 'r--')
    plt.title(f"Force Parity (MAE: {mean_absolute_error(true_f, pred_f):.3f} eV/Å)")
    plt.xlabel("Reference Force (eV/Å)")
    plt.ylabel("Predicted Force (eV/Å)")
    plt.savefig(f"{output_prefix}_force.png", dpi=300); plt.close()